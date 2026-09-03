#!/usr/bin/env python3
"""Sample Ascend NPU power, AI-core utilization, and HBM usage to CSV."""

from __future__ import annotations

import argparse
import csv
import ipaddress
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


BOARD_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s*$")
CHIP_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
DETAIL_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s+"       # AI-core utilization
    r"(\d+)\s*/\s*(\d+)\s+"          # device memory used / total
    r"(\d+)\s*/\s*(\d+)\s*$"         # HBM used / total
)

CSV_FIELDS = [
    "timestamp",
    "host",
    "ip",
    "npu_id",
    "chip_id",
    "phy_id",
    "name",
    "health",
    "bus_id",
    "board_power_w",
    "aicore_pct",
    "hbm_used_mb",
    "hbm_total_mb",
    "hbm_usage_pct",
    "sample_duration_s",
    "error",
]


@dataclass
class Board:
    npu_id: int
    name: str
    health: str
    power_w: Optional[float]


def numeric_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def parse_npu_smi(text: str) -> list[dict[str, object]]:
    """Parse the two-line-per-chip table produced by ``npu-smi info``."""
    devices: list[dict[str, object]] = []
    current_board: Optional[Board] = None
    last_power_by_board: dict[int, float] = {}

    for line in text.splitlines():
        if not line.startswith("|"):
            continue

        columns = line.strip().strip("|").split("|")
        if len(columns) != 3:
            continue
        left, middle, right = (column.strip() for column in columns)

        board_match = BOARD_RE.fullmatch(left)
        if board_match and middle in {"OK", "Warning", "Alarm", "Critical"}:
            right_fields = right.split()
            power = numeric_or_none(right_fields[0]) if right_fields else None
            npu_id = int(board_match.group(1))
            if power is not None:
                last_power_by_board[npu_id] = power
            else:
                power = last_power_by_board.get(npu_id)
            current_board = Board(
                npu_id=npu_id,
                name=board_match.group(2),
                health=middle,
                power_w=power,
            )
            continue

        chip_match = CHIP_RE.fullmatch(left)
        detail_match = DETAIL_RE.fullmatch(right)
        if current_board is None or chip_match is None or detail_match is None:
            continue

        hbm_used = int(detail_match.group(4))
        hbm_total = int(detail_match.group(5))
        devices.append(
            {
                "npu_id": current_board.npu_id,
                "chip_id": int(chip_match.group(1)),
                "phy_id": int(chip_match.group(2)),
                "name": current_board.name,
                "health": current_board.health,
                "bus_id": middle,
                # npu-smi reports one power value per board, not per chip.
                "board_power_w": current_board.power_w,
                "aicore_pct": float(detail_match.group(1)),
                "hbm_used_mb": hbm_used,
                "hbm_total_mb": hbm_total,
                "hbm_usage_pct": round(100.0 * hbm_used / hbm_total, 3)
                if hbm_total
                else None,
            }
        )

    if not devices:
        raise ValueError("no NPU device rows could be parsed")
    return devices


def discover_ip(explicit_ip: Optional[str]) -> str:
    if explicit_ip:
        return explicit_ip

    candidates: list[str] = []
    try:
        output = subprocess.run(
            ["hostname", "-I"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        candidates.extend(output.split())
    except (OSError, subprocess.SubprocessError):
        pass

    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4 and not address.is_loopback and not address.is_link_local:
            return candidate

    # The CLI override is preferable on a multi-interface host. This fallback
    # still gives a safe directory name if no usable address is discoverable.
    return socket.gethostname()


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def collect(command: str, timeout: float) -> tuple[list[dict[str, object]], float]:
    started = time.monotonic()
    result = subprocess.run(
        [command, "info"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"npu-smi exited with {result.returncode}: {message}")
    return parse_npu_smi(result.stdout), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Ascend NPU utilization in one CSV row per physical chip."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/dpc-zhouy/zhouy/npu_monitoring_log/logs"),
        help="Root directory (default: /dpc-zhouy/zhouy/npu_monitoring_log/logs)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between the start of samples (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout for each npu-smi call in seconds (default: 30)",
    )
    parser.add_argument(
        "--ip",
        help="IP used as the directory name; recommended on multi-interface hosts",
    )
    parser.add_argument(
        "--npu-smi",
        default="npu-smi",
        help="npu-smi executable or absolute path (default: npu-smi)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take one sample and exit (useful for testing)",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    worker_ip = discover_ip(args.ip)
    hostname = socket.gethostname()
    start_time = datetime.now().astimezone()
    output_dir = args.output_root / safe_component(worker_ip)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{start_time.strftime('%Y%m%d_%H%M%S')}.csv"

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(f"Writing NPU samples to {log_path}", flush=True)
    next_sample = time.monotonic()

    # Line buffering makes each completed CSV row visible immediately.
    with log_path.open("x", newline="", encoding="utf-8", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()

        while not stopping:
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                devices, elapsed = collect(args.npu_smi, args.timeout)
                for device in devices:
                    writer.writerow(
                        {
                            "timestamp": timestamp,
                            "host": hostname,
                            "ip": worker_ip,
                            **device,
                            "sample_duration_s": round(elapsed, 3),
                            "error": "",
                        }
                    )
            except Exception as exc:  # Keep monitoring after transient tool failures.
                elapsed = max(0.0, time.monotonic() - next_sample)
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "host": hostname,
                        "ip": worker_ip,
                        "sample_duration_s": round(elapsed, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"[{timestamp}] collection failed: {exc}", file=sys.stderr, flush=True)

            if args.once:
                break

            # Fixed-rate scheduling avoids accumulating subprocess runtime as drift.
            next_sample += args.interval
            delay = next_sample - time.monotonic()
            if delay < 0:
                missed = int((-delay) // args.interval) + 1
                next_sample += missed * args.interval
                delay = next_sample - time.monotonic()

            # Short waits allow SIGTERM from a service manager to stop promptly.
            while delay > 0 and not stopping:
                time.sleep(min(delay, 1.0))
                delay = next_sample - time.monotonic()

    print(f"Monitoring stopped at {datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}; log saved at {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

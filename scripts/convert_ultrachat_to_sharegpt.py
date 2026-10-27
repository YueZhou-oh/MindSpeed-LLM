#!/usr/bin/env python3
"""Convert openbmb/UltraChat into multi-turn ShareGPT JSONL."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


SUPPORTED_SUFFIXES = (".json", ".jsonl", ".json.gz", ".jsonl.gz")


def open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def discover_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")

    files = [
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.name.endswith(SUPPORTED_SUFFIXES)
    ]
    if not files:
        raise FileNotFoundError(f"no JSON/JSONL files found under: {input_path}")
    return sorted(files)


def iter_local_records(input_path: Path) -> Iterator[dict[str, Any]]:
    for path in discover_files(input_path):
        with open_text(path) as stream:
            for line_number, line in enumerate(stream, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"WARNING: skipping invalid JSON at {path}:{line_number}: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if isinstance(record, dict):
                    yield record


def iter_hf_records(dataset_name: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face mode requires `pip install datasets`."
        ) from exc

    return load_dataset(dataset_name, split=split, streaming=True)


def convert_record(
    record: dict[str, Any], odd_turn_policy: str
) -> tuple[dict[str, Any] | None, str | None]:
    turns = record.get("data")
    if not isinstance(turns, list) or len(turns) < 2:
        return None, "invalid_data"

    # Do not silently remove an interior turn: doing so would reverse all later roles.
    if any(not isinstance(turn, str) or not turn.strip() for turn in turns):
        return None, "empty_or_nonstring_turn"

    if len(turns) % 2:
        if odd_turn_policy == "reject":
            return None, "odd_turn_count"
        turns = turns[:-1]  # Drop an unmatched final user turn.

    if len(turns) < 2:
        return None, "invalid_data"

    conversations = [
        {
            "from": "human" if index % 2 == 0 else "gpt",
            "value": turn.strip(),
        }
        for index, turn in enumerate(turns)
    ]

    converted: dict[str, Any] = {"conversations": conversations}
    if record.get("id") is not None:
        converted["id"] = str(record["id"])
    return converted, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert UltraChat's alternating `data` list into multi-turn "
            "ShareGPT `conversations` JSONL."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="Downloaded UltraChat JSON/JSONL file or directory (recursive)",
    )
    source.add_argument(
        "--hf-dataset",
        default=None,
        help="Stream a Hugging Face dataset, e.g. openbmb/UltraChat",
    )
    parser.add_argument("--split", default="train", help="HF split (default: train)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument(
        "--odd-turn-policy",
        choices=("drop", "reject"),
        default="drop",
        help="Handle an unmatched final user turn (default: drop)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after writing this many samples; useful for testing",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10000, help="Progress interval"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        print(f"ERROR: output already exists: {output}; use --overwrite", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.input is not None:
        records = iter_local_records(args.input.expanduser().resolve())
    else:
        records = iter_hf_records(args.hf_dataset, args.split)

    read_count = 0
    written_count = 0
    rejected: dict[str, int] = {}

    with output.open("w", encoding="utf-8") as sink:
        for record in records:
            read_count += 1
            converted, reason = convert_record(record, args.odd_turn_policy)
            if converted is None:
                rejected[reason or "unknown"] = rejected.get(reason or "unknown", 0) + 1
                continue

            sink.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written_count += 1

            if args.log_interval > 0 and written_count % args.log_interval == 0:
                print(
                    f"read={read_count:,} written={written_count:,} "
                    f"rejected={sum(rejected.values()):,}",
                    file=sys.stderr,
                    flush=True,
                )

            if args.max_samples is not None and written_count >= args.max_samples:
                break

    print(
        json.dumps(
            {
                "read": read_count,
                "written": written_count,
                "rejected": sum(rejected.values()),
                "rejection_reasons": rejected,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

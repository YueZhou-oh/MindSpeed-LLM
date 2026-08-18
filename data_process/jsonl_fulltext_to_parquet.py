#!/usr/bin/env python3
"""Extract ``fullText`` from JSONL files into sharded raw-text Parquet files.

Each input line must be one complete JSON object. This stage deliberately does
not tokenize: MindSpeed-LLM's ``preprocess_data.py`` performs Qwen tokenization
when it creates the indexed ``.bin/.idx`` dataset.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq


RAW_SCHEMA = pa.schema([pa.field("text", pa.string(), nullable=False)])


@dataclass
class FileStats:
    path: str
    lines: int = 0
    kept: int = 0
    empty_or_short: int = 0
    malformed: int = 0
    missing_fulltext: int = 0
    shards: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/dpc-zhouy/zhouy/papers/core-EN", type=Path)
    parser.add_argument("--output-dir", default="/dpc-zhouy/zhouy/papers/core-parquet", type=Path)
    parser.add_argument("--field", default="fullText")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--rows-per-shard", type=int, default=500_000)
    parser.add_argument("--row-group-size", type=int, default=10_000)
    parser.add_argument("--min-chars", type=int, default=200)
    return parser.parse_args()


def normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    # NFC preserves scientific symbols more faithfully than compatibility
    # normalization (NFKC), while canonicalizing Unicode composition.
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").replace("\n..\n..\n..", "")
    text = text.strip()     #.strip('\n..\n..\n..')
    return text or None


def process_one_file(task: tuple[int, str, dict]) -> FileStats:
    file_index, input_name, cfg = task
    input_path = Path(input_name)
    output_dir = Path(cfg["output_dir"])
    stats = FileStats(path=input_name)

    writer: pq.ParquetWriter | None = None
    rows_in_shard = 0
    shard_index = 0

    def open_writer() -> pq.ParquetWriter:
        nonlocal shard_index
        name = output_dir / f"part-{file_index:07d}-{shard_index:07d}.parquet"
        shard_index += 1
        return pq.ParquetWriter(
            name,
            schema=RAW_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
        )

    def emit(texts: list[str]) -> None:
        nonlocal writer, rows_in_shard
        if not texts:
            return

        table = pa.Table.from_arrays(
            [pa.array(texts, type=pa.string())],
            schema=RAW_SCHEMA,
        )
        offset = 0
        while offset < table.num_rows:
            if writer is None:
                writer = open_writer()
                rows_in_shard = 0
                stats.shards += 1

            room = cfg["rows_per_shard"] - rows_in_shard
            piece = table.slice(offset, room)
            writer.write_table(piece, row_group_size=cfg["row_group_size"])
            offset += piece.num_rows
            rows_in_shard += piece.num_rows
            stats.kept += piece.num_rows

            if rows_in_shard >= cfg["rows_per_shard"]:
                writer.close()
                writer = None

    pending: list[str] = []
    with input_path.open("rb") as handle:
        for line in handle:
            stats.lines += 1
            if not line.strip():
                stats.empty_or_short += 1
                continue
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                stats.malformed += 1
                continue
            if not isinstance(obj, dict) or cfg["field"] not in obj:
                stats.missing_fulltext += 1
                continue

            abstract = normalize_text(obj.get("abstract"))
            fulltext = normalize_text(obj.get(cfg["field"]))

            text = "\n\n".join(
                value
                for value in (abstract, fulltext)
                if isinstance(value, str) and value
            )

            if text is None or len(text) < cfg["min_chars"]:
                stats.empty_or_short += 1
                continue

            pending.append(text)
            if len(pending) >= cfg["batch_size"]:
                emit(pending)
                pending.clear()
        emit(pending)

    if writer is not None:
        writer.close()
    return stats


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.parquet")):
        raise SystemExit(f"Refusing to mix with existing Parquet files in: {output_dir}")

    files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise SystemExit(f"No .json or .jsonl files found under: {input_dir}")

    cfg = {
        "output_dir": str(output_dir),
        "field": args.field,
        "batch_size": args.batch_size,
        "rows_per_shard": args.rows_per_shard,
        "row_group_size": args.row_group_size,
        "min_chars": args.min_chars,
    }
    tasks = [(index, str(path), cfg) for index, path in enumerate(files)]
    totals = FileStats(path="TOTAL")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(tasks))) as pool:
        for stats in pool.imap_unordered(process_one_file, tasks):
            print(
                f"{stats.path}: lines={stats.lines:,} kept={stats.kept:,} "
                f"malformed={stats.malformed:,} missing={stats.missing_fulltext:,} "
                f"short={stats.empty_or_short:,}",
                flush=True,
            )
            for name in (
                "lines",
                "kept",
                "empty_or_short",
                "malformed",
                "missing_fulltext",
                "shards",
            ):
                setattr(totals, name, getattr(totals, name) + getattr(stats, name))

    # Rename after workers finish so filenames follow Hugging Face's shard style.
    parts = sorted(output_dir.glob("part-*.parquet"))
    width = max(5, len(str(max(0, len(parts) - 1))))
    for index, old_path in enumerate(parts):
        new_path = output_dir / f"train-{index:0{width}d}-of-{len(parts):0{width}d}.parquet"
        old_path.replace(new_path)

    print(
        f"TOTAL: lines={totals.lines:,} kept={totals.kept:,} "
        f"malformed={totals.malformed:,} missing={totals.missing_fulltext:,} "
        f"short={totals.empty_or_short:,} shards={len(parts):,}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
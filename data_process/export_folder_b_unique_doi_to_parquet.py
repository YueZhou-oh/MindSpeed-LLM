#!/usr/bin/env python3
"""
Export Folder-B JSONL records selected by a DOI comparison SQLite database.

The input files are parsed in parallel. Every worker opens the existing SQLite
database read-only and writes its own Parquet part, so there are no concurrent
database writes and no concurrent writes to the same Parquet file.
"""

import argparse
import gzip
import io
import json
import multiprocessing
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote


GZIP_MAGIC = b"\x1f\x8b"
SQLITE_QUERY_BATCH_LIMIT = 900
STAT_KEYS = (
    "files",
    "total_lines",
    "empty_lines",
    "invalid_json",
    "non_dict_json",
    "missing_doi",
    "doi_selected",
    "selected_missing_full_text",
    "written_rows",
    "read_errors",
)

SELECTION_SQL = {
    # DOI occurs exactly once in Folder B, whether or not it also occurs in A.
    "unique-in-b": "count_b = 1",
    # DOI occurs exactly once in B and does not occur in A.
    "only-in-b": "count_b > 0 AND count_a = 0",
    # All lines in B with a DOI represented in the comparison database.
    "all-in-b": "count_b > 0",
}


def empty_stats():
    return {key: 0 for key in STAT_KEYS}


def add_stats(total, partial):
    for key in STAT_KEYS:
        total[key] += partial[key]


def normalize_doi(value):
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)

    doi = value.strip().lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if doi.startswith(prefix):
            doi = doi[len(prefix):].strip()
            break
    return doi or None


def extract_doi(record):
    doi = record.get("doi")
    if doi:
        return normalize_doi(doi)

    for key in ("externalids", "externalIds", "external_ids"):
        external_ids = record.get(key)
        if isinstance(external_ids, dict):
            doi = external_ids.get("doi") or external_ids.get("DOI")
            if doi:
                return normalize_doi(doi)
    return None


def normalize_text(value):
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return "\n\n".join(parts) or None
    return None


def extract_full_text(record):
    content = record.get("content")
    if isinstance(content, dict):
        return normalize_text(content.get("text"))
    return None


def open_text_file(path):
    """Detect gzip using magic bytes, including incorrectly named .gz.gz data."""
    raw = open(path, "rb")
    magic = raw.read(2)
    raw.seek(0)
    if magic == GZIP_MAGIC:
        compressed = gzip.GzipFile(fileobj=raw, mode="rb")
        return io.TextIOWrapper(compressed, encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def find_files(folder, database_path, output_dir):
    database_path = Path(database_path).resolve()
    output_dir = Path(output_dir).resolve()
    database_files = {
        Path(str(database_path) + suffix)
        for suffix in ("", "-wal", "-shm", "-journal")
    }
    for path in Path(folder).resolve().rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in database_files:
            continue
        if resolved == output_dir or output_dir in resolved.parents:
            continue
        yield resolved


def distribute_files(files, worker_count):
    """Balance 10-GB-scale files by their on-disk sizes."""
    weighted = []
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        weighted.append((size, str(path)))

    weighted.sort(reverse=True)
    groups = [[] for _ in range(worker_count)]
    group_sizes = [0] * worker_count
    for size, path in weighted:
        index = min(range(worker_count), key=group_sizes.__getitem__)
        groups[index].append(path)
        group_sizes[index] += size
    return [group for group in groups if group]


def open_readonly_database(database_path):
    uri = f"file:{quote(str(Path(database_path).resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=120)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-65536")  # 64 MiB per worker
    conn.execute("PRAGMA mmap_size=268435456")  # 256 MiB virtual mapping
    return conn


def selected_dois(conn, dois, selection):
    """Return the subset accepted by the selected count_a/count_b rule."""
    unique_dois = list(dict.fromkeys(dois))
    accepted = set()
    condition = SELECTION_SQL[selection]

    for start in range(0, len(unique_dois), SQLITE_QUERY_BATCH_LIMIT):
        chunk = unique_dois[start:start + SQLITE_QUERY_BATCH_LIMIT]
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            f"SELECT doi FROM doi_counts WHERE doi IN ({placeholders}) "
            f"AND {condition}"
        )
        accepted.update(row[0] for row in conn.execute(sql, chunk))
    return accepted


def import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required. Install it with: pip install pyarrow"
        ) from error
    return pa, pq


def process_file_group(task):
    (
        worker_id,
        file_paths,
        database_path,
        output_path,
        selection,
        query_batch_size,
        parquet_batch_size,
        compression,
        keep_empty,
        log_interval,
    ) = task

    pa, pq = import_pyarrow()
    conn = open_readonly_database(database_path)
    stats = empty_stats()
    writer = None
    candidates = []
    output_dois = []
    output_texts = []

    schema = pa.schema([
        pa.field("doi", pa.string(), nullable=False),
        pa.field("full_text", pa.large_string(), nullable=keep_empty),
    ])
    compression_value = None if compression == "none" else compression

    def write_output_batch():
        nonlocal writer
        if not output_dois:
            return
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema,
                compression=compression_value,
                use_dictionary=["doi"],
            )
        table = pa.Table.from_arrays(
            [
                pa.array(output_dois, type=pa.string()),
                pa.array(output_texts, type=pa.large_string()),
            ],
            schema=schema,
        )
        writer.write_table(table, row_group_size=parquet_batch_size)
        stats["written_rows"] += len(output_dois)
        output_dois.clear()
        output_texts.clear()

    def filter_candidate_batch():
        if not candidates:
            return
        accepted = selected_dois(conn, [item[0] for item in candidates], selection)
        for doi, full_text in candidates:
            if doi not in accepted:
                continue
            stats["doi_selected"] += 1
            if full_text is None:
                stats["selected_missing_full_text"] += 1
                if not keep_empty:
                    continue
            output_dois.append(doi)
            output_texts.append(full_text)
            if len(output_dois) >= parquet_batch_size:
                write_output_batch()
        candidates.clear()

    try:
        for file_path in file_paths:
            stats["files"] += 1
            try:
                with open_text_file(file_path) as file:
                    for line in file:
                        stats["total_lines"] += 1
                        if not line.strip():
                            stats["empty_lines"] += 1
                            continue
                        try:
                            record = json.loads(line)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            stats["invalid_json"] += 1
                            continue
                        if not isinstance(record, dict):
                            stats["non_dict_json"] += 1
                            continue

                        doi = extract_doi(record)
                        if doi is None:
                            stats["missing_doi"] += 1
                            continue

                        candidates.append((doi, extract_full_text(record)))
                        if len(candidates) >= query_batch_size:
                            filter_candidate_batch()

                        if log_interval > 0 and stats["total_lines"] % log_interval == 0:
                            print(
                                f"[worker {worker_id}] lines={stats['total_lines']:,}, "
                                f"selected={stats['doi_selected']:,}, "
                                f"written={stats['written_rows']:,}, file={file_path}",
                                flush=True,
                            )
            except (OSError, EOFError) as error:
                stats["read_errors"] += 1
                print(f"[READ ERROR] worker {worker_id}, {file_path}: {error}", flush=True)

        filter_candidate_batch()
        write_output_batch()
    finally:
        if writer is not None:
            writer.close()
        conn.close()

    part_path = output_path if stats["written_rows"] else None
    return worker_id, part_path, stats


def validate_database(database_path):
    conn = open_readonly_database(database_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(doi_counts)")}
        required = {"doi", "count_a", "count_b"}
        if not required.issubset(columns):
            raise ValueError(
                f"doi_counts must contain {sorted(required)}; found {sorted(columns)}"
            )
    finally:
        conn.close()


def prepare_output_directory(output_dir, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite to replace existing part files."
        )

    if overwrite:
        unexpected = [
            path for path in existing
            if not (path.is_file() and (path.name.startswith("part-") and
                                        path.suffix == ".parquet" or
                                        path.name == "_summary.json"))
        ]
        if unexpected:
            names = ", ".join(path.name for path in unexpected[:5])
            raise RuntimeError(
                f"Refusing to remove unrelated output contents: {names}"
            )
        for path in existing:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Export selected Folder-B records to a Parquet dataset."
    )
    parser.add_argument("--folder-b", default="/dpc-zhouy/zhouy/papers/arxiv_extracted", help="Folder B containing JSONL files")
    parser.add_argument("--database", default="/dpc-zhouy/zhouy/papers/doi_comparison.sqlite", help="Existing DOI comparison SQLite DB")
    parser.add_argument("--output-dir", default="/dpc-zhouy/zhouy/papers/only_in_s2orc_parquet", help="Output Parquet dataset directory")
    parser.add_argument(
        "--selection",
        choices=tuple(SELECTION_SQL),
        default="only-in-b",
        help=(
            "unique-in-b: count_b=1; only-in-b: count_b>0 and count_a=0; "
            "all-in-b: count_b>0"
        ),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=900,
        help="Records checked in each SQLite lookup batch (maximum: 900)",
    )
    parser.add_argument("--parquet-batch-size", type=int, default=20000)
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy", "gzip", "none"),
        default="zstd",
    )
    parser.add_argument(
        "--keep-empty-full-text",
        action="store_true",
        help="Write selected DOI rows even when full_text is missing",
    )
    parser.add_argument("--log-interval", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 1 <= args.query_batch_size <= SQLITE_QUERY_BATCH_LIMIT:
        parser.error("--query-batch-size must be between 1 and 900")
    if args.parquet_batch_size < 1:
        parser.error("--parquet-batch-size must be at least 1")
    if args.log_interval < 0:
        parser.error("--log-interval cannot be negative")

    folder_b = Path(args.folder_b).resolve()
    database_path = Path(args.database).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not folder_b.is_dir():
        raise ValueError(f"Folder B does not exist: {folder_b}")
    if not database_path.is_file():
        raise ValueError(f"SQLite database does not exist: {database_path}")

    import_pyarrow()
    validate_database(database_path)
    prepare_output_directory(output_dir, args.overwrite)

    files = list(find_files(folder_b, database_path, output_dir))
    if not files:
        raise ValueError(f"No files found under Folder B: {folder_b}")

    worker_count = min(args.workers, len(files))
    groups = distribute_files(files, worker_count)
    tasks = []
    for worker_id, group in enumerate(groups, start=1):
        output_path = output_dir / f"part-{worker_id:05d}.parquet"
        tasks.append((
            worker_id, group, str(database_path), str(output_path),
            args.selection, args.query_batch_size, args.parquet_batch_size,
            args.compression, args.keep_empty_full_text, args.log_interval,
        ))

    print(
        f"Folder B: {len(files):,} files; workers: {worker_count}; "
        f"selection: {args.selection}",
        flush=True,
    )

    total_stats = empty_stats()
    parts = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = [executor.submit(process_file_group, task) for task in tasks]
        for future in as_completed(futures):
            worker_id, part_path, stats = future.result()
            add_stats(total_stats, stats)
            if part_path:
                parts.append(Path(part_path).name)
            print(
                f"[finished worker {worker_id}] files={stats['files']:,}, "
                f"lines={stats['total_lines']:,}, selected={stats['doi_selected']:,}, "
                f"written={stats['written_rows']:,}",
                flush=True,
            )

    parts.sort()
    summary = {
        "folder_b": str(folder_b),
        "database": str(database_path),
        "selection": args.selection,
        "workers": worker_count,
        "compression": args.compression,
        "keep_empty_full_text": args.keep_empty_full_text,
        "parquet_parts": parts,
        "stats": total_stats,
    }
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nEXPORT RESULTS")
    for key in STAT_KEYS:
        print(f"{key + ':':34s} {total_stats[key]:,}")
    print(f"Parquet parts:                     {len(parts):,}")
    print(f"Output directory:                  {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare two JSONL-style datasets by DOI using multiprocessing and SQLite.

Workers never write to the same database. Each worker parses a balanced group
of files into a private SQLite shard, and the parent process merges the shards
into the final database. This avoids both excessive RAM use and SQLite lock
contention.
"""

import argparse
import gzip
import io
import json
import multiprocessing
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


GZIP_MAGIC = b"\x1f\x8b"
STAT_KEYS = (
    "files", "total_lines", "empty_lines", "valid_json_dicts",
    "invalid_json", "non_dict_json", "missing_doi", "doi_lines",
    "read_errors", "no_full_text",
)


def empty_stats():
    return {key: 0 for key in STAT_KEYS}


def add_stats(total, partial):
    for key in STAT_KEYS:
        total[key] += partial[key]


def normalize_doi(value):
    """Normalize DOI strings before comparison."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)

    doi = value.strip().lower()
    prefixes = (
        "https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
        "http://dx.doi.org/", "doi:",
    )
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):].strip()
            break
    return doi or None


def extract_doi(record):
    """Read DOI from common CORE/S2ORC field layouts."""
    doi = record.get("doi")
    if doi:
        return normalize_doi(doi)

    for key in ("externalids", "externalIds", "external_ids"):
        external_ids = record.get(key)
        if not isinstance(external_ids, dict):
            continue
        doi = external_ids.get("doi") or external_ids.get("DOI")
        if doi:
            return normalize_doi(doi)
    return None


def has_full_text(record):
    """Check common CORE/S2ORC full-text field layouts."""
    for key in ("fullText", "fulltext", "full_text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if value is not None and not isinstance(value, str):
            return True

    content = record.get("content")
    if isinstance(content, dict):
        value = content.get("text")
        if isinstance(value, str) and value.strip():
            return True
        if value is not None and not isinstance(value, str):
            return True
    return False


def open_text_file(path):
    """Open plain-text or gzip data based on magic bytes, not the suffix."""
    raw = open(path, "rb")
    magic = raw.read(2)
    raw.seek(0)
    if magic == GZIP_MAGIC:
        binary_stream = gzip.GzipFile(fileobj=raw, mode="rb")
        return io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def find_files(folder, database_path):
    folder = Path(folder).resolve()
    database_path = Path(database_path).resolve()
    ignored_names = {
        database_path.name, database_path.name + "-wal",
        database_path.name + "-shm", database_path.name + "-journal",
    }
    for path in folder.rglob("*"):
        if path.is_file() and path.name not in ignored_names:
            yield path.resolve()


def distribute_files(files, worker_count):
    """Greedily balance workers using compressed/on-disk file sizes."""
    weighted_files = []
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        weighted_files.append((size, str(path)))

    weighted_files.sort(reverse=True)
    groups = [[] for _ in range(worker_count)]
    group_sizes = [0] * worker_count
    for size, path in weighted_files:
        index = min(range(worker_count), key=group_sizes.__getitem__)
        groups[index].append(path)
        group_sizes[index] += size
    return [group for group in groups if group]


def create_shard_database(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        "CREATE TABLE doi_counts "
        "(doi TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
    )
    return conn


def write_worker_batch(conn, doi_counter):
    if not doi_counter:
        return
    conn.executemany(
        """
        INSERT INTO doi_counts(doi, count) VALUES (?, ?)
        ON CONFLICT(doi) DO UPDATE SET count = count + excluded.count
        """,
        doi_counter.items(),
    )
    conn.commit()


def process_file_group(task):
    """Worker entry point: parse assigned files and create one DOI shard."""
    worker_id, file_paths, shard_path, batch_size, log_interval, folder_label = task
    stats = empty_stats()
    doi_counter = Counter()
    pending_doi_lines = 0
    conn = create_shard_database(shard_path)

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

                        stats["valid_json_dicts"] += 1
                        if not has_full_text(record):
                            stats["no_full_text"] += 1

                        doi = extract_doi(record)
                        if doi is None:
                            stats["missing_doi"] += 1
                            continue

                        stats["doi_lines"] += 1
                        doi_counter[doi] += 1
                        pending_doi_lines += 1

                        if pending_doi_lines >= batch_size:
                            write_worker_batch(conn, doi_counter)
                            doi_counter.clear()
                            pending_doi_lines = 0

                        if log_interval > 0 and stats["total_lines"] % log_interval == 0:
                            print(
                                f"[Folder {folder_label}, worker {worker_id}] "
                                f"lines={stats['total_lines']:,}, "
                                f"DOI lines={stats['doi_lines']:,}, "
                                f"current file={file_path}",
                                flush=True,
                            )
            except Exception as error:
                stats["read_errors"] += 1
                print(
                    f"[READ ERROR] Folder {folder_label}, worker {worker_id}, "
                    f"{file_path}: {error}",
                    flush=True,
                )

        write_worker_batch(conn, doi_counter)
    finally:
        conn.close()
    return worker_id, shard_path, stats


def create_database(database_path):
    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-262144")  # Approximately 256 MiB
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS doi_counts (
            doi TEXT PRIMARY KEY,
            count_a INTEGER NOT NULL DEFAULT 0,
            count_b INTEGER NOT NULL DEFAULT 0
        ) WITHOUT ROWID
        """
    )
    conn.commit()
    return conn


def merge_shard(conn, shard_path, folder_label):
    """Merge one worker database into the final database in the parent."""
    conn.execute("ATTACH DATABASE ? AS shard", (str(shard_path),))
    try:
        if folder_label == "A":
            conn.execute(
                """
                INSERT INTO doi_counts(doi, count_a, count_b)
                SELECT doi, count, 0 FROM shard.doi_counts WHERE 1
                ON CONFLICT(doi) DO UPDATE SET
                    count_a = count_a + excluded.count_a
                """
            )
        else:
            conn.execute(
                """
                INSERT INTO doi_counts(doi, count_a, count_b)
                SELECT doi, 0, count FROM shard.doi_counts WHERE 1
                ON CONFLICT(doi) DO UPDATE SET
                    count_b = count_b + excluded.count_b
                """
            )
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE shard")


def process_folder(folder, folder_label, conn, database_path, batch_size,
                   log_interval, workers, temp_root):
    files = list(find_files(folder, database_path))
    stats = empty_stats()
    if not files:
        return stats

    actual_workers = min(workers, len(files))
    groups = distribute_files(files, actual_workers)
    folder_temp = Path(temp_root) / f"folder_{folder_label.lower()}"
    folder_temp.mkdir(parents=True, exist_ok=True)

    tasks = []
    for index, group in enumerate(groups, start=1):
        shard_path = folder_temp / f"worker_{index:03d}.sqlite"
        tasks.append((index, group, str(shard_path), batch_size,
                      log_interval, folder_label))

    print(
        f"Folder {folder_label}: {len(files):,} files, "
        f"{actual_workers} worker processes",
        flush=True,
    )
    # "spawn" prevents workers from inheriting the parent's open final-DB
    # connection, which is unsafe with SQLite when Linux defaults to "fork".
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=actual_workers,
        mp_context=mp_context,
    ) as executor:
        futures = [executor.submit(process_file_group, task) for task in tasks]
        for future in as_completed(futures):
            worker_id, shard_path, worker_stats = future.result()
            add_stats(stats, worker_stats)
            merge_shard(conn, shard_path, folder_label)
            print(
                f"[Folder {folder_label}] worker {worker_id} finished: "
                f"files={worker_stats['files']:,}, "
                f"lines={worker_stats['total_lines']:,}, "
                f"DOI lines={worker_stats['doi_lines']:,}",
                flush=True,
            )
    return stats


def calculate_results(conn):
    return conn.execute(
        """
        SELECT
            SUM(CASE WHEN count_a > 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN count_b > 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN count_a > 0 AND count_b > 0 THEN 1 ELSE 0 END),
            SUM(CASE WHEN count_a > 0 AND count_b > 0
                     THEN MIN(count_a, count_b) ELSE 0 END)
        FROM doi_counts
        """
    ).fetchone()


def print_folder_stats(name, stats, unique_dois):
    duplicate_doi_lines = stats["doi_lines"] - unique_dois
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Files:                    {stats['files']:,}")
    print(f"Total lines:              {stats['total_lines']:,}")
    print(f"Empty lines:              {stats['empty_lines']:,}")
    print(f"Valid JSON dictionaries:  {stats['valid_json_dicts']:,}")
    print(f"Invalid JSON lines:       {stats['invalid_json']:,}")
    print(f"Non-dictionary JSON:      {stats['non_dict_json']:,}")
    print(f"Missing/empty DOI:        {stats['missing_doi']:,}")
    print(f"Lines containing DOI:     {stats['doi_lines']:,}")
    print(f"Unique normalized DOIs:   {unique_dois:,}")
    print(f"Duplicate DOI lines:      {duplicate_doi_lines:,}")
    print(f"Files with read errors:   {stats['read_errors']:,}")
    print(f"Papers with no full text: {stats['no_full_text']:,}")


def remove_existing_database(database_path):
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = Path(str(database_path) + suffix)
        if path.is_file():
            path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Compare JSON-lines datasets using DOI with multiprocessing."
    )
    parser.add_argument("--folder-a", default="/dpc-zhouy/zhouy/papers/core-EN",
                        help="First input folder")
    parser.add_argument("--folder-b", default="/dpc-zhouy/zhouy/papers/arxiv_extracted",
                        help="Second input folder")
    parser.add_argument("--database", default="/dpc-zhouy/zhouy/papers/doi_comparison.sqlite",
                        help="Final SQLite database")
    parser.add_argument("--workers", type=int,
                        default=32,
                        help="Parser processes per folder (default: min(8, CPU count))")
    parser.add_argument("--batch-size", type=int, default=100_000,
                        help="DOIs accumulated per worker before shard insertion")
    parser.add_argument("--log-interval", type=int, default=500_000,
                        help="Print worker progress every N lines; 0 disables it")
    parser.add_argument("--temp-dir", default="/dpc-zhouy/zhouy/papers/doi_temp",
                        help="Parent directory for temporary worker databases")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep worker shard databases after completion")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing comparison database")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.log_interval < 0:
        parser.error("--log-interval cannot be negative")

    folder_a = Path(args.folder_a).resolve()
    folder_b = Path(args.folder_b).resolve()
    database_path = Path(args.database).resolve()
    if not folder_a.is_dir():
        raise ValueError(f"Folder A does not exist: {folder_a}")
    if not folder_b.is_dir():
        raise ValueError(f"Folder B does not exist: {folder_b}")

    if database_path.exists():
        if args.overwrite:
            remove_existing_database(database_path)
        else:
            raise FileExistsError(
                f"Database already exists: {database_path}\n"
                "Use --overwrite or provide another --database path."
            )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = Path(args.temp_dir).resolve() if args.temp_dir else None
    if temp_parent:
        temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(
        prefix="doi_compare_", dir=str(temp_parent) if temp_parent else None
    ))

    conn = create_database(database_path)
    completed = False
    try:
        print(f"Processing Folder A: {folder_a}", flush=True)
        stats_a = process_folder(
            folder_a, "A", conn, database_path, args.batch_size,
            args.log_interval, args.workers, temp_root,
        )
        print(f"\nProcessing Folder B: {folder_b}", flush=True)
        stats_b = process_folder(
            folder_b, "B", conn, database_path, args.batch_size,
            args.log_interval, args.workers, temp_root,
        )

        unique_a, unique_b, shared_unique, matched_occurrences = calculate_results(conn)
        unique_a, unique_b = unique_a or 0, unique_b or 0
        shared_unique = shared_unique or 0
        matched_occurrences = matched_occurrences or 0

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print_folder_stats("Folder A", stats_a, unique_a)
        print_folder_stats("Folder B", stats_b, unique_b)
        print("\nCombined")
        print("--------")
        print(f"Total lines in both folders: {stats_a['total_lines'] + stats_b['total_lines']:,}")
        print(f"Total DOI lines:             {stats_a['doi_lines'] + stats_b['doi_lines']:,}")
        print(f"Shared unique DOI samples:   {shared_unique:,}")
        print(f"Only in Folder A:            {unique_a - shared_unique:,}")
        print(f"Only in Folder B:            {unique_b - shared_unique:,}")
        print(f"One-to-one matched lines:    {matched_occurrences:,}")
        completed = True
    finally:
        conn.close()
        if args.keep_temp:
            print(f"Temporary worker databases: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    if completed:
        print(f"\nComparison database: {database_path}")


if __name__ == "__main__":
    main()
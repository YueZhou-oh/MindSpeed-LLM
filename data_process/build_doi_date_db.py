#!/usr/bin/env python3
"""Build a resumable SQLite DOI -> publication-date database from Parquet.

Stage 1 scans only the configured DOI column using multiple processes. Each
process writes a private SQLite shard, avoiding shared-writer contention. The
shards are then merged into the final indexed database.

Stage 2 resolves pending DOIs through batched OpenAlex API requests and commits
every response immediately. Interrupted or rate-limited runs can be resumed by
running the same command again, optionally with --skip-extract.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode
from urllib.request import Request, urlopen

import pyarrow.parquet as pq


VERSION = "1.0.0"
OPENALEX_URL = "https://api.openalex.org/works"
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)
os.environ["OPENALEX_API_KEY"] = "CYlsPAfQUjKy2LfaCvercE"

@dataclass
class ExtractStats:
    shard: str
    files: int = 0
    rows: int = 0
    valid_dois: int = 0
    invalid_or_missing_dois: int = 0
    unique_dois: int = 0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default='only_in_s2orc_parquet_deduplicated', type=Path)
    parser.add_argument("--db", default='/dpc-zhouy/zhouy/papers/doi_n_publish_date.sqlite', type=Path)
    parser.add_argument("--doi-column", default="doi")
    parser.add_argument("--extract-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--extract-batch-size", type=int, default=65_536)
    parser.add_argument("--insert-batch-size", type=int, default=10_000)
    parser.add_argument("--temp-dir", default='/dpc-zhouy/cache', type=Path)
    parser.add_argument(
        "--openalex-api-key",
        default=os.environ.get("OPENALEX_API_KEY"),
        help="Prefer environment variable OPENALEX_API_KEY",
    )
    parser.add_argument("--api-workers", type=int, default=8)
    parser.add_argument("--api-batch-size", type=int, default=100)
    parser.add_argument("--requests-per-second", type=float, default=8.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--user-agent", default="doi-date-builder/1.0")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-resolve", action="store_true")
    parser.add_argument(
        "--retry-status",
        action="append",
        choices=("not_found", "no_date", "error"),
        default=[],
        help="Reset this status to pending before API resolution; repeat as needed",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep per-process extraction databases after a successful merge",
    )
    return parser.parse_args()


def normalize_doi(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    doi = unquote(value).strip().lower()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi\s*:\s*", "", doi, flags=re.I)
    doi = doi.strip().rstrip(".,;")
    return doi if DOI_RE.fullmatch(doi) else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA busy_timeout=60000")


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS doi_dates (
            doi TEXT PRIMARY KEY,
            publication_date TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            source TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at TEXT
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_doi_dates_status ON doi_dates(status)"
    )
    connection.commit()


def insert_dois(
    connection: sqlite3.Connection,
    values: set[str],
) -> int:
    if not values:
        return 0
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO doi_dates(doi, status) VALUES (?, 'pending')",
        ((doi,) for doi in values),
    )
    connection.commit()
    return connection.total_changes - before


def extract_group(task: tuple[int, list[str], dict]) -> ExtractStats:
    group_index, relative_names, cfg = task
    shard_path = Path(cfg["temp_dir"]) / f"doi-part-{group_index:04d}.sqlite"
    stats = ExtractStats(shard=str(shard_path))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(shard_path, timeout=60.0)
        configure_database(connection)
        create_schema(connection)
        pending: set[str] = set()
        for relative_name in relative_names:
            path = Path(cfg["input_dir"]) / relative_name
            parquet_file = pq.ParquetFile(path)
            schema = parquet_file.schema_arrow
            if schema.get_field_index(cfg["doi_column"]) < 0:
                raise ValueError(f"{path}: column {cfg['doi_column']!r} not found")
            stats.files += 1
            for batch in parquet_file.iter_batches(
                batch_size=cfg["extract_batch_size"],
                columns=[cfg["doi_column"]],
                use_threads=False,
            ):
                values = batch.column(0).to_pylist()
                stats.rows += len(values)
                for value in values:
                    doi = normalize_doi(value)
                    if doi is None:
                        stats.invalid_or_missing_dois += 1
                        continue
                    stats.valid_dois += 1
                    pending.add(doi)
                if len(pending) >= cfg["insert_batch_size"]:
                    stats.unique_dois += insert_dois(connection, pending)
                    pending.clear()
        stats.unique_dois += insert_dois(connection, pending)
    except Exception as exc:
        stats.error = f"{type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()
    return stats


def partition_files(files: list[Path], workers: int) -> list[list[Path]]:
    groups: list[list[Path]] = [[] for _ in range(min(workers, len(files)))]
    # Greedy size balancing prevents one worker receiving all very large files.
    loads = [0] * len(groups)
    for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True):
        index = min(range(len(groups)), key=loads.__getitem__)
        groups[index].append(path)
        loads[index] += path.stat().st_size
    return groups


def merge_shards(database: Path, shard_paths: list[Path], keep_temp: bool) -> None:
    connection = sqlite3.connect(database, timeout=60.0)
    configure_database(connection)
    create_schema(connection)
    try:
        for index, shard_path in enumerate(shard_paths):
            alias = f"part{index}"
            connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(shard_path),))
            connection.execute(
                f"""
                INSERT OR IGNORE INTO main.doi_dates(
                    doi, publication_date, status, source, attempts, last_error, updated_at
                )
                SELECT doi, publication_date, status, source, attempts, last_error, updated_at
                FROM {alias}.doi_dates
                """
            )
            connection.commit()
            connection.execute(f"DETACH DATABASE {alias}")
    finally:
        connection.close()
    if not keep_temp:
        for shard_path in shard_paths:
            shard_path.unlink(missing_ok=True)
            Path(str(shard_path) + "-wal").unlink(missing_ok=True)
            Path(str(shard_path) + "-shm").unlink(missing_ok=True)


class RateLimiter:
    def __init__(self, requests_per_second: float):
        if requests_per_second <= 0:
            raise ValueError("requests-per-second must be positive")
        self.interval = 1.0 / requests_per_second
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if delay:
            time.sleep(delay)


def openalex_publication_date(work: dict) -> str | None:
    value = work.get("publication_date")
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    year = work.get("publication_year")
    if isinstance(year, int) and 1 <= year <= 9999:
        return f"{year:04d}"
    return None


def fetch_openalex_batch(
    dois: list[str],
    cfg: dict,
    limiter: RateLimiter,
) -> tuple[list[str], dict[str, str | None]]:
    parameters = {
        "filter": "doi:" + "|".join(f"https://doi.org/{doi}" for doi in dois),
        "select": "doi,publication_date,publication_year",
        "per_page": str(len(dois)),
        "api_key": cfg["api_key"],
    }
    url = OPENALEX_URL + "?" + urlencode(parameters)
    headers = {"User-Agent": cfg["user_agent"], "Accept": "application/json"}
    last_error: Exception | None = None
    for attempt in range(cfg["max_retries"] + 1):
        limiter.wait()
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=cfg["request_timeout"]) as response:
                payload = json.load(response)
            results: dict[str, str | None] = {}
            for work in payload.get("results", []):
                doi = normalize_doi(work.get("doi"))
                if doi:
                    results[doi] = openalex_publication_date(work)
            return dois, results
        except HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403}:
                raise RuntimeError(
                    "OpenAlex authentication failed; check OPENALEX_API_KEY"
                ) from exc
            if exc.code in {400, 414}:
                if len(dois) == 1:
                    # The DOI passed local syntax validation but is not accepted
                    # by this endpoint; record it as not found and continue.
                    return dois, {}
                midpoint = len(dois) // 2
                _, left = fetch_openalex_batch(dois[:midpoint], cfg, limiter)
                _, right = fetch_openalex_batch(dois[midpoint:], cfg, limiter)
                return dois, left | right
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60.0, 2.0 ** attempt)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            delay = min(60.0, 2.0 ** attempt)
        if attempt < cfg["max_retries"]:
            time.sleep(delay)
    raise RuntimeError(f"OpenAlex request failed after retries: {last_error}")


def reset_retry_statuses(connection: sqlite3.Connection, statuses: list[str]) -> None:
    for status in statuses:
        connection.execute(
            "UPDATE doi_dates SET status='pending', last_error=NULL WHERE status=?",
            (status,),
        )
    connection.commit()


def write_api_result(
    connection: sqlite3.Connection,
    requested: list[str],
    results: dict[str, str | None],
) -> Counter:
    now = utc_now()
    counts = Counter()
    rows = []
    for doi in requested:
        if doi not in results:
            status = "not_found"
            publication_date = None
        elif results[doi] is None:
            status = "no_date"
            publication_date = None
        else:
            status = "resolved"
            publication_date = results[doi]
        counts[status] += 1
        rows.append((publication_date, status, now, doi))
    connection.executemany(
        """
        UPDATE doi_dates
        SET publication_date=?, status=?, source='openalex', attempts=attempts+1,
            last_error=NULL, updated_at=?
        WHERE doi=?
        """,
        rows,
    )
    connection.commit()
    return counts


def resolve_pending(database: Path, args: argparse.Namespace) -> Counter:
    if not args.openalex_api_key:
        raise SystemExit(
            "OpenAlex API key required. Set OPENALEX_API_KEY or pass --openalex-api-key."
        )
    if not 1 <= args.api_batch_size <= 100:
        raise SystemExit("--api-batch-size must be between 1 and 100")
    if args.api_workers < 1:
        raise SystemExit("--api-workers must be positive")

    connection = sqlite3.connect(database, timeout=60.0)
    configure_database(connection)
    create_schema(connection)
    reset_retry_statuses(connection, args.retry_status)
    limiter = RateLimiter(args.requests_per_second)
    cfg = {
        "api_key": args.openalex_api_key,
        "user_agent": args.user_agent,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
    }
    totals = Counter()
    try:
        with ThreadPoolExecutor(max_workers=args.api_workers) as executor:
            while True:
                limit = args.api_batch_size * args.api_workers
                pending = [
                    row[0]
                    for row in connection.execute(
                        "SELECT doi FROM doi_dates WHERE status='pending' LIMIT ?",
                        (limit,),
                    ).fetchall()
                ]
                if not pending:
                    break
                batches = [
                    pending[start:start + args.api_batch_size]
                    for start in range(0, len(pending), args.api_batch_size)
                ]
                futures = {
                    executor.submit(fetch_openalex_batch, batch, cfg, limiter): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    requested, results = future.result()
                    counts = write_api_result(connection, requested, results)
                    totals.update(counts)
                    totals["api_requests"] += 1
                processed = sum(totals[s] for s in ("resolved", "not_found", "no_date"))
                print(
                    f"API: processed={processed:,} resolved={totals['resolved']:,} "
                    f"not_found={totals['not_found']:,} no_date={totals['no_date']:,} "
                    f"requests={totals['api_requests']:,}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        connection.close()
    return totals


def database_status_counts(database: Path) -> dict[str, int]:
    connection = sqlite3.connect(database)
    try:
        return {
            status: count
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM doi_dates GROUP BY status ORDER BY status"
            )
        }
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    database = args.db.resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    extraction_results: list[ExtractStats] = []

    if not args.skip_extract:
        if args.input_dir is None:
            raise SystemExit("--input-dir is required unless --skip-extract is used")
        input_dir = args.input_dir.resolve()
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {input_dir}")
        files = sorted(path for path in input_dir.rglob("*.parquet") if path.is_file())
        if not files:
            raise SystemExit(f"No Parquet files found under: {input_dir}")
        if args.extract_workers < 1:
            raise SystemExit("--extract-workers must be positive")

        temp_dir = (
            args.temp_dir.resolve()
            if args.temp_dir
            else database.parent / f"{database.name}.build-parts"
        )
        temp_dir.mkdir(parents=True, exist_ok=True)
        groups = partition_files(files, args.extract_workers)
        cfg = {
            "input_dir": str(input_dir),
            "temp_dir": str(temp_dir),
            "doi_column": args.doi_column,
            "extract_batch_size": args.extract_batch_size,
            "insert_batch_size": args.insert_batch_size,
        }
        tasks = [
            (index, [str(path.relative_to(input_dir)) for path in group], cfg)
            for index, group in enumerate(groups)
        ]
        context = mp.get_context("spawn")
        with context.Pool(processes=len(tasks)) as pool:
            for result in pool.imap_unordered(extract_group, tasks):
                extraction_results.append(result)
                print(
                    f"EXTRACT: shard={result.shard} files={result.files:,} "
                    f"rows={result.rows:,} valid_dois={result.valid_dois:,} "
                    f"unique={result.unique_dois:,} error={result.error}",
                    file=sys.stderr,
                    flush=True,
                )
        failures = [result for result in extraction_results if result.error]
        if failures:
            raise SystemExit(
                f"DOI extraction failed in {len(failures)} shard(s); temp DBs retained"
            )
        merge_shards(
            database,
            [Path(result.shard) for result in extraction_results],
            args.keep_temp,
        )

    connection = sqlite3.connect(database, timeout=60.0)
    configure_database(connection)
    create_schema(connection)
    connection.close()

    api_totals = Counter()
    api_error = None
    if not args.skip_resolve:
        try:
            api_totals = resolve_pending(database, args)
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"
            print(
                f"API resolution paused; committed progress is safe. {api_error}",
                file=sys.stderr,
            )

    summary = {
        "version": VERSION,
        "database": str(database),
        "elapsed_seconds": round(time.time() - start, 3),
        "extraction": [asdict(result) for result in extraction_results],
        "api_run": dict(api_totals),
        "api_error": api_error,
        "database_status": database_status_counts(database),
    }
    summary_path = database.with_suffix(database.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 2 if api_error else 0


if __name__ == "__main__":
    raise SystemExit(main())

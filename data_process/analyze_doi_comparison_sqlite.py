#!/usr/bin/env python3
"""Detailed, read-only statistics for compare_core_s2orc_sqlite.py databases."""

import argparse
import json
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


PERCENTILES = (0.50, 0.90, 0.95, 0.99, 0.999)
BUCKETS = (
    ("1", "{column} = 1"),
    ("2", "{column} = 2"),
    ("3-5", "{column} BETWEEN 3 AND 5"),
    ("6-10", "{column} BETWEEN 6 AND 10"),
    ("11-100", "{column} BETWEEN 11 AND 100"),
    ("101-1000", "{column} BETWEEN 101 AND 1000"),
    (">1000", "{column} > 1000"),
)


def open_readonly_database(database_path, cache_mb, mmap_mb):
    uri_path = quote(str(Path(database_path).resolve()))
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=300)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(f"PRAGMA cache_size=-{cache_mb * 1024}")
    conn.execute(f"PRAGMA mmap_size={mmap_mb * 1024 * 1024}")
    return conn


def validate_database(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(doi_counts)")}
    required = {"doi", "count_a", "count_b"}
    if not required.issubset(columns):
        raise ValueError(
            f"Table doi_counts must contain {sorted(required)}; "
            f"found {sorted(columns)}"
        )


def scalar(conn, sql, parameters=()):
    value = conn.execute(sql, parameters).fetchone()[0]
    return 0 if value is None else value


def database_metadata(conn, database_path):
    database_path = Path(database_path).resolve()
    page_size = scalar(conn, "PRAGMA page_size")
    page_count = scalar(conn, "PRAGMA page_count")
    freelist_count = scalar(conn, "PRAGMA freelist_count")
    wal_path = Path(str(database_path) + "-wal")
    shm_path = Path(str(database_path) + "-shm")
    return {
        "path": str(database_path),
        "sqlite_version": sqlite3.sqlite_version,
        "database_size_bytes": database_path.stat().st_size,
        "wal_size_bytes": wal_path.stat().st_size if wal_path.is_file() else 0,
        "shm_size_bytes": shm_path.stat().st_size if shm_path.is_file() else 0,
        "page_size_bytes": page_size,
        "page_count": page_count,
        "freelist_pages": freelist_count,
        "estimated_used_bytes": (page_count - freelist_count) * page_size,
        "journal_mode": scalar(conn, "PRAGMA journal_mode"),
        "user_version": scalar(conn, "PRAGMA user_version"),
    }


def aggregate_statistics(conn):
    expressions = {
        "union_unique_dois": "COUNT(*)",
        "total_occurrences_a": "SUM(count_a)",
        "total_occurrences_b": "SUM(count_b)",
        "total_occurrences_combined": "SUM(count_a + count_b)",
        "unique_dois_a": "SUM(CASE WHEN count_a > 0 THEN 1 ELSE 0 END)",
        "unique_dois_b": "SUM(CASE WHEN count_b > 0 THEN 1 ELSE 0 END)",
        "shared_unique_dois": (
            "SUM(CASE WHEN count_a > 0 AND count_b > 0 THEN 1 ELSE 0 END)"
        ),
        "only_in_a": "SUM(CASE WHEN count_a > 0 AND count_b = 0 THEN 1 ELSE 0 END)",
        "only_in_b": "SUM(CASE WHEN count_b > 0 AND count_a = 0 THEN 1 ELSE 0 END)",
        "occurs_once_in_a": "SUM(CASE WHEN count_a = 1 THEN 1 ELSE 0 END)",
        "occurs_once_in_b": "SUM(CASE WHEN count_b = 1 THEN 1 ELSE 0 END)",
        "once_and_only_in_a": (
            "SUM(CASE WHEN count_a = 1 AND count_b = 0 THEN 1 ELSE 0 END)"
        ),
        "once_and_only_in_b": (
            "SUM(CASE WHEN count_b = 1 AND count_a = 0 THEN 1 ELSE 0 END)"
        ),
        "occurs_once_in_but_shared": (
            "SUM(CASE WHEN count_b = 1 AND count_a > 0 THEN 1 ELSE 0 END)"
        ),
        "once_in_b_and_once_in_a": (
            "SUM(CASE WHEN count_b = 1 AND count_a = 1 THEN 1 ELSE 0 END)"
        ),
        "duplicated_doi_keys_a": "SUM(CASE WHEN count_a > 1 THEN 1 ELSE 0 END)",
        "duplicated_doi_keys_b": "SUM(CASE WHEN count_b > 1 THEN 1 ELSE 0 END)",
        "duplicate_line_occurrences_a": (
            "SUM(CASE WHEN count_a > 1 THEN count_a - 1 ELSE 0 END)"
        ),
        "duplicate_line_occurrences_b": (
            "SUM(CASE WHEN count_b > 1 THEN count_b - 1 ELSE 0 END)"
        ),
        "one_to_one_matched_occurrences": (
            "SUM(CASE WHEN count_a > 0 AND count_b > 0 "
            "THEN MIN(count_a, count_b) ELSE 0 END)"
        ),
        "occurrences_a_from_shared_dois": (
            "SUM(CASE WHEN count_a > 0 AND count_b > 0 THEN count_a ELSE 0 END)"
        ),
        "occurrences_b_from_shared_dois": (
            "SUM(CASE WHEN count_a > 0 AND count_b > 0 THEN count_b ELSE 0 END)"
        ),
        "occurrences_a_from_only_a_dois": (
            "SUM(CASE WHEN count_a > 0 AND count_b = 0 THEN count_a ELSE 0 END)"
        ),
        "occurrences_b_from_only_b_dois": (
            "SUM(CASE WHEN count_b > 0 AND count_a = 0 THEN count_b ELSE 0 END)"
        ),
        "invalid_negative_counts": (
            "SUM(CASE WHEN count_a < 0 OR count_b < 0 THEN 1 ELSE 0 END)"
        ),
        "invalid_zero_zero_rows": (
            "SUM(CASE WHEN count_a = 0 AND count_b = 0 THEN 1 ELSE 0 END)"
        ),
        "null_or_empty_dois": (
            "SUM(CASE WHEN doi IS NULL OR TRIM(doi) = '' THEN 1 ELSE 0 END)"
        ),
    }

    for dataset, column in (("a", "count_a"), ("b", "count_b")):
        for label, condition in BUCKETS:
            alias = f"bucket_{dataset}_{label.replace('>', 'gt').replace('-', '_')}"
            expressions[alias] = (
                f"SUM(CASE WHEN {condition.format(column=column)} THEN 1 ELSE 0 END)"
            )

    select_list = ",\n".join(
        f"{expression} AS \"{name}\"" for name, expression in expressions.items()
    )
    row = conn.execute(f"SELECT {select_list} FROM doi_counts").fetchone()
    return {name: (0 if row[name] is None else row[name]) for name in expressions}


def build_buckets(aggregates, dataset):
    result = {}
    for label, _ in BUCKETS:
        key = f"bucket_{dataset}_{label.replace('>', 'gt').replace('-', '_')}"
        result[label] = aggregates.pop(key)
    return result


def weighted_distribution(conn, column):
    rows = conn.execute(
        f"""
        SELECT {column} AS occurrences, COUNT(*) AS doi_count
        FROM doi_counts
        WHERE {column} > 0
        GROUP BY {column}
        ORDER BY {column}
        """
    ).fetchall()

    if not rows:
        return {
            "minimum": 0,
            "maximum": 0,
            "mean": 0.0,
            "percentiles": {percentile_name(q): 0 for q in PERCENTILES},
            "frequency_value_count": 0,
        }

    total_dois = sum(row["doi_count"] for row in rows)
    total_occurrences = sum(row["occurrences"] * row["doi_count"] for row in rows)
    targets = {
        q: max(1, math.ceil(q * total_dois))
        for q in PERCENTILES
    }
    percentile_values = {}
    cumulative = 0
    for row in rows:
        cumulative += row["doi_count"]
        for q, target in targets.items():
            if q not in percentile_values and cumulative >= target:
                percentile_values[q] = row["occurrences"]

    return {
        "minimum": rows[0]["occurrences"],
        "maximum": rows[-1]["occurrences"],
        "mean": total_occurrences / total_dois,
        "percentiles": {
            percentile_name(q): percentile_values[q] for q in PERCENTILES
        },
        "frequency_value_count": len(rows),
    }


def percentile_name(q):
    value = q * 100
    return f"p{int(value) if value.is_integer() else value:g}"


def top_dois(conn, order_expression, condition, top_n):
    if top_n <= 0:
        return []
    rows = conn.execute(
        f"""
        SELECT doi, count_a, count_b, count_a + count_b AS count_combined
        FROM doi_counts
        WHERE {condition}
        ORDER BY {order_expression} DESC, doi ASC
        LIMIT ?
        """,
        (top_n,),
    ).fetchall()
    return [dict(row) for row in rows]


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def derived_statistics(stats):
    unique_a = stats["unique_dois_a"]
    unique_b = stats["unique_dois_b"]
    shared = stats["shared_unique_dois"]
    union = stats["union_unique_dois"] - stats["invalid_zero_zero_rows"]
    total_a = stats["total_occurrences_a"]
    total_b = stats["total_occurrences_b"]
    return {
        "jaccard_unique_doi": divide(shared, union),
        "overlap_coefficient": divide(shared, min(unique_a, unique_b)),
        "fraction_of_a_unique_dois_also_in_b": divide(shared, unique_a),
        "fraction_of_b_unique_dois_also_in_a": divide(shared, unique_b),
        "fraction_of_a_unique_dois_only_in_a": divide(stats["only_in_a"], unique_a),
        "fraction_of_b_unique_dois_only_in_b": divide(stats["only_in_b"], unique_b),
        "mean_occurrences_per_unique_doi_a": divide(total_a, unique_a),
        "mean_occurrences_per_unique_doi_b": divide(total_b, unique_b),
        "duplicate_line_fraction_a": divide(stats["duplicate_line_occurrences_a"], total_a),
        "duplicate_line_fraction_b": divide(stats["duplicate_line_occurrences_b"], total_b),
        "singleton_doi_fraction_a": divide(stats["occurs_once_in_a"], unique_a),
        "singleton_doi_fraction_b": divide(stats["occurs_once_in_b"], unique_b),
        "one_to_one_match_fraction_of_a_occurrences": divide(
            stats["one_to_one_matched_occurrences"], total_a
        ),
        "one_to_one_match_fraction_of_b_occurrences": divide(
            stats["one_to_one_matched_occurrences"], total_b
        ),
    }


def format_bytes(value):
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:,.2f} {unit}"
        value /= 1024


def print_integer_section(title, mapping):
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(key) for key in mapping) + 2
    for key, value in mapping.items():
        print(f"{key + ':':<{width}} {value:,}")


def print_ratio_section(title, mapping):
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(key) for key in mapping) + 2
    for key, value in mapping.items():
        print(f"{key + ':':<{width}} {value:.6f} ({value * 100:.4f}%)")


def print_float_section(title, mapping):
    print(f"\n{title}")
    print("-" * len(title))
    width = max(len(key) for key in mapping) + 2
    for key, value in mapping.items():
        print(f"{key + ':':<{width}} {value:,.6f}")


def print_distribution(dataset, distribution, buckets):
    title = f"Folder {dataset.upper()} occurrence distribution"
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Minimum occurrences per DOI: {distribution['minimum']:,}")
    print(f"Maximum occurrences per DOI: {distribution['maximum']:,}")
    print(f"Mean occurrences per DOI:    {distribution['mean']:,.6f}")
    for name, value in distribution["percentiles"].items():
        print(f"{name.upper():<28} {value:,}")
    print("DOI-count buckets:")
    for name, value in buckets.items():
        print(f"  {name:<10} {value:,}")


def print_top(title, rows):
    if not rows:
        return
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'DOI':<55} {'count_a':>12} {'count_b':>12} {'combined':>12}")
    for row in rows:
        doi = row["doi"]
        if len(doi) > 55:
            doi = doi[:52] + "..."
        print(
            f"{doi:<55} {row['count_a']:>12,} "
            f"{row['count_b']:>12,} {row['count_combined']:>12,}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed statistics from a DOI comparison SQLite database."
    )
    parser.add_argument("database", nargs="?", default="doi_comparison.sqlite")
    parser.add_argument(
        "--output-json",
        default="doi_comparison_statistics.json",
        help="JSON report path; use an empty string to disable JSON output",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top repeated DOIs per list; 0 disables")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip exact percentiles and top-DOI sorting",
    )
    parser.add_argument("--cache-mb", type=int, default=256)
    parser.add_argument("--mmap-mb", type=int, default=1024)
    args = parser.parse_args()

    if args.top_n < 0:
        parser.error("--top-n cannot be negative")
    if args.cache_mb < 1 or args.mmap_mb < 0:
        parser.error("--cache-mb must be positive and --mmap-mb cannot be negative")

    database_path = Path(args.database).resolve()
    if not database_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    started = time.perf_counter()
    conn = open_readonly_database(database_path, args.cache_mb, args.mmap_mb)
    try:
        validate_database(conn)
        metadata = database_metadata(conn, database_path)

        print(f"Analyzing: {database_path}", flush=True)
        aggregates = aggregate_statistics(conn)
        buckets = {
            "folder_a": build_buckets(aggregates, "a"),
            "folder_b": build_buckets(aggregates, "b"),
        }
        derived = derived_statistics(aggregates)

        if args.fast:
            distributions = {}
            top = {}
        else:
            print("Computing Folder A exact weighted percentiles...", flush=True)
            distribution_a = weighted_distribution(conn, "count_a")
            print("Computing Folder B exact weighted percentiles...", flush=True)
            distribution_b = weighted_distribution(conn, "count_b")
            distributions = {"folder_a": distribution_a, "folder_b": distribution_b}

            top_n = args.top_n
            top = {
                "folder_a": top_dois(conn, "count_a", "count_a > 0", top_n),
                "folder_b": top_dois(conn, "count_b", "count_b > 0", top_n),
                "only_in_a": top_dois(
                    conn, "count_a", "count_a > 0 AND count_b = 0", top_n
                ),
                "only_in_b": top_dois(
                    conn, "count_b", "count_b > 0 AND count_a = 0", top_n
                ),
                "combined": top_dois(
                    conn, "count_a + count_b", "count_a + count_b > 0", top_n
                ),
            }
    finally:
        conn.close()

    elapsed = time.perf_counter() - started
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "database": metadata,
        "counts": aggregates,
        "requested_statistics": {
            "occurs_once_in_b": aggregates["occurs_once_in_b"],
            "only_in_b": aggregates["only_in_b"],
            "once_and_only_in_b": aggregates["once_and_only_in_b"],
        },
        "derived_ratios": derived,
        "occurrence_buckets": buckets,
        "occurrence_distributions": distributions,
        "top_repeated_dois": top,
    }

    print("\nDATABASE")
    print("--------")
    print(f"Size:          {format_bytes(metadata['database_size_bytes'])}")
    print(f"SQLite:        {metadata['sqlite_version']}")
    print(f"Journal mode:  {metadata['journal_mode']}")

    print_integer_section(
        "DOI set statistics",
        {
            "Union unique DOIs": aggregates["union_unique_dois"],
            "Unique DOIs in A": aggregates["unique_dois_a"],
            "Unique DOIs in B": aggregates["unique_dois_b"],
            "Shared unique DOIs": aggregates["shared_unique_dois"],
            "Only in A": aggregates["only_in_a"],
            "Only in B": aggregates["only_in_b"],
        },
    )
    print_integer_section(
        "Requested Folder B statistics",
        {
            "Occurs once in B": aggregates["occurs_once_in_b"],
            "Only in B (count_b > 0, count_a = 0)": aggregates["only_in_b"],
            "Once and only in B": aggregates["once_and_only_in_b"],
            "Occurs once in B but is shared with A": aggregates["occurs_once_in_but_shared"],
        },
    )
    print_integer_section(
        "Line-occurrence and duplication statistics",
        {
            "Total DOI occurrences in A": aggregates["total_occurrences_a"],
            "Total DOI occurrences in B": aggregates["total_occurrences_b"],
            "Duplicate line occurrences in A": aggregates["duplicate_line_occurrences_a"],
            "Duplicate line occurrences in B": aggregates["duplicate_line_occurrences_b"],
            "DOI keys duplicated in A": aggregates["duplicated_doi_keys_a"],
            "DOI keys duplicated in B": aggregates["duplicated_doi_keys_b"],
            "One-to-one matched occurrences": aggregates["one_to_one_matched_occurrences"],
        },
    )
    mean_values = {
        key: value for key, value in derived.items()
        if key.startswith("mean_occurrences")
    }
    ratio_values = {
        key: value for key, value in derived.items()
        if not key.startswith("mean_occurrences")
    }
    print_ratio_section("Overlap and duplication ratios", ratio_values)
    print_float_section("Mean occurrence counts", mean_values)
    print_integer_section(
        "Integrity checks (all should be zero)",
        {
            "Negative count rows": aggregates["invalid_negative_counts"],
            "Rows with count_a = count_b = 0": aggregates["invalid_zero_zero_rows"],
            "Null or empty DOI rows": aggregates["null_or_empty_dois"],
        },
    )

    if distributions:
        print_distribution("a", distributions["folder_a"], buckets["folder_a"])
        print_distribution("b", distributions["folder_b"], buckets["folder_b"])
        print_top("Most repeated DOIs in Folder A", top["folder_a"])
        print_top("Most repeated DOIs in Folder B", top["folder_b"])
        print_top("Most repeated DOIs occurring only in Folder B", top["only_in_b"])

    print(f"\nElapsed time: {elapsed:,.2f} seconds")
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(f"JSON report:  {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os
from pathlib import Path

import duckdb


INPUT_GLOB = (
    "/dpc-zhouy/zhouy/papers/"
    "only_in_s2orc_parquet/*.parquet"
)

OUTPUT_DIR = Path(
    "/dpc-zhouy/zhouy/papers/"
    "only_in_s2orc_parquet_deduplicated"
)

WORK_DATABASE = (
    "/dpc-zhouy/zhouy/papers/"
    "deduplicate_locations.duckdb"
)

TEMP_DIRECTORY = Path(
    "/dpc-zhouy/zhouy/papers/"
    "duckdb_deduplicate_tmp"
)

THREADS = 16
MEMORY_LIMIT = "500GB"
MAX_TEMP_SIZE = "2TB"


def sql_string(value):
    """Safely quote a path as a DuckDB SQL string."""
    return "'" + str(value).replace("'", "''") + "'"


def table_exists(con, table_name):
    result = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()

    return result[0] > 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(WORK_DATABASE)

    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(
        f"SET temp_directory = {sql_string(TEMP_DIRECTORY)}"
    )
    con.execute(
        f"SET max_temp_directory_size = '{MAX_TEMP_SIZE}'"
    )
    con.execute("SET enable_progress_bar = true")

    # ---------------------------------------------------------
    # Phase 1: assign a stable integer ID to every input file.
    # ---------------------------------------------------------
    if not table_exists(con, "source_files"):
        print("Creating source file table...", flush=True)

        con.execute(
            f"""
            CREATE TABLE source_files AS
            SELECT
                row_number() OVER (ORDER BY file) - 1 AS file_id,
                file AS filename
            FROM glob({sql_string(INPUT_GLOB)})
            """
        )

        con.execute("CHECKPOINT")

    source_file_count = con.execute(
        "SELECT COUNT(*) FROM source_files"
    ).fetchone()[0]

    print(
        f"Input Parquet files: {source_file_count:,}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Phase 2: select one physical row location for each DOI.
    #
    # The existing Parquet dataset already excludes empty
    # full_text rows, so this phase only reads DOI and metadata.
    # It does not retain full_text in the GROUP BY hash table.
    # ---------------------------------------------------------
    if not table_exists(con, "winner_locations"):
        print(
            "Finding one row location for each DOI...",
            flush=True,
        )

        con.execute(
            f"""
            CREATE TABLE winner_locations AS
            WITH grouped AS (
                SELECT
                    p.doi,

                    arg_min(
                        struct_pack(
                            file_id := f.file_id,
                            row_no := p.file_row_number
                        ),

                        CAST(f.file_id AS HUGEINT)
                            * 1000000000000
                            + p.file_row_number
                    ) AS location

                FROM read_parquet(
                    {sql_string(INPUT_GLOB)},
                    filename = true,
                    file_row_number = true
                ) AS p

                INNER JOIN source_files AS f
                    ON p.filename = f.filename

                WHERE
                    p.doi IS NOT NULL
                    AND trim(p.doi) <> ''

                GROUP BY p.doi
            )

            SELECT
                doi,
                location.file_id AS file_id,
                location.row_no AS row_no
            FROM grouped
            """
        )

        con.execute("CHECKPOINT")

    unique_dois = con.execute(
        "SELECT COUNT(*) FROM winner_locations"
    ).fetchone()[0]

    print(
        f"Unique DOI rows selected: {unique_dois:,}",
        flush=True,
    )

    # ---------------------------------------------------------
    # Phase 3: process one source Parquet file at a time.
    #
    # This prevents a global join from retaining large amounts
    # of full-text data. Each output part is independently
    # restartable.
    # ---------------------------------------------------------
    source_files = con.execute(
        """
        SELECT file_id, filename
        FROM source_files
        ORDER BY file_id
        """
    ).fetchall()

    for position, (file_id, filename) in enumerate(
        source_files,
        start=1,
    ):
        output_file = (
            OUTPUT_DIR /
            f"part-{int(file_id):05d}.parquet"
        )

        if output_file.is_file():
            print(
                f"[{position}/{source_file_count}] "
                f"Skipping existing {output_file.name}",
                flush=True,
            )
            continue

        print(
            f"[{position}/{source_file_count}] "
            f"Processing {filename}",
            flush=True,
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    p.doi,
                    p.full_text

                FROM read_parquet(
                    {sql_string(filename)},
                    file_row_number = true
                ) AS p

                INNER JOIN winner_locations AS w
                    ON w.file_id = {int(file_id)}
                   AND w.row_no = p.file_row_number
                   AND w.doi = p.doi
            )
            TO {sql_string(output_file)}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 10000
            )
            """
        )

    # Parquet COUNT(*) generally uses file metadata and is cheap.
    output_glob = OUTPUT_DIR / "*.parquet"

    written_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet({sql_string(output_glob)})
        """
    ).fetchone()[0]

    print("\nDEDUPLICATION RESULTS")
    print("---------------------")
    print(f"Expected unique DOI rows: {unique_dois:,}")
    print(f"Written Parquet rows:     {written_rows:,}")
    print(f"Output files:             {source_file_count:,}")

    if written_rows == unique_dois:
        print("Validation:               PASS")
    else:
        print("Validation:               FAIL")
        print(
            "Some output parts may be missing or incomplete."
        )

    con.close()


if __name__ == "__main__":
    main()
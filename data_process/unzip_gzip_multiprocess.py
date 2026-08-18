#!/usr/bin/env python3
"""
Decompress .gz and .gz.gz files in parallel.

The script detects the actual number of gzip layers from file content instead
of relying only on the filename. For example, a file named data.json.gz.gz that
contains only one gzip layer will still be processed correctly.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class Job:
    source: Path
    destination: Path
    suffix_layers: int


def strip_gzip_suffixes(path: Path) -> tuple[Path, int]:
    """
    Remove all trailing .gz suffixes.

    Examples:
        file.json.gz       -> file.json, 1
        file.json.gz.gz    -> file.json, 2
    """
    name = path.name
    layers = 0

    while name.lower().endswith(".gz"):
        name = name[:-3]
        layers += 1

    return path.with_name(name), layers


def discover_jobs(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
) -> list[Job]:
    """Find all .gz and .gz.gz files."""
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    jobs: list[Job] = []

    for source in iterator:
        if not source.is_file():
            continue

        if not source.name.lower().endswith(".gz"):
            continue

        relative_path = source.relative_to(input_dir)
        output_relative_path, suffix_layers = strip_gzip_suffixes(relative_path)

        jobs.append(
            Job(
                source=source,
                destination=output_dir / output_relative_path,
                suffix_layers=suffix_layers,
            )
        )

    jobs.sort(key=lambda job: str(job.source))
    return jobs


def reject_destination_collisions(jobs: list[Job]) -> None:
    """
    Prevent two source files from writing to the same destination.

    For example:
        data.json.gz
        data.json.gz.gz

    Both would otherwise produce:
        data.json
    """
    destinations: dict[Path, list[Path]] = {}

    for job in jobs:
        destinations.setdefault(job.destination, []).append(job.source)

    collisions = {
        destination: sources
        for destination, sources in destinations.items()
        if len(sources) > 1
    }

    if not collisions:
        return

    details = "\n".join(
        f"  {destination} <- {', '.join(map(str, sources))}"
        for destination, sources in collisions.items()
    )

    raise ValueError(
        "Multiple input files would create the same output file:\n"
        f"{details}\n"
        "Rename one of the colliding input files and run again."
    )


def decompress_one(
    job: Job,
    overwrite: bool,
    buffer_size: int,
) -> tuple[str, str, int | None]:
    """
    Decompress one file.

    The file content is checked for the gzip magic header before opening each
    gzip layer. This handles mislabeled .gz.gz files containing only one layer.
    """
    destination = job.destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        return "skipped", str(destination), None

    temporary = destination.with_name(
        f".{destination.name}.part-{os.getpid()}"
    )

    try:
        with ExitStack() as stack:
            stream = stack.enter_context(job.source.open("rb"))
            actual_layers = 0

            # Decompress at most as many layers as indicated by the filename,
            # but verify each layer using the gzip magic bytes.
            for _ in range(job.suffix_layers):
                magic_bytes = stream.peek(2)[:2]

                if magic_bytes != GZIP_MAGIC:
                    break

                stream = stack.enter_context(
                    gzip.GzipFile(fileobj=stream, mode="rb")
                )
                actual_layers += 1

            if actual_layers == 0:
                raise gzip.BadGzipFile(
                    "filename ends in .gz, but the file has no gzip header"
                )

            output = stack.enter_context(temporary.open("wb"))

            shutil.copyfileobj(
                stream,
                output,
                length=buffer_size,
            )

        # Atomic replacement: incomplete temporary output is never exposed.
        os.replace(temporary, destination)

        return "written", str(destination), actual_layers

    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompress .gz and .gz.gz files in parallel. "
            "Actual gzip layers are detected from file content."
        )
    )

    parser.add_argument("--input_dir", type=Path, default="/dpc-zhouy/zhouy/papers/arxiv", help="Folder containing gzip files")
    parser.add_argument("--output_dir", type=Path, default="/dpc-zhouy/zhouy/papers/arxiv_extracted", help="New folder for decompressed files")
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes (default: number of CPU cores)",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Process only files directly inside input_dir",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files instead of skipping them",
    )

    parser.add_argument(
        "--buffer-mib",
        type=int,
        default=4,
        help="Streaming buffer size per worker in MiB (default: 4)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(
            f"Error: input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 2

    if args.workers < 1:
        print(
            "Error: --workers must be at least 1",
            file=sys.stderr,
        )
        return 2

    if args.buffer_mib < 1:
        print(
            "Error: --buffer-mib must be at least 1",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=not args.no_recursive,
    )

    try:
        reject_destination_collisions(jobs)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    if not jobs:
        print("No .gz or .gz.gz files found.")
        return 0

    print(f"Found {len(jobs):,} gzip files.")
    print(f"Worker processes: {min(args.workers, len(jobs))}")
    print(f"Output directory: {output_dir}")

    written = 0
    skipped = 0
    failed = 0

    buffer_size = args.buffer_mib * 1024 * 1024
    worker_count = min(args.workers, len(jobs))

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {
            executor.submit(
                decompress_one,
                job,
                args.overwrite,
                buffer_size,
            ): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                status, destination, actual_layers = future.result()

                if status == "written":
                    written += 1

                    layer_note = ""

                    if actual_layers != job.suffix_layers:
                        layer_note = (
                            f" (detected {actual_layers} gzip layer(s); "
                            f"filename suggests {job.suffix_layers})"
                        )

                    print(
                        f"[OK]   {job.source} -> "
                        f"{destination}{layer_note}"
                    )

                else:
                    skipped += 1
                    print(f"[SKIP] {destination} already exists")

            except Exception as error:
                failed += 1
                print(
                    f"[FAIL] {job.source}: {error}",
                    file=sys.stderr,
                )

    print(
        f"Done: written={written:,}, "
        f"skipped={skipped:,}, "
        f"failed={failed:,}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Recursively count sequences, documents, and tokens in Megatron .idx files."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


MMAP_INDEX_MAGIC = b"MMIDIDX\x00\x00"
SEQUENCE_LENGTH_BYTES = 4
LENGTHS_PER_CHUNK = 1_000_000


def read_idx_counts(idx_path: Path) -> tuple[int, int, int, int]:
    """Return version, sequence count, document count, and total tokens."""
    with idx_path.open("rb") as stream:
        magic = stream.read(len(MMAP_INDEX_MAGIC))
        if magic != MMAP_INDEX_MAGIC:
            raise ValueError(f"unsupported index header {magic!r}")

        version_data = stream.read(8)
        dtype_code = stream.read(1)
        sequence_data = stream.read(8)
        document_data = stream.read(8)

        if not all((version_data, dtype_code, sequence_data, document_data)):
            raise ValueError("truncated index header")

        version = struct.unpack("<Q", version_data)[0]
        sequence_count = struct.unpack("<Q", sequence_data)[0]
        document_count = struct.unpack("<Q", document_data)[0]

        # Megatron stores one little-endian int32 sequence length per entry
        # immediately after the fixed-size index header. Stream the table in
        # chunks so even indexes with hundreds of millions of entries use
        # bounded memory.
        total_tokens = 0
        remaining = sequence_count
        while remaining:
            count = min(remaining, LENGTHS_PER_CHUNK)
            data = stream.read(count * SEQUENCE_LENGTH_BYTES)
            if len(data) != count * SEQUENCE_LENGTH_BYTES:
                raise ValueError("truncated sequence-length table")
            total_tokens += sum(value[0] for value in struct.iter_unpack("<i", data))
            remaining -= count

        return version, sequence_count, document_count, total_tokens


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively count entries in Megatron/MindSpeed .idx files."
    )
    # parser.add_argument("--root", type=Path, default="/dpc-zhouy/zhouy/instructft_data", help="Root directory to search")
    # parser.add_argument("--postfix", default="*_labels_document.idx", type=str)
    
    parser.add_argument("--root", type=Path, default="/dpc-zhouy/zhouy/pretraining", help="Root directory to search")
    parser.add_argument("--postfix", default="*_text_document.idx", type=str)

    # parser.add_argument("--root", type=Path, default="/dpc-zhouy/zhouy/papers/core_bin_idx", help="Root directory to search")
    # parser.add_argument("--postfix", default="*clean_text_document.idx", type=str)
    # parser.add_argument("--postfix", default="*52m_text_document.idx", type=str)

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    idx_files = sorted(root.rglob(args.postfix))
    if not idx_files:
        print(f"No .idx files found under {root}", file=sys.stderr)
        return 1

    total_sequences = 0
    total_documents = 0
    successful = 0

    total_tokens = 0

    print("sequences\tdocuments\ttokens\tversion\tidx_file")
    for idx_path in idx_files:
        try:
            version, sequences, documents, tokens = read_idx_counts(idx_path)
        except (OSError, ValueError) as exc:
            print(f"ERROR\t-\t-\t-\t{idx_path}\t{exc}", file=sys.stderr)
            continue

        print(f"{sequences}\t{documents}\t{tokens}\t{version}\t{idx_path}")
        total_sequences += sequences
        total_documents += documents
        total_tokens += tokens
        successful += 1

    print(
        f"TOTAL\tsequences={total_sequences}\tdocuments={total_documents}"
        f"\ttokens={total_tokens}\tfiles={successful}/{len(idx_files)}"
    )
    return 0 if successful == len(idx_files) else 2


if __name__ == "__main__":
    raise SystemExit(main())


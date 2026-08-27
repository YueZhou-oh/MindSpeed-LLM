#!/usr/bin/env python3
"""Clean the ``text`` column of many Parquet files using file-level workers.

The program streams Arrow record batches, preserves every non-text column and
the original Arrow schema metadata, drops rows whose cleaned text fails the
configured filters, and writes each result to the same relative path under a
new output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "1.0.0"

SECTION_HEADINGS = {
    "abstract", "background", "introduction", "methods", "method",
    "materials and methods", "methods and analysis", "results", "discussion",
    "conclusion", "conclusions", "limitations", "keywords", "summary",
}

TAIL_HEADINGS = {
    "authors' contributions", "author contributions", "author contribution",
    "acknowledgements", "acknowledgments", "supplementary material",
    "funding", "funding information", "conflict of interest",
    "conflicts of interest", "competing interests", "data availability",
    "ethics approval", "references", "bibliography",
}

BOILERPLATE_PATTERNS = [
    re.compile(r"^downloaded\s+from\b", re.I),
    re.compile(r"^https?://\S+$", re.I),
    re.compile(r"^copyright\s+.{0,120}$", re.I),
    re.compile(r"^(?:©|\(c\))\s*\d{4}\b", re.I),
    re.compile(r"^(?:volume|vol\.)\s*\d+.{0,80}$", re.I),
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.I),
    re.compile(r"^\d+\s+[A-Z][\w.'-]+(?:\s+et\s+al\.)?$"),
    re.compile(r"^.*\bopen access article distributed under the terms\b.*$", re.I),
    re.compile(r"^.*\bcreative commons attribution\b.*$", re.I),
    re.compile(r"^\W*corresponding author\b.*(?:email|e-mail|tel|fax)\b", re.I),
    re.compile(r"^this paper was (?:guest )?edited by\b", re.I),
    re.compile(r"^in any medium, provided the original work is properly cited\b", re.I),
    re.compile(r"^for commercial re-use, please contact\b", re.I),
    re.compile(r"^[A-Z].*\(\d{4}\).*\d+\s*[–-]\s*\d+.*[A-Z]{3,}.*$"),
]

WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
SENTENCE_END_RE = re.compile(r"[.!?][\]\)\"'’”]*$")
LIST_RE = re.compile(r"^(?:[-•*]|\(?\d+[.)]|[A-Za-z][.)])\s+")
FIG_TABLE_RE = re.compile(r"^(?:figure|fig\.?|table)\s+[A-Za-z0-9]+\b", re.I)

ENGLISH_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "between", "by",
    "for", "from", "had", "has", "have", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "these", "this", "to", "was", "were",
    "which", "with", "we", "using", "our", "than", "may",
}

COMMON_HYPHENATED = {
    "long-term", "short-term", "follow-up", "patient-years", "well-known",
    "non-selective", "double-blind", "placebo-controlled", "randomized-controlled",
    "state-of-the-art", "large-scale", "high-quality", "real-world",
}

HYPHEN_PREFIXES = {
    "anti", "co", "cross", "ex", "high", "low", "mid", "multi", "non",
    "pre", "post", "pro", "re", "self", "semi", "short", "long",
}


@dataclass
class FileStats:
    path: str
    rows_in: int = 0
    rows_out: int = 0
    null_or_non_string: int = 0
    too_short: int = 0
    low_quality: int = 0
    controls_removed: int = 0
    glyph_repairs: int = 0
    noise_lines_removed: int = 0
    boilerplate_lines_removed: int = 0
    dehyphenations: int = 0
    front_matter_removed: int = 0
    tail_sections_removed: int = 0
    batches: int = 0
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=Path("/dpc-zhouy/zhouy/papers/only_in_s2orc_parquet_deduplicated"))
    parser.add_argument("--output-dir", default=Path("/dpc-zhouy/zhouy/papers/only_in_s2orc_parquet_deduplicated_clean"))
    parser.add_argument("--text-column", default="full_text")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--row-group-size", type=int, default=10_000)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--min-quality-score", type=float, default=0.55)
    parser.add_argument("--max-suspicious-ratio", type=float, default=0.0003)
    parser.add_argument("--compression", default="zstd", choices=("zstd", "snappy", "gzip", "none"))
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--keep-front-matter", action="store_true")
    parser.add_argument("--keep-tail-sections", action="store_true")
    parser.add_argument(
        "--arrow-use-threads",
        action="store_true",
        help="Enable Arrow threads inside each process; normally leave disabled",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace matching files already present under output-dir",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = text.strip()
    return text or None


def normalize_heading(line: str) -> str:
    line = line.replace("’", "'").replace("‘", "'")
    return re.sub(r"[\s:._-]+", " ", line.strip().lower()).strip()


def is_heading(line: str) -> bool:
    value = normalize_heading(line)
    if value in SECTION_HEADINGS or value in TAIL_HEADINGS:
        return True
    words = value.split()
    return 0 < len(words) <= 8 and len(line) <= 100 and (
        line.isupper() or line.istitle()
    )


def remove_control_characters(text: str) -> tuple[str, int]:
    output: list[str] = []
    removed = 0
    for char in text:
        category = unicodedata.category(char)
        if category == "Cc" and char not in "\n\t":
            removed += 1
            continue
        if category == "Cf" and char not in {"\u200c", "\u200d"}:
            removed += 1
            continue
        output.append(char)
    return "".join(output), removed


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ").replace("\u00ad", "").replace("\ufeff", "")
    return re.sub(r"[ \t]+", " ", line).strip()


def repair_high_confidence_pdf_glyphs(text: str, cleaning: Counter) -> str:
    # Repair '=' mis-extracted as '¼' only immediately before a numeric value.
    text, replacements = re.subn(r"¼(?=\s*[+\-−]?(?:\d|\.\d))", "=", text)
    cleaning["glyph_repairs"] += replacements
    return text


def is_symbol_line(line: str) -> bool:
    value = re.sub(r"\s+", "", line)
    if not value:
        return False
    if len(value) >= 8 and len(set(value)) <= 3:
        return True
    alphanumeric = sum(char.isalnum() for char in value)
    return len(value) >= 20 and alphanumeric / len(value) < 0.08


def filter_layout_lines(lines: list[str], cleaning: Counter) -> list[str]:
    normalized = [normalize_line(line) for line in lines]
    running_keys = Counter()
    for line in normalized:
        if 20 <= len(line) <= 160 and re.search(r"\s+\d{1,4}$", line):
            key = re.sub(r"(?:^\d{1,4}\s+|\s+\d{1,4}$)", "", line)
            running_keys[key] += 1

    kept: list[str] = []
    for line in normalized:
        if not line:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if is_symbol_line(line):
            cleaning["noise_lines_removed"] += 1
            continue
        if any(pattern.match(line) for pattern in BOILERPLATE_PATTERNS):
            cleaning["boilerplate_lines_removed"] += 1
            continue
        running_key = re.sub(r"(?:^\d{1,4}\s+|\s+\d{1,4}$)", "", line)
        if running_keys.get(running_key, 0) >= 2:
            cleaning["boilerplate_lines_removed"] += 1
            continue
        if len(line) <= 2 and not line.isalpha():
            cleaning["noise_lines_removed"] += 1
            continue
        kept.append(line)
    while kept and kept[-1] == "":
        kept.pop()
    return kept


def strip_front_matter(lines: list[str], cleaning: Counter) -> list[str]:
    marker = None
    marker_re = re.compile(r"^(?:abstract|background|introduction)\b", re.I)
    for index, line in enumerate(lines[:120]):
        if index >= 3 and marker_re.match(line):
            marker = index
            break
    if marker is None or marker <= 3:
        return lines

    title_lines: list[str] = []
    for line in lines[:min(marker, 10)]:
        if not line:
            if title_lines:
                break
            continue
        author_like = line.count(",") >= 2 and (
            re.search(r"\b[A-Z]\.", line)
            or re.search(r"[A-Za-z]\d+(?:\*|,|$)", line)
        )
        affiliation_like = re.match(
            r"^\d*(?:department|division|faculty|school|university|institute|centre|center)\b",
            line,
            re.I,
        )
        if author_like or affiliation_like:
            break
        title_lines.append(line)
    if not title_lines:
        return lines
    cleaning["front_matter_removed"] += 1
    return [" ".join(title_lines), ""] + lines[marker:]


def strip_tail_sections(lines: list[str], cleaning: Counter) -> list[str]:
    start = int(len(lines) * 0.50)
    for index in range(start, len(lines)):
        if normalize_heading(lines[index]) in TAIL_HEADINGS:
            cleaning["tail_sections_removed"] += 1
            return lines[:index]
    return lines


def should_dehyphenate(left_word: str, right_word: str) -> bool:
    original_left = left_word.rstrip("-")
    left = original_left.lower()
    compound = f"{left}-{right_word.lower()}"
    if (
        compound in COMMON_HYPHENATED
        or left in HYPHEN_PREFIXES
        or (len(original_left) >= 2 and original_left.isupper())
    ):
        return False
    return True


def join_wrapped_lines(lines: list[str], cleaning: Counter) -> str:
    paragraphs: list[str] = []
    current = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            if current:
                paragraphs.append(current.strip())
                current = ""
            index += 1
            continue

        if is_heading(line) or LIST_RE.match(line) or FIG_TABLE_RE.match(line):
            if current:
                paragraphs.append(current.strip())
                current = ""
            paragraphs.append(line)
            index += 1
            continue

        current = line if not current else current + " " + line
        if index + 1 < len(lines) and lines[index + 1]:
            next_line = lines[index + 1]
            left_match = re.search(r"([A-Za-z][A-Za-z-]*)-$", current)
            right_match = re.match(r"([a-z]+)\b", next_line)
            if left_match and right_match:
                if should_dehyphenate(left_match.group(1), right_match.group(1)):
                    current = current[:-1]
                    cleaning["dehyphenations"] += 1
                lines[index + 1] = "\u0000" + next_line

        next_value = lines[index + 1] if index + 1 < len(lines) else ""
        next_clean = next_value.lstrip("\u0000")
        if SENTENCE_END_RE.search(current) and (
            not next_value
            or is_heading(next_clean)
            or LIST_RE.match(next_clean)
            or FIG_TABLE_RE.match(next_clean)
        ):
            paragraphs.append(current.strip())
            current = ""
        index += 1
        if index < len(lines) and lines[index].startswith("\u0000"):
            continuation = lines[index][1:]
            current = current + continuation if current else continuation
            index += 1

    if current:
        paragraphs.append(current.strip())
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def calculate_quality(text: str) -> tuple[float, float]:
    length = max(1, len(text))
    printable_ratio = sum(c.isprintable() or c in "\n\t" for c in text) / length
    alphabetic_ratio = sum(c.isalpha() for c in text) / length
    replacement_ratio = text.count("\ufffd") / length
    suspicious_ratio = sum(
        c in "□�¼þ" or unicodedata.category(c) == "Co" for c in text
    ) / length

    words = [word.lower() for word in WORD_RE.findall(text)]
    if words:
        english_ratio = sum(w in ENGLISH_FUNCTION_WORDS for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
    else:
        english_ratio = 0.0
        unique_ratio = 0.0
    paragraph_count = sum(bool(p.strip()) for p in text.split("\n\n"))

    score = (
        0.15 * min(1.0, len(words) / 500.0)
        + 0.15 * max(0.0, min(1.0, (printable_ratio - 0.94) / 0.06))
        + 0.15 * max(0.0, min(1.0, (alphabetic_ratio - 0.35) / 0.30))
        + 0.15 * max(0.0, min(1.0, english_ratio / 0.10))
        + 0.10 * max(0.0, min(1.0, unique_ratio / 0.20))
        + 0.20 * max(0.0, 1.0 - 1000.0 * (replacement_ratio + suspicious_ratio))
        + 0.10 * min(1.0, math.log2(paragraph_count + 1) / 4.0)
    )
    return score, suspicious_ratio


def clean_text(value: object, cfg: dict) -> tuple[str | None, Counter, str | None]:
    text = normalize_text(value)
    cleaning = Counter()
    if text is None:
        return None, cleaning, "null_or_non_string"

    text, cleaning["controls_removed"] = remove_control_characters(text)
    text = repair_high_confidence_pdf_glyphs(text, cleaning)
    lines = filter_layout_lines(text.split("\n"), cleaning)
    if not cfg["keep_front_matter"]:
        lines = strip_front_matter(lines, cleaning)
    if not cfg["keep_tail_sections"]:
        lines = strip_tail_sections(lines, cleaning)
    text = join_wrapped_lines(lines, cleaning)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < cfg["min_chars"]:
        return None, cleaning, "too_short"
    score, suspicious_ratio = calculate_quality(text)
    if score < cfg["min_quality_score"] or suspicious_ratio > cfg["max_suspicious_ratio"]:
        return None, cleaning, "low_quality"
    return text, cleaning, None


def process_one_file(task: tuple[str, dict]) -> FileStats:
    relative_name, cfg = task
    input_path = Path(cfg["input_dir"]) / relative_name
    output_path = Path(cfg["output_dir"]) / relative_name
    stats = FileStats(path=relative_name)
    temp_path = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    writer: pq.ParquetWriter | None = None

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parquet_file = pq.ParquetFile(input_path)
        schema = parquet_file.schema_arrow
        text_index = schema.get_field_index(cfg["text_column"])
        if text_index < 0:
            raise ValueError(f"column {cfg['text_column']!r} not found")
        text_field = schema.field(text_index)
        if not (pa.types.is_string(text_field.type) or pa.types.is_large_string(text_field.type)):
            raise TypeError(
                f"column {cfg['text_column']!r} must be string/large_string, got {text_field.type}"
            )

        writer_options = {
            "compression": None if cfg["compression"] == "none" else cfg["compression"],
            "use_dictionary": False,
        }
        if cfg["compression"] in {"zstd", "gzip"}:
            writer_options["compression_level"] = cfg["compression_level"]
        writer = pq.ParquetWriter(temp_path, schema=schema, **writer_options)

        for batch in parquet_file.iter_batches(
            batch_size=cfg["batch_size"],
            use_threads=cfg["arrow_use_threads"],
        ):
            stats.batches += 1
            stats.rows_in += batch.num_rows
            values = batch.column(text_index).to_pylist()
            keep_indices: list[int] = []
            cleaned_values: list[str] = []

            for row_index, value in enumerate(values):
                cleaned, cleaning, rejected_reason = clean_text(value, cfg)
                for name in (
                    "controls_removed", "glyph_repairs", "noise_lines_removed",
                    "boilerplate_lines_removed", "dehyphenations",
                    "front_matter_removed", "tail_sections_removed",
                ):
                    setattr(stats, name, getattr(stats, name) + cleaning[name])
                if rejected_reason:
                    setattr(stats, rejected_reason, getattr(stats, rejected_reason) + 1)
                    continue
                keep_indices.append(row_index)
                cleaned_values.append(cleaned)

            if not keep_indices:
                continue
            table = pa.Table.from_batches([batch], schema=schema)
            table = table.take(pa.array(keep_indices, type=pa.int64()))
            table = table.set_column(
                text_index,
                text_field,
                pa.array(cleaned_values, type=text_field.type),
            )
            writer.write_table(table, row_group_size=cfg["row_group_size"])
            stats.rows_out += table.num_rows

        writer.close()
        writer = None
        os.replace(temp_path, output_path)
    except Exception as exc:
        stats.error = f"{type(exc).__name__}: {exc}"
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if temp_path.exists():
            temp_path.unlink()
    return stats


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or is_within(output_dir, input_dir):
        raise SystemExit("output-dir must be outside input-dir")

    files = sorted(path for path in input_dir.rglob("*.parquet") if path.is_file())
    if not files:
        raise SystemExit(f"No Parquet files found under: {input_dir}")
    relative_files = [path.relative_to(input_dir) for path in files]
    existing = [output_dir / path for path in relative_files if (output_dir / path).exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Refusing to replace {len(existing)} existing output file(s); "
            "use an empty output-dir or pass --overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "text_column": args.text_column,
        "batch_size": args.batch_size,
        "row_group_size": args.row_group_size,
        "min_chars": args.min_chars,
        "min_quality_score": args.min_quality_score,
        "max_suspicious_ratio": args.max_suspicious_ratio,
        "compression": args.compression,
        "compression_level": args.compression_level,
        "keep_front_matter": args.keep_front_matter,
        "keep_tail_sections": args.keep_tail_sections,
        "arrow_use_threads": args.arrow_use_threads,
    }

    totals = Counter()
    results: list[FileStats] = []
    start = time.time()
    tasks = [(str(path), cfg) for path in relative_files]
    context = mp.get_context("spawn")
    with context.Pool(processes=min(args.workers, len(tasks))) as pool:
        for stats in pool.imap_unordered(process_one_file, tasks):
            results.append(stats)
            status = f"ERROR={stats.error}" if stats.error else "OK"
            print(
                f"{stats.path}: {status} rows_in={stats.rows_in:,} "
                f"rows_out={stats.rows_out:,} rejected={stats.rows_in-stats.rows_out:,} "
                f"noise_lines={stats.noise_lines_removed:,} "
                f"tail_sections={stats.tail_sections_removed:,}",
                flush=True,
            )
            for field in fields(FileStats):
                if field.name not in {"path", "error"}:
                    totals[field.name] += getattr(stats, field.name)

    failures = [stats for stats in results if stats.error]
    summary = {
        "version": VERSION,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(files),
        "successful_files": len(files) - len(failures),
        "failed_files": len(failures),
        "elapsed_seconds": round(time.time() - start, 3),
        "totals": dict(totals),
        "failures": [asdict(stats) for stats in failures],
        "parameters": {
            key: value for key, value in cfg.items()
            if key not in {"input_dir", "output_dir"}
        },
    }
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

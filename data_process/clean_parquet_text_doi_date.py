#!/usr/bin/env python3
"""Filter Parquet papers by DOI publication date, then clean their text.

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
import sqlite3
import sys
import time
import unicodedata
from calendar import monthrange
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "1.1.0"

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
    missing_doi: int = 0
    doi_not_in_date_db: int = 0
    invalid_publication_date: int = 0
    ambiguous_publication_date: int = 0
    published_on_or_after_cutoff: int = 0
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
    parser.add_argument("--doi-column", default="doi")
    parser.add_argument(
        "--doi-date-db",
        required=True,
        type=Path,
        help=(
            "Read-only SQLite DB containing doi_dates(doi TEXT PRIMARY KEY, "
            "publication_date TEXT)"
        ),
    )
    parser.add_argument(
        "--cutoff-date",
        default="2020-12-30",
        help="Keep only papers certainly published before this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--unknown-date-policy",
        choices=("drop", "keep", "error"),
        default="drop",
        help="Action for missing DOI/date or incomplete dates crossing the cutoff",
    )
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


DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)
DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")


def normalize_doi(value: object) -> str | None:
    """Normalize DOI URLs/prefixes to the canonical lowercase DOI string."""
    if not isinstance(value, str):
        return None
    doi = unquote(value).strip().lower()
    doi = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi\s*:\s*", "", doi, flags=re.I)
    doi = doi.strip().rstrip(".,;")
    if not DOI_RE.fullmatch(doi):
        return None
    return doi


def parse_cutoff_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid cutoff date {value!r}; expected YYYY-MM-DD"
        ) from exc
    return parsed


def classify_publication_date(value: object, cutoff: date) -> str:
    """Return before/after/ambiguous/invalid for YYYY[-MM[-DD]].

    Partial dates are interpreted as intervals. A paper is classified as
    ``before`` only when the entire possible interval is before the cutoff.
    """
    if not isinstance(value, str):
        return "invalid"
    match = DATE_RE.fullmatch(value.strip())
    if not match:
        return "invalid"
    year = int(match.group(1))
    month_text = match.group(2)
    day_text = match.group(3)
    try:
        if month_text is None:
            earliest = date(year, 1, 1)
            latest = date(year, 12, 31)
        elif day_text is None:
            month = int(month_text)
            earliest = date(year, month, 1)
            latest = date(year, month, monthrange(year, month)[1])
        else:
            exact = date(year, int(month_text), int(day_text))
            earliest = latest = exact
    except ValueError:
        return "invalid"

    if latest < cutoff:
        return "before"
    if earliest >= cutoff:
        return "after"
    return "ambiguous"


class DoiDateLookup:
    """Batch reader for a static, indexed DOI-to-publication-date SQLite DB."""

    def __init__(self, database_path: str):
        database_uri = Path(database_path).resolve().as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(
            database_uri,
            uri=True,
            timeout=60.0,
        )
        self.connection.execute("PRAGMA query_only=ON")
        # Fail early with a clear error when the expected schema is absent.
        self.connection.execute(
            "SELECT doi, publication_date FROM doi_dates LIMIT 0"
        ).fetchall()

    def close(self) -> None:
        self.connection.close()

    def get_many(self, dois: set[str]) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        values = list(dois)
        # Stay below SQLite's common 999-variable limit.
        for start in range(0, len(values), 900):
            chunk = values[start:start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT doi, publication_date FROM doi_dates "
                f"WHERE doi IN ({placeholders})",
                chunk,
            ).fetchall()
            result.update((doi, publication_date) for doi, publication_date in rows)
        return result


def apply_date_filter(
    doi_value: object,
    publication_dates: dict[str, str | None],
    cutoff: date,
    unknown_policy: str,
) -> tuple[bool, str | None]:
    doi = normalize_doi(doi_value)
    if doi is None:
        reason = "missing_doi"
    elif doi not in publication_dates:
        reason = "doi_not_in_date_db"
    else:
        classification = classify_publication_date(publication_dates[doi], cutoff)
        if classification == "before":
            return True, None
        if classification == "after":
            return False, "published_on_or_after_cutoff"
        reason = (
            "ambiguous_publication_date"
            if classification == "ambiguous"
            else "invalid_publication_date"
        )

    if unknown_policy == "keep":
        return True, reason
    if unknown_policy == "error":
        raise ValueError(
            f"cannot determine publication date: reason={reason}, DOI={doi_value!r}"
        )
    return False, reason


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
    doi_lookup: DoiDateLookup | None = None

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
        doi_index = schema.get_field_index(cfg["doi_column"])
        if doi_index < 0:
            raise ValueError(f"column {cfg['doi_column']!r} not found")
        doi_field = schema.field(doi_index)
        if not (pa.types.is_string(doi_field.type) or pa.types.is_large_string(doi_field.type)):
            raise TypeError(
                f"column {cfg['doi_column']!r} must be string/large_string, got {doi_field.type}"
            )
        doi_lookup = DoiDateLookup(cfg["doi_date_db"])
        cutoff = date.fromisoformat(cfg["cutoff_date"])

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
            doi_values = batch.column(doi_index).to_pylist()
            normalized_dois = {
                doi for value in doi_values if (doi := normalize_doi(value)) is not None
            }
            publication_dates = doi_lookup.get_many(normalized_dois)
            keep_indices: list[int] = []
            cleaned_values: list[str] = []

            for row_index, (value, doi_value) in enumerate(zip(values, doi_values)):
                keep_by_date, date_reason = apply_date_filter(
                    doi_value,
                    publication_dates,
                    cutoff,
                    cfg["unknown_date_policy"],
                )
                if date_reason:
                    setattr(stats, date_reason, getattr(stats, date_reason) + 1)
                if not keep_by_date:
                    continue
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
        doi_lookup.close()
        doi_lookup = None
        os.replace(temp_path, output_path)
    except Exception as exc:
        stats.error = f"{type(exc).__name__}: {exc}"
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if doi_lookup is not None:
            try:
                doi_lookup.close()
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
    doi_date_db = args.doi_date_db.resolve()
    try:
        cutoff = parse_cutoff_date(args.cutoff_date)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not doi_date_db.is_file():
        raise SystemExit(f"DOI date database does not exist: {doi_date_db}")
    try:
        database_check = DoiDateLookup(str(doi_date_db))
        database_check.close()
    except sqlite3.Error as exc:
        raise SystemExit(f"Invalid DOI date database: {exc}") from exc
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
        "doi_column": args.doi_column,
        "doi_date_db": str(doi_date_db),
        "cutoff_date": cutoff.isoformat(),
        "unknown_date_policy": args.unknown_date_policy,
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
                f"date_filtered={stats.published_on_or_after_cutoff:,} "
                f"unknown_date={stats.missing_doi + stats.doi_not_in_date_db + stats.invalid_publication_date + stats.ambiguous_publication_date:,} "
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

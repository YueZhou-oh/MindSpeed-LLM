#!/usr/bin/env python3
"""Clean ``fullText`` from JSONL files and write sharded text Parquet files.

Each input line must be one complete JSON object. This stage deliberately does
not tokenize: MindSpeed-LLM's ``preprocess_data.py`` performs Qwen tokenization
when it creates the indexed ``.bin/.idx`` dataset.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq


RAW_SCHEMA = pa.schema([pa.field("text", pa.string(), nullable=False)])


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
    lines: int = 0
    kept: int = 0
    empty_or_short: int = 0
    malformed: int = 0
    missing_fulltext: int = 0
    low_quality: int = 0
    controls_removed: int = 0
    noise_lines_removed: int = 0
    boilerplate_lines_removed: int = 0
    glyph_repairs: int = 0
    dehyphenations: int = 0
    front_matter_removed: int = 0
    tail_sections_removed: int = 0
    duplicate_abstracts_skipped: int = 0
    shards: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/dpc-zhouy/zhouy/papers/core-EN", type=Path)
    parser.add_argument("--output-dir", default="/dpc-zhouy/zhouy/papers/core-parquet-min100-fullnabstract-52m-clean", type=Path)
    parser.add_argument("--field", default="fullText")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--rows-per-shard", type=int, default=500_000)
    parser.add_argument("--row-group-size", type=int, default=10_000)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--min-quality-score", type=float, default=0.55)
    parser.add_argument("--max-suspicious-ratio", type=float, default=0.0003)
    parser.add_argument(
        "--keep-front-matter",
        action="store_true",
        help="Keep authors and affiliations before Abstract/Background/Introduction",
    )
    parser.add_argument(
        "--keep-tail-sections",
        action="store_true",
        help="Keep acknowledgements, funding, references, and similar tail sections",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    # NFC preserves scientific symbols more faithfully than compatibility
    # normalization (NFKC), while canonicalizing Unicode composition.
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
        if category == "Cc" and char not in "\n\t\f":
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
    # Some PDF font maps extract '=' as '¼'. Only repair it directly before a
    # numeric value; a global ¼ -> = replacement would corrupt genuine fractions.
    text, replacements = re.subn(
        r"¼(?=\s*[+\-−]?(?:\d|\.\d))",
        "=",
        text,
    )
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

    # Repeated running titles usually differ only by their page number.
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
    """Keep a probable title but remove author/affiliation blocks."""
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
    """Remove administrative and bibliography sections only in the last half."""
    start = int(len(lines) * 0.50)
    for index in range(start, len(lines)):
        if normalize_heading(lines[index]) in TAIL_HEADINGS:
            cleaning["tail_sections_removed"] += 1
            return lines[:index]
    return lines


def should_dehyphenate(left_word: str, right_word: str) -> bool:
    original_left = left_word.rstrip("-")
    left = original_left.lower()
    right = right_word.lower()
    compound = f"{left}-{right}"
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
                # Always concatenate the wrapped continuation without a space.
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

    length_score = min(1.0, len(words) / 500.0)
    printable_score = max(0.0, min(1.0, (printable_ratio - 0.94) / 0.06))
    alphabetic_score = max(0.0, min(1.0, (alphabetic_ratio - 0.35) / 0.30))
    english_score = max(0.0, min(1.0, english_ratio / 0.10))
    diversity_score = max(0.0, min(1.0, unique_ratio / 0.20))
    corruption_score = max(0.0, 1.0 - 1000.0 * (replacement_ratio + suspicious_ratio))
    structure_score = min(1.0, math.log2(paragraph_count + 1) / 4.0)

    score = (
        0.15 * length_score
        + 0.15 * printable_score
        + 0.15 * alphabetic_score
        + 0.15 * english_score
        + 0.10 * diversity_score
        + 0.20 * corruption_score
        + 0.10 * structure_score
    )
    return score, suspicious_ratio


def clean_fulltext(text: str, cfg: dict) -> tuple[str | None, Counter]:
    cleaning = Counter()
    text, controls_removed = remove_control_characters(text)
    cleaning["controls_removed"] = controls_removed
    text = repair_high_confidence_pdf_glyphs(text, cleaning)
    lines = filter_layout_lines(text.split("\n"), cleaning)
    if not cfg["keep_front_matter"]:
        lines = strip_front_matter(lines, cleaning)
    if not cfg["keep_tail_sections"]:
        lines = strip_tail_sections(lines, cleaning)
    cleaned = join_wrapped_lines(lines, cleaning)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return None, cleaning

    score, suspicious_ratio = calculate_quality(cleaned)
    cleaning["quality_score_milli"] = round(score * 1000)
    cleaning["suspicious_ratio_million"] = round(suspicious_ratio * 1_000_000)
    if score < cfg["min_quality_score"] or suspicious_ratio > cfg["max_suspicious_ratio"]:
        cleaning["low_quality"] += 1
        return None, cleaning
    return cleaned, cleaning


def combine_abstract_and_fulltext(
    abstract: str | None, fulltext: str, cleaning: Counter
) -> str:
    if not abstract:
        return fulltext
    normalized_abstract = re.sub(r"\s+", " ", abstract).strip().lower()
    normalized_prefix = re.sub(r"\s+", " ", fulltext[:20_000]).lower()
    probe = normalized_abstract[: min(300, len(normalized_abstract))]
    if len(probe) >= 80 and probe in normalized_prefix:
        cleaning["duplicate_abstracts_skipped"] += 1
        return fulltext
    return abstract + "\n\n" + fulltext


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
            if fulltext is None:
                stats.missing_fulltext += 1
                # continue
            else:
                # Clean only fullText after normalize_text(), as requested. Abstract
                # is normalized but not layout-cleaned because it is usually metadata.
                fulltext, cleaning = clean_fulltext(fulltext, cfg)
                for name in (
                    "controls_removed",
                    "noise_lines_removed",
                    "boilerplate_lines_removed",
                    "glyph_repairs",
                    "dehyphenations",
                    "front_matter_removed",
                    "tail_sections_removed",
                ):
                    setattr(stats, name, getattr(stats, name) + cleaning[name])
                if fulltext is None:
                    stats.low_quality += 1
                # continue

                # text = combine_abstract_and_fulltext(abstract, fulltext, cleaning)
                stats.duplicate_abstracts_skipped += cleaning["duplicate_abstracts_skipped"]
            
            text = "\n\n".join(
                value
                for value in (abstract, fulltext)
                if isinstance(value, str) and value
            )

            if len(text) < cfg["min_chars"]:
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
        "min_quality_score": args.min_quality_score,
        "max_suspicious_ratio": args.max_suspicious_ratio,
        "keep_front_matter": args.keep_front_matter,
        "keep_tail_sections": args.keep_tail_sections,
    }
    tasks = [(index, str(path), cfg) for index, path in enumerate(files)]
    totals = FileStats(path="TOTAL")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(args.workers, len(tasks))) as pool:
        for stats in pool.imap_unordered(process_one_file, tasks):
            print(
                f"{stats.path}: lines={stats.lines:,} kept={stats.kept:,} "
                f"malformed={stats.malformed:,} missing={stats.missing_fulltext:,} "
                f"short={stats.empty_or_short:,} low_quality={stats.low_quality:,} "
                f"noise_lines={stats.noise_lines_removed:,} "
                f"boilerplate={stats.boilerplate_lines_removed:,} "
                f"glyph_repairs={stats.glyph_repairs:,} "
                f"dehyphenated={stats.dehyphenations:,}",
                flush=True,
            )
            for name in (
                "lines",
                "kept",
                "empty_or_short",
                "malformed",
                "missing_fulltext",
                "low_quality",
                "controls_removed",
                "noise_lines_removed",
                "boilerplate_lines_removed",
                "glyph_repairs",
                "dehyphenations",
                "front_matter_removed",
                "tail_sections_removed",
                "duplicate_abstracts_skipped",
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
        f"short={totals.empty_or_short:,} low_quality={totals.low_quality:,} "
        f"controls_removed={totals.controls_removed:,} "
        f"noise_lines={totals.noise_lines_removed:,} "
        f"boilerplate={totals.boilerplate_lines_removed:,} "
        f"glyph_repairs={totals.glyph_repairs:,} "
        f"dehyphenated={totals.dehyphenations:,} "
        f"front_matter_removed={totals.front_matter_removed:,} "
        f"tail_sections_removed={totals.tail_sections_removed:,} "
        f"duplicate_abstracts_skipped={totals.duplicate_abstracts_skipped:,} "
        f"shards={len(parts):,}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

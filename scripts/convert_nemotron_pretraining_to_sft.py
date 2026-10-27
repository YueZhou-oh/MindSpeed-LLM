#!/usr/bin/env python3
"""Convert NVIDIA Nemotron-Pretraining-Specialized-v1.2 to SFT JSONL.

The source dataset stores pretraining-ready Q&A documents in a single ``text``
field.  This converter parses each of the four dataset subsets separately and
writes either ShareGPT or Alpaca records.  Processing is streaming and output
is sharded, so it can be used on the full dataset without loading it in RAM.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


CATEGORIES = {
    "Nemotron-Pretraining-Fact-Seeking": "fact_seeking",
    "Nemotron-Pretraining-Moral-Scenarios": "moral_scenarios",
    "Nemotron-Pretraining-Generative": "generative",
    "Nemotron-Pretraining-Multiple-Choice": "multiple_choice",
    "fact_seeking": "fact_seeking",
    "moral_scenarios": "moral_scenarios",
    "generative": "generative",
    "multiple_choice": "multiple_choice",
}

DEFAULT_SYSTEM = {
    "fact_seeking": "Answer the user's question accurately.",
    "moral_scenarios": "Answer the multiple-choice question about the moral scenario.",
    "generative": "Answer the user's question accurately and explain the reasoning when useful.",
    "multiple_choice": "Answer the multiple-choice question and explain the answer when useful.",
}

ANSWER_LINE_RE = re.compile(r"(?m)^Answer:\s*([^\n]*)(?:\n|$)")
MORAL_QUESTION_RE = re.compile(r"(?m)^(?:Quesion|Question):\s*")
GENERATIVE_QUESTION_RE = re.compile(r"(?m)^\s*\*\*Question\*\*:\s*")
GENERATIVE_RESPONSE_RE = re.compile(
    r"(?m)^\s*(?:"
    r"\*\*Thoughts\*\*:|"
    r"\*\*(?:Final )?Answer\*\*:|"
    r"\*\*Response\*\*:|"
    r"#{1,6}\s+(?:Final\s+)?Answer\s*:?[ \t]*|"
    r"(?:Final\s+)?Answer\s*:|"
    r"Response\s*:"
    r")\s*"
)
MCQ_CONTEXT_RE = re.compile(
    r"(?im)^Contextual Information and Knowledge:\s*"
)


class Reject(ValueError):
    """A source record cannot be converted safely."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass
class Example:
    system: str
    user: str
    assistant: str


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        raise Reject("missing_text", "text must be a string")
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        raise Reject("empty_text")
    return value


def split_answer(text: str, *, require_one: bool = True) -> tuple[str, str, str]:
    matches = list(ANSWER_LINE_RE.finditer(text))
    if not matches:
        raise Reject("missing_answer_marker")
    if require_one and len(matches) != 1:
        raise Reject("multiple_answer_markers", f"found {len(matches)} Answer markers")
    match = matches[-1] if require_one else matches[0]
    prompt = text[: match.start()].strip()
    answer = match.group(1).strip()
    tail = text[match.end() :].strip()
    if not prompt or not answer:
        raise Reject("empty_prompt_or_response")
    return prompt, answer, tail


def parse_fact_seeking(text: str, system_override: str | None) -> list[Example]:
    prompt, answer, tail = split_answer(text)
    if tail:
        raise Reject("unexpected_text_after_answer")
    return [Example(system_override or DEFAULT_SYSTEM["fact_seeking"], prompt, answer)]


def parse_multiple_choice(text: str, system_override: str | None) -> list[Example]:
    prompt, answer, tail = split_answer(text, require_one=False)
    context_match = MCQ_CONTEXT_RE.match(tail)
    if context_match:
        tail = tail[context_match.end() :].strip()
    response = answer if not tail else f"{answer}\n\n{tail}"
    return [
        Example(system_override or DEFAULT_SYSTEM["multiple_choice"], prompt, response)
    ]


def parse_moral_scenarios(text: str, system_override: str | None) -> list[Example]:
    starts = list(MORAL_QUESTION_RE.finditer(text))
    if not starts:
        raise Reject("missing_moral_question_marker")

    source_intro = text[: starts[0].start()].strip()
    system = system_override or source_intro or DEFAULT_SYSTEM["moral_scenarios"]
    examples: list[Example] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.end() : end].strip()
        prompt, answer, tail = split_answer(block)
        response = answer if not tail else f"{answer}\n\n{tail}"
        examples.append(Example(system, prompt, response))
    return examples


def parse_generative(text: str, system_override: str | None) -> list[Example]:
    question_markers = list(GENERATIVE_QUESTION_RE.finditer(text))
    if len(question_markers) != 1:
        raise Reject(
            "generative_question_count",
            f"expected one **Question** marker, found {len(question_markers)}",
        )
    body = text[question_markers[0].end() :].strip()
    response_match = GENERATIVE_RESPONSE_RE.search(body)
    if not response_match:
        raise Reject("missing_generative_response_marker")
    prompt = body[: response_match.start()].strip()
    response = body[response_match.end() :].strip()
    if not prompt or not response:
        raise Reject("empty_prompt_or_response")
    return [
        Example(system_override or DEFAULT_SYSTEM["generative"], prompt, response)
    ]


PARSERS = {
    "fact_seeking": parse_fact_seeking,
    "moral_scenarios": parse_moral_scenarios,
    "generative": parse_generative,
    "multiple_choice": parse_multiple_choice,
}


def canonical_category(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if value in CATEGORIES:
        return CATEGORIES[value]
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    for name, category in CATEGORIES.items():
        name_normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if normalized == name_normalized or name_normalized in normalized:
            return category
    return None


def infer_category(row: dict[str, Any], source: Path, override: str) -> str:
    if override != "auto":
        return override
    metadata = row.get("metadata")
    candidates: list[Any] = []
    if isinstance(metadata, dict):
        candidates.append(metadata.get("category"))
    candidates.extend([row.get("category"), str(source)])
    for candidate in candidates:
        category = canonical_category(candidate)
        if category:
            return category

    text = row.get("text", "")
    if "**Question**:" in text:
        return "generative"
    if re.search(r"(?m)^(?:Quesion|Question):", text) and text.count("\nAnswer:") > 1:
        return "moral_scenarios"
    if "Contextual Information and Knowledge:" in text or "\nOptions:" in text:
        return "multiple_choice"
    if "\nAnswer:" in text:
        return "fact_seeking"
    raise Reject("unknown_category")


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def iter_parquet(path: Path, batch_size: int) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading Parquet requires pyarrow: pip install pyarrow"
        ) from exc
    parquet = pq.ParquetFile(path)
    index = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            index += 1
            yield index, row


def input_files(inputs: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(inputs, (str, Path)):
        input_paths = [Path(inputs)]
    else:
        input_paths = [Path(path) for path in inputs]

    found: list[Path] = []

    for path in input_paths:
        if path.is_file() and path.suffix.lower() in {".jsonl", ".parquet"}:
            found.append(path)
        elif path.is_dir():
            found.extend(path.rglob("*.jsonl"))
            found.extend(path.rglob("*.parquet"))
        else:
            raise FileNotFoundError(f"No supported input found at {path}")
    unique = sorted({path.resolve() for path in found})
    if not unique:
        raise FileNotFoundError("No .jsonl or .parquet files found")
    return unique


def rows_from(path: Path, batch_size: int) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".parquet":
        yield from iter_parquet(path, batch_size)
    else:
        yield from iter_jsonl(path)


def output_record(
    example: Example,
    row: dict[str, Any],
    output_format: str,
    source: Path,
    source_index: int,
    expansion_index: int,
    metadata_mode: str,
) -> dict[str, Any]:
    if output_format == "sharegpt":
        result: dict[str, Any] = {
            "system": example.system,
            "conversations": [
                {"from": "human", "value": example.user},
                {"from": "gpt", "value": example.assistant},
            ],
        }
    else:
        result = {
            "system": example.system,
            "instruction": example.user,
            "input": "",
            "output": example.assistant,
        }

    if metadata_mode in {"compact", "full"}:
        for key in ("uuid", "license"):
            if key in row:
                result[key] = row[key]
    if metadata_mode == "full":
        if "metadata" in row:
            result["metadata"] = row["metadata"]
        result["conversion_metadata"] = {
            "source_file": str(source),
            "source_row": source_index,
            "expansion_index": expansion_index,
        }
    return result


class ShardedWriter:
    def __init__(self, root: Path, output_format: str, rows_per_shard: int) -> None:
        self.root = root
        self.output_format = output_format
        self.rows_per_shard = rows_per_shard
        self.counts: Counter[str] = Counter()
        self.handles: dict[str, Any] = {}
        self.current_shards: dict[str, int] = {}

    def _open(self, category: str, shard: int) -> Any:
        category_dir = self.root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        path = category_dir / f"{category}_{self.output_format}_{shard:05d}.jsonl"
        return path.open("w", encoding="utf-8")

    def write(self, category: str, record: dict[str, Any]) -> None:
        count = self.counts[category]
        shard = count // self.rows_per_shard
        if self.current_shards.get(category) != shard:
            old = self.handles.pop(category, None)
            if old:
                old.close()
            self.handles[category] = self._open(category, shard)
            self.current_shards[category] = shard
        self.handles[category].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[category] += 1

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=Path("/dpc-zhouy/zhouy/post-training-data/Nemotron-Pretraining-Specialized-v1.2"),
        help="Input JSONL/Parquet file(s) or directories (searched recursively).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/dpc-zhouy/zhouy/post-training-data/nemotron_specialized_sharegpt_convert"))
    parser.add_argument("--format", choices=("sharegpt", "alpaca"), default="sharegpt")
    parser.add_argument(
        "--category",
        choices=("auto", *PARSERS.keys()),
        default="auto",
        help="Override category detection when processing a single known subset.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Use one system prompt for every output record.",
    )
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--parquet-batch-size", type=int, default=8192)
    parser.add_argument(
        "--metadata-mode",
        choices=("none", "compact", "full"),
        default="compact",
        help="compact keeps uuid/license; full also keeps source metadata/location.",
    )
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=100_000)
    parser.add_argument(
        "--rejected-jsonl",
        type=Path,
        default=None,
        help="Optional audit log; stores identifiers and reasons, not full text.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rows_per_shard < 1 or args.parquet_batch_size < 1:
        raise SystemExit("Shard and batch sizes must be positive")

    files = input_files(args.input)
    output_dir = args.output_dir.resolve()
    existing = list(output_dir.rglob("*.jsonl")) if output_dir.exists() else []
    if existing:
        raise SystemExit(
            f"Refusing to mix with {len(existing)} existing JSONL file(s) in {output_dir}; "
            "choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rejected_handle = None
    if args.rejected_jsonl:
        args.rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
        rejected_handle = args.rejected_jsonl.open("w", encoding="utf-8")

    writer = ShardedWriter(output_dir, args.format, args.rows_per_shard)
    stats: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    stop = False
    try:
        for source in files:
            print(f"Reading {source}", file=sys.stderr, flush=True)
            for source_index, row in rows_from(source, args.parquet_batch_size):
                stats["source_rows_read"] += 1
                try:
                    category = infer_category(row, source, args.category)
                    text = normalize_text(row.get("text"))
                    examples = PARSERS[category](text, args.system_prompt)
                    for expansion_index, example in enumerate(examples):
                        record = output_record(
                            example,
                            row,
                            args.format,
                            source,
                            source_index,
                            expansion_index,
                            args.metadata_mode,
                        )
                        writer.write(category, record)
                        stats["sft_examples_written"] += 1
                        stats[f"written_{category}"] += 1
                except Reject as exc:
                    stats["source_rows_rejected"] += 1
                    reasons[exc.reason] += 1
                    if rejected_handle:
                        rejection = {
                            "uuid": row.get("uuid"),
                            "source_file": str(source),
                            "source_row": source_index,
                            "reason": exc.reason,
                            "detail": exc.detail,
                        }
                        rejected_handle.write(json.dumps(rejection, ensure_ascii=False) + "\n")

                if args.log_interval and stats["source_rows_read"] % args.log_interval == 0:
                    print(
                        json.dumps(
                            {
                                "read": stats["source_rows_read"],
                                "written": stats["sft_examples_written"],
                                "rejected": stats["source_rows_rejected"],
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                if (
                    args.max_source_rows is not None
                    and stats["source_rows_read"] >= args.max_source_rows
                ):
                    stop = True
                    break
            if stop:
                break
    finally:
        writer.close()
        if rejected_handle:
            rejected_handle.close()

    report = {
        "format": args.format,
        "output_dir": str(output_dir),
        "input_files": len(files),
        **dict(stats),
        "rejection_reasons": dict(reasons),
        "output_counts_by_category": dict(writer.counts),
    }
    report_path = output_dir / "conversion_stats.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

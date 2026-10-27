#!/usr/bin/env bash
set -euo pipefail

# Build one MindSpeed-LLM SFT bin/idx pair from QingyiSi/Alpaca-CoT.
#
# Policy:
#   1. Keep only records whose instruction/input/output are English.
#   2. Exclude FLAN-Muffin, FLAN-derived CoT, Natural-Instructions/NIv2,
#      xP3/P3, and combined files that duplicate those sources.
#   3. Use gpt4all_without_p3.json instead of gpt4all.json.
#   4. Normalize accepted records to Alpaca {instruction,input,output,system}.
#   5. Send the single consolidated JSONL to preprocess_data.py once, thereby
#      producing one output prefix (one .bin/.idx pair per json key).

MIND_SPEED_ROOT="${MIND_SPEED_ROOT:-/dpc-zhouy/zhouy/MindSpeed-LLM}"
CONDA_SH="${CONDA_SH:-/dpc-zhouy/zhouy/miniconda3/bin/activate}"
CONDA_ENV="${CONDA_ENV:-py311}"
ASCEND_ENV="${ASCEND_ENV:-/dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh}"

# Change these paths for your machine.
INPUT_ROOT="${INPUT_ROOT:-/dpc-zhouy/zhouy/post-training-data/Alpaca-CoT}"
WORK_ROOT="${WORK_ROOT:-/dpc-zhouy/zhouy/post-training-data/alpaca_cot_english_no_flan}"
FILTERED_JSONL="${FILTERED_JSONL:-${WORK_ROOT}/alpaca_cot_english_no_flan.jsonl}"
STATS_JSON="${STATS_JSON:-${WORK_ROOT}/filter_stats.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-/dpc-zhouy/zhouy/instructft_data/alpaca_cot_4K/alpaca_cot_english_no_flan_no_moss}"
TOKENIZER="${TOKENIZER:-/dpc-zhouy/zhouy/ckpts/Qwen3-8B}"
CACHE_DIR="${CACHE_DIR:-/dpc-zhouy/cache}"

# Download once from:
# https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
LID_MODEL="${LID_MODEL:-/dpc-zhouy/zhouy/ckpts/lid.176.ftz}"
EN_THRESHOLD="${EN_THRESHOLD:-0.55}"
LANG_BATCH_SIZE="${LANG_BATCH_SIZE:-2048}"

WORKERS="${WORKERS:-32}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
LOG_INTERVAL="${LOG_INTERVAL:-100000}"
REBUILD_FILTERED="${REBUILD_FILTERED:-0}"
ALLOW_FAILED_FILES="${ALLOW_FAILED_FILES:-0}"

mkdir -p "$WORK_ROOT" "$(dirname "$OUTPUT_PREFIX")"

if [[ ! -d "$INPUT_ROOT" ]]; then
    echo "ERROR: INPUT_ROOT is not a directory: $INPUT_ROOT" >&2
    exit 1
fi

if [[ ! -f "$LID_MODEL" ]]; then
    echo "ERROR: fastText language model not found: $LID_MODEL" >&2
    echo "Download lid.176.ftz and set LID_MODEL=/path/to/lid.176.ftz" >&2
    exit 1
fi

cd "$MIND_SPEED_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
source "$ASCEND_ENV"

python - <<'PY'
missing = []
for module in ("ijson", "fasttext"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python packages: " + ", ".join(missing) +
        ". Install with: pip install ijson fasttext-wheel"
    )
PY

if [[ "$REBUILD_FILTERED" == "1" || ! -s "$FILTERED_JSONL" ]]; then
    PARTIAL_JSONL="${FILTERED_JSONL}.partial"
    if [[ -e "$PARTIAL_JSONL" ]]; then
        echo "ERROR: stale partial output exists: $PARTIAL_JSONL" >&2
        echo "Inspect or remove it before restarting the filtering stage." >&2
        exit 1
    fi

    echo "Filtering Alpaca-CoT"
    echo "Input root: $INPUT_ROOT"
    echo "Filtered output: $FILTERED_JSONL"

    python - "$INPUT_ROOT" "$PARTIAL_JSONL" "$STATS_JSON" \
        "$LID_MODEL" "$EN_THRESHOLD" "$LANG_BATCH_SIZE" "$LOG_INTERVAL" \
        "$ALLOW_FAILED_FILES" <<'PY'
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import fasttext
import ijson


input_root = Path(sys.argv[1]).resolve()
partial_output = Path(sys.argv[2]).resolve()
stats_path = Path(sys.argv[3]).resolve()
lid_model_path = Path(sys.argv[4]).resolve()
en_threshold = float(sys.argv[5])
batch_size = int(sys.argv[6])
log_interval = int(sys.argv[7])
allow_failed_files = sys.argv[8] == "1"

if not 0.0 <= en_threshold <= 1.0:
    raise ValueError("EN_THRESHOLD must be between 0 and 1")
if batch_size < 1:
    raise ValueError("LANG_BATCH_SIZE must be positive")

model = fasttext.load_model(str(lid_model_path))
stats = Counter()
file_stats = {}


def normalized_path(path):
    return path.relative_to(input_root).as_posix()


def source_exclusion_reason(relative):
    """Return a reason before opening a known unwanted source file."""
    low = "/" + relative.lower().replace("\\", "/")

    # Explicit FLAN data and datasets already represented in the user's FLAN
    # collection (CoT, NIv2 and P3/T0).
    if low.startswith("/flan-muffin/") or "/flan" in low:
        return "flan"
    if low.startswith("/chain-of-thought/") or "formatted_cot_data" in low:
        return "flan_cot"
    if low.startswith("/natural-instructions/"):
        return "flan_niv2_overlap"
    if low.startswith("/xp3/"):
        return "flan_p3_overlap"
    if low.startswith("/combination/"):
        return "combined_duplicate_or_flan"

    # gpt4all.json contains P3; the repository supplies a P3-free replacement.
    if low == "/gpt4all/gpt4all.json":
        return "flan_p3_overlap"

    # Large, clearly non-English sources. Mixed/uncertain sources that are not
    # caught here still go through per-record fastText language identification.
    non_english_components = (
        "/chinese-instruction-collection/",
        "/chinese-medical/",
        "/belle_cn/",
        "/firefly/",
        "/gaokao/",
        "/pclue/",
        "/csl/",
        "/coig/",
    )
    if any(component in low for component in non_english_components):
        return "non_english_source"

    name = Path(low).name
    if re.search(r"(?:^|[_-])(?:zh|cn|chinese)(?:[_-]|\.)", name):
        return "non_english_filename"
    return None


def iter_objects(path):
    """Stream a JSON array or a sequence of JSON objects."""
    with path.open("rb") as handle:
        first = b""
        while not first:
            chunk = handle.read(4096)
            if not chunk:
                return
            first = chunk.lstrip()[:1]
        handle.seek(0)
        prefix = "item" if first == b"[" else ""
        yield from ijson.items(handle, prefix, multiple_values=(first != b"["))


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def normalize_record(row):
    if not isinstance(row, dict):
        return None, "not_object"

    instruction = as_text(row.get("instruction"))
    input_text = as_text(row.get("input"))
    output = as_text(row.get("output"))

    # Accommodate a few common one-turn aliases without guessing about
    # multi-turn conversation structures.
    if not instruction:
        instruction = as_text(row.get("prompt")) or as_text(row.get("question"))
    if not output:
        output = as_text(row.get("response")) or as_text(row.get("answer"))

    if not instruction and input_text:
        instruction, input_text = input_text, ""
    if not instruction:
        return None, "missing_instruction"
    if not output:
        return None, "missing_output"

    system = as_text(row.get("system")) or as_text(row.get("system_prompt"))
    return {
        "system": system,
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }, None


def language_text(record):
    # Instruction and optional input are the strongest language signal. Include
    # some response text so very short prompts can still be classified.
    pieces = [record["instruction"][:2500], record["input"][:1000], record["output"][:1500]]
    return " ".join(piece.replace("\n", " ") for piece in pieces if piece).strip()


def contains_too_much_non_latin_script(text):
    letters = 0
    non_latin = 0
    for char in text:
        code = ord(char)
        if char.isalpha():
            letters += 1
            if (
                0x0400 <= code <= 0x052F      # Cyrillic
                or 0x0590 <= code <= 0x08FF   # Hebrew/Arabic
                or 0x0900 <= code <= 0x0DFF   # Indic scripts
                or 0x0E00 <= code <= 0x0FFF   # Thai/Lao/Tibetan
                or 0x3040 <= code <= 0x30FF   # Japanese kana
                or 0x3400 <= code <= 0x9FFF   # CJK
                or 0xAC00 <= code <= 0xD7AF   # Hangul
            ):
                non_latin += 1
    return letters >= 20 and non_latin / letters > 0.05


def flush_batch(pending, output_handle, relative, local_stats):
    if not pending:
        return
    texts = [item[1] for item in pending]
    labels, probabilities = model.predict(texts, k=1)
    for (record, text), item_labels, item_probabilities in zip(
        pending, labels, probabilities
    ):
        label = item_labels[0].removeprefix("__label__")
        probability = float(item_probabilities[0])
        if contains_too_much_non_latin_script(text):
            stats["rejected_non_english_script"] += 1
            local_stats["rejected_non_english_script"] += 1
        elif label != "en" or probability < en_threshold:
            stats["rejected_language"] += 1
            local_stats["rejected_language"] += 1
        else:
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            local_stats["written"] += 1
    pending.clear()


json_files = sorted(
    path for path in input_root.rglob("*")
    if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
)
stats["discovered_files"] = len(json_files)

partial_output.parent.mkdir(parents=True, exist_ok=True)
with partial_output.open("x", encoding="utf-8") as output_handle:
    for file_index, path in enumerate(json_files, 1):
        relative = normalized_path(path)
        exclusion = source_exclusion_reason(relative)
        if exclusion:
            stats["excluded_files"] += 1
            stats[f"excluded_file_{exclusion}"] += 1
            continue

        local_stats = Counter()
        pending = []
        try:
            for row in iter_objects(path):
                stats["read"] += 1
                local_stats["read"] += 1
                record, rejection = normalize_record(row)
                if rejection:
                    stats[f"rejected_{rejection}"] += 1
                    local_stats[f"rejected_{rejection}"] += 1
                    continue
                text = language_text(record)
                if not text:
                    stats["rejected_empty_language_text"] += 1
                    local_stats["rejected_empty_language_text"] += 1
                    continue
                pending.append((record, text))
                if len(pending) >= batch_size:
                    flush_batch(pending, output_handle, relative, local_stats)

                if log_interval and stats["read"] % log_interval == 0:
                    print(json.dumps({
                        "files_processed": stats["processed_files"],
                        "read": stats["read"],
                        "written": stats["written"],
                        "rejected_language": (
                            stats["rejected_language"] +
                            stats["rejected_non_english_script"]
                        ),
                    }), flush=True)
            flush_batch(pending, output_handle, relative, local_stats)
            stats["processed_files"] += 1
            file_stats[relative] = dict(local_stats)
            print(
                f"[{file_index}/{len(json_files)}] {relative}: "
                f"read={local_stats['read']} written={local_stats['written']}",
                flush=True,
            )
        except Exception as exc:
            stats["failed_files"] += 1
            file_stats[relative] = {
                **dict(local_stats),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"WARNING: failed to parse {relative}: {exc}", file=sys.stderr)

report = {
    "input_root": str(input_root),
    "output": str(partial_output.with_suffix("")),
    "language_model": str(lid_model_path),
    "english_probability_threshold": en_threshold,
    "summary": dict(stats),
    "files": file_stats,
}
stats_path.parent.mkdir(parents=True, exist_ok=True)
stats_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

if stats["written"] == 0:
    raise RuntimeError("No English SFT records were written")
if stats["failed_files"] and not allow_failed_files:
    raise RuntimeError(
        f"{stats['failed_files']} source file(s) failed to parse. "
        f"Review {stats_path}; then fix the input or rerun with "
        "ALLOW_FAILED_FILES=1 to knowingly omit them. The partial JSONL was retained."
    )

final_output = Path(str(partial_output).removesuffix(".partial"))
os.replace(partial_output, final_output)
print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
print(f"Filtered JSONL: {final_output}")
print(f"Statistics: {stats_path}")
PY
else
    echo "Reusing existing filtered JSONL: $FILTERED_JSONL"
fi

if [[ -e "${OUTPUT_PREFIX}_packed_input_ids_document.bin" || \
      -e "${OUTPUT_PREFIX}_packed_input_ids_document.idx" ]]; then
    echo "ERROR: output bin/idx already exists for prefix: $OUTPUT_PREFIX" >&2
    exit 1
fi

echo "Preprocessing one consolidated Alpaca JSONL"
echo "Output prefix: $OUTPUT_PREFIX"

python preprocess_data.py \
    --input "$FILTERED_JSONL" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path "$TOKENIZER" \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type qwen3 \
    --seq-length "$SEQ_LENGTH" \
    --workers "$WORKERS" \
    --log-interval "$LOG_INTERVAL" \
    --cache-dir "$CACHE_DIR"

echo "Finished. Expected files include:"
echo "  ${OUTPUT_PREFIX}_packed_input_ids_document.bin"
echo "  ${OUTPUT_PREFIX}_packed_input_ids_document.idx"

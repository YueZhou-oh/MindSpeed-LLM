#!/usr/bin/env bash
# Nemotron-SFT-Science-v2: one independent SFT bin/idx group per JSONL.
# Embeds the converter; run with bash. No separate Python script is needed.
set -euo pipefail

REPO="${REPO:-/dpc-zhouy/zhouy/MindSpeed-LLM}"
INPUT_ROOT="${INPUT_ROOT:-/dpc-zhouy/zhouy/post-training-data/Nemotron-SFT-Science-v2}"
CONVERTED_ROOT="${CONVERTED_ROOT:-/dpc-zhouy/zhouy/post-training-data/science_sft_jsonl_8K}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/dpc-zhouy/zhouy/instructft_data/science_bin_8K}"
TOKENIZER="${TOKENIZER:-/dpc-zhouy/zhouy/ckpts/Qwen3-8B}"
CACHE_DIR="${CACHE_DIR:-/dpc-zhouy/cache}"
WORKERS="${WORKERS:-32}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LIMIT="${LIMIT:-}"  # Optional input-record limit PER FILE for a smoke test.
SETUP_ENV="${SETUP_ENV:-1}"  # Set 0 if your environment is already activated.

if [[ "$SETUP_ENV" == 1 ]]; then
    # Vendor activation scripts can reference unset environment variables.
    set +u
    source /dpc-zhouy/zhouy/miniconda3/bin/activate
    conda activate py311
    source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
command -v "$PYTHON_BIN" >/dev/null
[[ -f "$REPO/preprocess_data.py" ]] || { echo "Missing preprocess_data.py: $REPO" >&2; exit 1; }
[[ -d "$INPUT_ROOT" ]] || { echo "Missing input directory: $INPUT_ROOT" >&2; exit 1; }
[[ -d "$TOKENIZER" ]] || { echo "Missing tokenizer: $TOKENIZER" >&2; exit 1; }
REPO=$(realpath "$REPO")
INPUT_ROOT=$(realpath "$INPUT_ROOT")
TOKENIZER=$(realpath "$TOKENIZER")
CONVERTED_ROOT=$(realpath -m "$CONVERTED_ROOT")
OUTPUT_ROOT=$(realpath -m "$OUTPUT_ROOT")
CACHE_DIR=$(realpath -m "$CACHE_DIR")
case "$CONVERTED_ROOT/" in "$INPUT_ROOT/"*) echo 'CONVERTED_ROOT must be outside INPUT_ROOT' >&2; exit 1;; esac
case "$OUTPUT_ROOT/" in "$INPUT_ROOT/"*) echo 'OUTPUT_ROOT must be outside INPUT_ROOT' >&2; exit 1;; esac
mkdir -p "$CONVERTED_ROOT" "$OUTPUT_ROOT" "$CACHE_DIR"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Keep a real Python script for multiprocessing spawn.
# Qwen3 thinking formatters expect tool_format="qwen3", not "qwen".
# Use one corrected copy for BOTH length checking and preprocessing.
export SCIENCE_TEMPLATE_PATH="$CONVERTED_ROOT/science_qwen3_templates.json"
"$PYTHON_BIN" - "$REPO/configs/finetune/templates.json" "$SCIENCE_TEMPLATE_PATH" <<'FIX_QWEN_TEMPLATE'
import json
import sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
config = json.loads(source.read_text(encoding="utf-8"))
found = False
for template in config:
    if template.get("name") != "qwen3":
        continue
    found = True
    for key in ("format_function", "format_tools"):
        if key not in template:
            raise ValueError(f"Qwen3 template is missing {key}")
        template[key]["tool_format"] = "qwen3"
if not found:
    raise ValueError("Qwen3 template not found")
destination.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Corrected Qwen3 tool template: {destination}")
FIX_QWEN_TEMPLATE

CONVERTER_FILE="$CONVERTED_ROOT/convert_science_sharegpt.py"
# Keep the file list in a Bash array; no temporary directory or list file.
mapfile -d '' -t INPUT_FILES < <(find "$INPUT_ROOT" -type f -name '*.jsonl' -print0 | sort -z)
[[ ${#INPUT_FILES[@]} -gt 0 ]] || { echo 'No JSONL input files found' >&2; exit 1; }

# Check every destination before starting, so existing files are never overwritten.
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    RELATIVE_PATH="${INPUT_FILE#"$INPUT_ROOT"/}"
    STEM="${RELATIVE_PATH%.jsonl}"
    CONVERTED_FILE="$CONVERTED_ROOT/$STEM.jsonl"
    OUTPUT_PREFIX="$OUTPUT_ROOT/$STEM"
    for existing in "$CONVERTED_FILE" "$CONVERTED_FILE.rejected.jsonl" "$CONVERTED_FILE.summary.json"; do
        [[ ! -e "$existing" ]] || { echo "Already exists: $existing; choose fresh output roots." >&2; exit 1; }
    done
    for key in input_ids attention_mask labels; do
        for ext in bin idx; do
            existing="${OUTPUT_PREFIX}_packed_${key}_document.${ext}"
            [[ ! -e "$existing" ]] || { echo "Already exists: $existing; choose fresh output roots." >&2; exit 1; }
        done
    done
done

cat > "$CONVERTER_FILE" <<'SCIENCE_CONVERTER_PY'
#!/usr/bin/env python3
"""Convert Nemotron Science to normalized ShareGPT JSONL for preprocess_data.py.
Uses installed MindSpeed Qwen3 template to reject complete conversations longer
than --seq-length - 1, before MindSpeed can truncate individual turns.
Does not create bins. Run the preprocess_data.py command shown in the answer.
Requires the same Python environment as MindSpeed-LLM. Never executes tool code.
Unsupported tool patterns are logged, not silently flattened.
"""
import argparse
import gzip
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from collections import Counter

def normalize(row):
    """Convert OpenAI tool calls to Qwen assistant/observation messages."""
    raw = row.get('messages')
    if not isinstance(raw, list) or not raw:
        raise ValueError('missing_messages')
    messages, systems = [], []
    pending = False
    for m in raw:
        role = m.get('role')
        content = m.get('content') or ''
        if not isinstance(content, str):
            raise ValueError('non_text_content')
        if role in ('system', 'developer') and not messages:
            # Qwen3 has no separate developer role. Leading system/developer
            # instructions have equivalent prompt precedence here, so combine
            # them into the single system message without discarding content.
            systems.append(content)
            continue
        if role == 'assistant':
            if pending:
                raise ValueError('missing_tool_response')
            reasoning = m.get('reasoning_content') or m.get('reasoning') or ''
            if not isinstance(reasoning, str):
                raise ValueError('non_text_reasoning')
            calls = m.get('tool_calls') or []
            if calls:
                # Standard ShareGPT templates assume alternating turns. Reject
                # parallel calls rather than silently mis-associate observations.
                if len(calls) != 1:
                    raise ValueError('parallel_tool_calls_not_supported')
                f = calls[0]['function']
                arguments = f['arguments']
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError('tool_arguments_not_object')
                # Some vendor records duplicate the structured tool call in
                # content using DSML. Ignore that redundant representation.
                # Ordinary prose accompanying a structured call is valid and
                # is retained before the Qwen tool-call block.
                prose = '' if '<｜DSML｜tool_calls>' in content else content.strip()
                payload = json.dumps(
                    {'name': f['name'], 'arguments': arguments},
                    ensure_ascii=False,
                )
                thought = '<think>\n' + reasoning + '\n</think>\n' if reasoning else ''
                # Preformat exactly once. Keeping role=assistant prevents
                # FunctionFormatterForThink from trying to json.loads() DSML.
                tool_text = thought
                if prose:
                    tool_text += prose + '\n'
                tool_text += '<tool_call>\n' + payload + '\n</tool_call>'
                messages.append({'role': 'assistant', 'content': tool_text})
                pending = calls[0].get('id') or True
            else:
                if reasoning:
                    if '<think>' in content:
                        raise ValueError('ambiguous_duplicate_reasoning')
                    content = '<think>\n' + reasoning + '\n</think>\n' + content
                if not content.strip():
                    raise ValueError('empty_assistant')
                messages.append({'role': role, 'content': content})
        elif role == 'tool':
            if not pending:
                raise ValueError('orphan_tool_response')
            if isinstance(pending, str) and m.get('tool_call_id') and m['tool_call_id'] != pending:
                raise ValueError('tool_id_mismatch')
            messages.append({'role': 'observation', 'content': content})
            pending = False
        elif role == 'user':
            if pending:
                raise ValueError('missing_tool_response')
            messages.append({'role': role, 'content': content})
        else:
            raise ValueError('unsupported_role_' + str(role))
    if pending or not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('missing_final_assistant')
    for i, m in enumerate(messages):
        allowed = ('user', 'observation') if i % 2 == 0 else ('assistant', 'function')
        if m['role'] not in allowed:
            raise ValueError('nonalternating_turns')
    if len(messages) % 2:
        raise ValueError('odd_message_count')
    schemas = row.get('tools') or []
    if isinstance(schemas, str):
        schemas = json.loads(schemas)
    schemas = [t.get('function', t) for t in schemas]
    if any('<tool_call>' in m['content'] for m in messages) and not schemas:
        raise ValueError('missing_tool_schemas')
    return messages, '\n\n'.join(systems), json.dumps(schemas, ensure_ascii=False)

def initialize(repo, tokenizer_path, template_path, seq_length):
    global TOK, TEMPLATE, LIMIT
    sys.path.insert(0, repo)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    from mindspeed_llm.tasks.preprocess.templates import get_model_template, fix_model_tokenizer
    TOK = AutoTokenizer.from_pretrained(tokenizer_path)
    n_before = len(TOK)
    fix_model_tokenizer(TOK, 'qwen3', template_path, True)
    if len(TOK) != n_before:
        raise RuntimeError('Template adds vocabulary entries: align tokenizer/model before preprocessing')
    TEMPLATE = get_model_template('qwen3', template_path, True)
    LIMIT = seq_length
    if TOK.pad_token_id is None:
        raise RuntimeError('Tokenizer must define pad_token_id')

def encode(item):
    source, row = item
    identity = {'source': source, 'uuid': row.get('uuid')}
    try:
        messages, system, schemas = normalize(row)
    except (ValueError, KeyError, TypeError) as e:
        return identity, None, str(e)
    # Very large cutoff avoids per-turn truncation before the whole-dialogue check.
    pairs = TEMPLATE.encode_multiturn(TOK, messages, system, schemas, cutoff_len=10**9)
    ids, labels = [], []
    for prompt, answer in pairs:
        mask = [-100] * len(prompt)
        if ids and TEMPLATE.efficient_eos and mask:
            mask[0] = TOK.eos_token_id
        ids.extend(prompt)
        ids.extend(answer)
        labels.extend(mask)
        labels.extend(answer)
    if TEMPLATE.efficient_eos:
        ids.append(TOK.eos_token_id)
        labels.append(TOK.eos_token_id)
    if len(ids) > LIMIT - 1:
        return identity, None, 'overlength:' + str(len(ids))
    if not ids or not any(x != -100 for x in labels):
        return identity, None, 'no_supervised_tokens'
    return identity, (ids, labels), None

def records(args):
    path = Path(args.input)
    if path.exists():
        files = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() and (p.name.endswith(('.jsonl', '.jsonl.gz', '.json', '.parquet'))))
        if not files:
            raise ValueError('No local JSONL/JSON/Parquet files found')
        for p in files:
            if p.suffix == '.parquet':
                import pyarrow.parquet as pq
                for batch in pq.ParquetFile(p).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        yield str(p), row
            else:
                opener = gzip.open if p.name.endswith('.gz') else open
                with opener(p, 'rt', encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        if line.strip():
                            row = json.loads(line)
                            if not isinstance(row, dict):
                                raise ValueError(f'{p}:{line_no}: expected one JSON object per line')
                            yield f'{p}:{line_no}', row
    else:
        if args.input != 'nvidia/Nemotron-SFT-Science-v2':
            raise ValueError('Input path does not exist; only the explicit HF dataset ID enables online reading')
        from datasets import get_dataset_config_names, load_dataset
        configs = args.subsets or get_dataset_config_names(args.input)
        for config in configs:
            ds = load_dataset(args.input, name=config, split='train', streaming=True)
            for row in ds:
                yield config, row

def convert(item):
    identity, encoded, reason = encode(item)
    if reason:
        return identity, None, reason
    source, row = item
    messages, system, tools = normalize(row)
    result = {"messages": messages, "system": system, "tools": tools,
              "uuid": str(row.get("uuid") or ""),
              "metadata_json": json.dumps(row.get("metadata"), ensure_ascii=False),
              "license_json": json.dumps(row.get("license"), ensure_ascii=False),
              "source": source}
    return identity, result, None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Local JSONL/JSONL.GZ/Parquet path, or nvidia/Nemotron-SFT-Science-v2')
    p.add_argument('--output', required=True, help='New output JSONL file')
    p.add_argument('--repo', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--seq-length', type=int, default=4096)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int)
    p.add_argument('--subsets', nargs='+', help='HF config names; online input only')
    a=p.parse_args()
    if a.workers < 1 or a.seq_length < 2: p.error('Invalid workers or sequence length')
    output=Path(a.output).resolve()
    source=Path(a.input)
    if source.is_dir() and source.resolve() in output.parents:
        p.error('Place output outside the input directory to avoid reading generated output')
    paths=[output,Path(str(output)+'.rejected.jsonl'),Path(str(output)+'.summary.json')]
    if any(x.exists() for x in paths): raise FileExistsError('Choose a fresh output path')
    output.parent.mkdir(parents=True,exist_ok=True)
    repo=str(Path(a.repo).resolve())
    template_path=os.environ.get('SCIENCE_TEMPLATE_PATH') or str(Path(repo)/'configs/finetune/templates.json')
    if not Path(template_path).is_file(): raise FileNotFoundError(template_path)
    initargs=(repo,a.tokenizer,template_path,a.seq_length)
    initialize(*initargs)  # fail early on dependency/template errors
    rows=records(a)
    if a.limit is not None: rows=itertools.islice(rows,a.limit)
    counts,reasons=Counter(),Counter()
    with mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=initargs) as pool, output.open('w',encoding='utf-8') as out, paths[1].open('w',encoding='utf-8') as rejects:
        for identity,row,reason in pool.imap(convert,rows,chunksize=16):
            counts['read']+=1
            if reason:
                counts['rejected']+=1
                reasons[reason.split(':')[0]]+=1
                rejects.write(json.dumps(dict(identity,reason=reason),ensure_ascii=False)+'\n')
            else:
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
                counts['written']+=1
            if counts['read']%10000==0: print(dict(counts),flush=True)
    report=dict(counts,rejection_reasons=dict(reasons),arguments=vars(a))
    paths[2].write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not counts['written']: raise RuntimeError('No samples written; inspect rejection report')

if __name__=='__main__':
    main()

SCIENCE_CONVERTER_PY

FILE_COUNT=0
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    RELATIVE_PATH="${INPUT_FILE#"$INPUT_ROOT"/}"
    STEM="${RELATIVE_PATH%.jsonl}"
    CONVERTED_FILE="$CONVERTED_ROOT/$STEM.jsonl"
    OUTPUT_PREFIX="$OUTPUT_ROOT/$STEM"
    LOG_FILE="${OUTPUT_PREFIX}.$(date +%Y%m%d_%H%M%S)_$$.preprocess.log"
    mkdir -p "$(dirname "$CONVERTED_FILE")" "$(dirname "$OUTPUT_PREFIX")"
    CONVERT_ARGS=(--input "$INPUT_FILE" --output "$CONVERTED_FILE"
                  --repo "$REPO" --tokenizer "$TOKENIZER"
                  --seq-length "$SEQ_LENGTH" --workers "$WORKERS")
    if [[ -n "$LIMIT" ]]; then CONVERT_ARGS+=(--limit "$LIMIT"); fi

    echo "Converting: $INPUT_FILE"
    "$PYTHON_BIN" -u "$CONVERTER_FILE" "${CONVERT_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
    echo "Preprocessing: $CONVERTED_FILE"
    "$PYTHON_BIN" -u preprocess_data.py \
        --input "$CONVERTED_FILE" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "$TOKENIZER" \
        --handler-name SharegptStyleInstructionHandler \
        --prompt-type qwen3 \
        --prompt-type-path "$SCIENCE_TEMPLATE_PATH" \
        --enable-thinking true \
        --map-keys '{"messages":"messages","system":"system","tools":"tools","tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant","observation_tag":"observation","function_tag":"function","system_tag":"system"}}' \
        --seq-length "$SEQ_LENGTH" \
        --pack \
        --workers "$WORKERS" \
        --log-interval 10000 \
        --cache-dir "$CACHE_DIR" 2>&1 | tee -a "$LOG_FILE"
    for key in input_ids attention_mask labels; do
        for ext in bin idx; do
            result="${OUTPUT_PREFIX}_packed_${key}_document.${ext}"
            [[ -s "$result" ]] || { echo "Missing/empty output: $result" >&2; exit 1; }
        done
    done
    FILE_COUNT=$((FILE_COUNT + 1))
    echo "Finished $STEM; log: $LOG_FILE"
done
echo "Finished $FILE_COUNT JSONL files. Separate bin/idx groups: $OUTPUT_ROOT"

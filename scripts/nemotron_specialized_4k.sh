python preprocess_data.py \
        --input /dpc-zhouy/zhouy/post-training-data/nemotron_specialized_sharegpt_convert/fact_seeking \
        --output-prefix /dpc-zhouy/zhouy/instructft_data/fact_seeking/nemotron_fact_seeking_1 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path /dpc-zhouy/zhouy/ckpts/Qwen3-8B \
        --handler-name SharegptStyleInstructionHandler \
        --map-keys '{"messages":"conversations","system":"system"}' \
        --prompt-type qwen3 \
        --pack  \
        --seq-length 4096 \
        --workers 32 \
        --log-interval 100000 \
        --cache-dir /dpc-zhouy/cache

python preprocess_data.py \
        --input /dpc-zhouy/zhouy/post-training-data/nemotron_specialized_sharegpt_convert/fact_seeking_2 \
        --output-prefix /dpc-zhouy/zhouy/instructft_data/fact_seeking/nemotron_fact_seeking_2 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path /dpc-zhouy/zhouy/ckpts/Qwen3-8B \
        --handler-name SharegptStyleInstructionHandler \
        --map-keys '{"messages":"conversations","system":"system"}' \
        --prompt-type qwen3 \
        --pack  \
        --seq-length 4096 \
        --workers 32 \
        --log-interval 100000 \
        --cache-dir /dpc-zhouy/cache

python preprocess_data.py \
        --input /dpc-zhouy/zhouy/post-training-data/nemotron_specialized_sharegpt_convert/fact_seeking_3 \
        --output-prefix /dpc-zhouy/zhouy/instructft_data/fact_seeking/nemotron_fact_seeking_3 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path /dpc-zhouy/zhouy/ckpts/Qwen3-8B \
        --handler-name SharegptStyleInstructionHandler \
        --map-keys '{"messages":"conversations","system":"system"}' \
        --prompt-type qwen3 \
        --pack  \
        --seq-length 4096 \
        --workers 32 \
        --log-interval 100000 \
        --cache-dir /dpc-zhouy/cache

# #!/usr/bin/env bash
# set -euo pipefail

# # cd /dpc-zhouy/zhouy/MindSpeed-LLM

# source /dpc-zhouy/zhouy/miniconda3/bin/activate
# conda activate py311
# source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh


# SFT_ROOT="/dpc-zhouy/zhouy/post-training-data/nemotron_specialized_sharegpt_convert"
# BIN_ROOT="/dpc-zhouy/zhouy/instructft_data/nemotron_specialized_4K"
# TOKENIZER="/dpc-zhouy/zhouy/ckpts/Qwen3-8B"
# CACHE_DIR="/dpc-zhouy/cache"

# WORKERS=32
# SEQ_LENGTH=4096
# HANDLER="SharegptStyleInstructionHandler"

# mkdir -p "$BIN_ROOT"

# while IFS= read -r -d '' INPUT_FILE; do
#     RELATIVE="${INPUT_FILE#"$SFT_ROOT"/}"
#     STEM="${RELATIVE%.jsonl}"
#     OUTPUT_PREFIX="${BIN_ROOT}/${STEM}"

#     mkdir -p "$(dirname "$OUTPUT_PREFIX")"

#     echo "Input:  $INPUT_FILE"
#     echo "Output: $OUTPUT_PREFIX"

#     python ../preprocess_data.py \
#         --input "$INPUT_FILE" \
#         --output-prefix "$OUTPUT_PREFIX" \
#         --tokenizer-type PretrainedFromHF \
#         --tokenizer-name-or-path "$TOKENIZER" \
#         --handler-name "$HANDLER" \
#         --prompt-type qwen3 \
#         --seq-length "$SEQ_LENGTH" \
#         --workers "$WORKERS" \
#         --log-interval 100000 \
#         --cache-dir "$CACHE_DIR"

# done < <(
#     find "$SFT_ROOT" \
#         -type f \
#         -name '*.jsonl' \
#         -print0 |
#     sort -z
# )

# echo "Finished: $BIN_ROOT"
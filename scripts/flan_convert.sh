# #!/bin/bash
# set -euo pipefail

# DATA_DIR=/dpc-zhouy/zhouy/post-training-data/Nemotron-SFT-Science-v2
# PART_DIR=/dpc-zhouy/zhouy/instructft_data/nemotron_science_v2
# TOKENIZER=/dpc-zhouy/zhouy/ckpts/Qwen3-8B

# mkdir -p "${PART_DIR}"

# for NAME in rqa so syn_mcq vendor; do
#     echo "Processing ${NAME}.jsonl"

#     python ./preprocess_data.py \
#         --input "${DATA_DIR}/${NAME}.jsonl" \
#         --tokenizer-name-or-path "${TOKENIZER}" \
#         --output-prefix "${PART_DIR}/${NAME}" \
#         --handler-name OpenAIInstructionHandler \
#         --tokenizer-type PretrainedFromHF \
#         --prompt-type qwen3 \
#         --enable-thinking none \
#         --drop-thinking false \
#         --seq-length 8192 \
#         --workers 16 \
#         --log-interval 1000 \
#         --cache-dir /dpc-zhouy/cache \
#         --overwrite-cache
# done

#!/bin/bash

FLAN_ROOT=/dpc-zhouy/zhouy/post-training-data/FLAN
OUTPUT_ROOT=/dpc-zhouy/zhouy/instructft_data/FLAN_4K
TOKENIZER=/dpc-zhouy/zhouy/ckpts/Qwen3-8B
SEQ_LENGTH=4096

mkdir -p "${OUTPUT_ROOT}"

for INPUT_DIR in "${FLAN_ROOT}"/*_data; do
    [ -d "${INPUT_DIR}" ] || continue

    NAME=$(basename "${INPUT_DIR}")

    if [[ "${NAME}" == cot_* ]]; then
        ENABLE_THINKING=true
    else
        ENABLE_THINKING=false
    fi

    python ./preprocess_data.py \
        --input "${INPUT_DIR}" \
        --tokenizer-name-or-path "${TOKENIZER}" \
        --output-prefix "${OUTPUT_ROOT}/${NAME}" \
        --handler-name AlpacaStyleInstructionHandler \
        --map-keys '{"prompt":"inputs","query":null,"response":"targets"}' \
        --tokenizer-type PretrainedFromHF \
        --prompt-type qwen3 \
        --enable-thinking "${ENABLE_THINKING}" \
        --seq-length ${SEQ_LENGTH} \
        --workers 32 \
        --log-interval 10000 \
        --cache-dir /dpc-zhouy/cache 
done
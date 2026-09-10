#!/bin/bash

cd /dpc-zhouy/zhouy/MindSpeed-LLM/
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
MODEL_NAME="SciLLM-base-8B"
# MODEL_NAME="Qwen3-8B-Base"
# MODEL_NAME="SciLLM-Instruct-8B-demo"
TOKENIZER_PATH="/dpc-zhouy/zhouy/ckpts/Qwen3-8B"
CHECKPOINT="/dpc-zhouy/zhouy/ckpts/${MODEL_NAME}"

# DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/mmlu/test"
# TASK="mmlu"
# MAX_NEW_TOKEN=3

# TASK="mmlu_ppl"
# MAX_NEW_TOKEN=1

# DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/agieval"
# TASK="agieval"

# DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/boolq"
# TASK="boolq"
# MAX_NEW_TOKEN=4


# DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/piqa/piqa_validation.parquet"
# TASK="piqa"
# MAX_NEW_TOKEN=4

# DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/siqa/social_iqa_validation.parquet"
# TASK="siqa"
# MAX_NEW_TOKEN=4

TASKS=(
    piqa
    siqa
    hellaswag
    arc-e
    arc-c
    obqa
    winogrande
    boolq
)
DATA_PATHS=(
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/piqa/piqa_validation.parquet"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/siqa/social_iqa_validation.parquet"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/Hellaswag/validation-00000-of-00001.parquet"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/arc_easy/test.jsonl"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/arc_challenge/test.jsonl"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/OBQA/test-00000-of-00001.parquet"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/winogrande_xl/validation-00000-of-00001.parquet"
    "/dpc-zhouy/zhouy/ckpts/evaluate_data/boolq"
)

MAX_NEW_TOKEN=4

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=8
PP=1
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
ITERATION=$(tr -d '[:space:]' < "${CHECKPOINT}/latest_checkpointed_iteration.txt")
LOG_FILE="./log_eval/common_sense_${MODEL_NAME}_iter_${ITERATION}_${TIMESTAMP}.log"


torchrun $DISTRIBUTED_ARGS evaluation.py \
        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --task-data-path "${DATA_PATHS[@]}" \
        --task "${TASKS[@]}" \
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-layers 36 \
        --hidden-size 4096 \
        --ffn-hidden-size 12288 \
        --num-attention-heads 32 \
        --group-query-attention \
        --num-query-groups 8 \
        --seq-length ${SEQ_LENGTH} \
        --max-new-tokens ${MAX_NEW_TOKEN} \
        --max-position-embeddings 32768 \
        --disable-bias-linear \
        --swiglu \
        --norm-epsilon 1e-6 \
        --padded-vocab-size 151936 \
        --make-vocab-size-divisible-by 1 \
        --position-embedding-type rope \
        --load ${CHECKPOINT} \
        --no-chat-template \
        --kv-channels 128 \
        --qk-layernorm \
        --untie-embeddings-and-output-weights \
        --rotary-base 1000000 \
        --use-rotary-position-embeddings \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --normalization RMSNorm \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-gradient-accumulation-fusion \
        --attention-softmax-in-fp32 \
        --tokenizer-not-use-fast \
        --exit-on-missing-checkpoint \
        --no-masked-softmax-fusion \
        --micro-batch-size 1 \
        --no-load-rng \
        --no-load-optim \
        --seed 42 \
        --bf16 \
        --transformer-impl local \
        --ckpt-format torch \
        > ${LOG_FILE} 2>&1 < /dev/null &
#!/bin/bash

cd /dpc-zhouy/zhouy/MindSpeed-LLM/
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
MODEL_NAME="SciLLM-base-8B"
TOKENIZER_PATH="/dpc-zhouy/zhouy/ckpts/Qwen3-8B"
CHECKPOINT="/dpc-zhouy/zhouy/ckpts/${MODEL_NAME}"
DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/mmlu/test"
TASK="mmlu"

# DATA_PATH="/dpc-zhouy/zhouy/ckpts/evaluate_data/agieval"
# TASK="agieval"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6003
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
LOG_FILE="./evaluate_log/${MODEL_NAME}_${TIMESTAMP}.log"


torchrun $DISTRIBUTED_ARGS evaluation.py \
        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --task-data-path ${DATA_PATH} \
        --task ${TASK} \
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
        --max-new-tokens 3 \
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
        | tee ${LOG_FILE}
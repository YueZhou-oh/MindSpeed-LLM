#!/bin/bash
# set -o pipefail

# Number of training nodes. NODE_RANK=0 is the master node.
NNODES=${NNODES:-8}
NODE_RANK=${NODE_RANK:-0}

cd /dpc-zhouy/zhouy/MindSpeed-LLM/
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2
export GLOO_SOCKET_IFNAME=enp66s0f5

NPUS_PER_NODE=8
MASTER_ADDR=10.16.201.163
MASTER_PORT=6008
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
MODEL_NAME="SciLLM-base-8B"
CKPT_SAVE_DIR="/dpc-zhouy/zhouy/ckpts/${MODEL_NAME}"
CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-${CKPT_SAVE_DIR}}"
TOKENIZER_PATH="/dpc-zhouy/zhouy/ckpts/Qwen3-8B"
DATA_PATH="
0.05 /dpc-zhouy/zhouy/pretraining/c4_text_document
0.20 /dpc-zhouy/zhouy/pretraining/wiki_text_document
0.05 /dpc-zhouy/zhouy/pretraining/stackexchange_text_document
0.20 /dpc-zhouy/zhouy/pretraining/arxiv_text_document
0.10 /dpc-zhouy/zhouy/pretraining/cc_2019_30_en_head_text_document
0.20 /dpc-zhouy/zhouy/pretraining/cc_2021_04_en_head_text_document
0.20 /dpc-zhouy/zhouy/pretraining/cc_2023_06_en_head_text_document
"

TP=8
PP=1
CP=1
MBS=4
GBS=2048
SEQ_LENGTH=4096
TRAIN_ITERS=50000
EVAL_INTER=100
SAVE_ITERS=2000


TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
LOG_FILE="./logs_0901/${MODEL_NAME}_nn${NNODES}rank${NODE_RANK}tp${TP}pp${PP}cp${CP}mbs${MBS}gbs${GBS}_${TIMESTAMP}.log"
MONITOR_BACKEND=${MONITOR_BACKEND:-wandb}  # tensorboard, wandb, both, or none
MONITOR_DIR=${MONITOR_DIR:-./monitoring/${MODEL_NAME}_stage2_nn${NNODES}tp${TP}pp${PP}cp${CP}mbs${MBS}gbs${GBS}}
NPU_PEAK_TFLOPS=${NPU_PEAK_TFLOPS:-402.5}  # Override this with the BF16 peak of the actual NPU.
AI_CORE_SAMPLE_INTERVAL=${AI_CORE_SAMPLE_INTERVAL:-0.1}
WANDB_PROJECT=${WANDB_PROJECT:-mindspeed-llm}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-${MODEL_NAME}-stage2-${SEQ_LENGTH}-nn${NNODES}-tp${TP}-pp${PP}-cp${CP}}

MONITOR_ARGS="
    --log-throughput \
    --log-mfu \
    --theoretical-device-tflops ${NPU_PEAK_TFLOPS} \
"

# --log-ai-core-utilization \
# --ai-core-utilization-sampling-interval ${AI_CORE_SAMPLE_INTERVAL}
# --log-params-norm \
# --log-num-zeros-in-grad \


case "${MONITOR_BACKEND}" in
    tensorboard)
        MONITOR_ARGS+=" --tensorboard-dir ${MONITOR_DIR}/tensorboard --tensorboard-log-interval 1 --log-timers-to-tensorboard --log-memory-to-tensorboard"
        ;;
    wandb)
        MONITOR_ARGS+=" --tensorboard-dir ${MONITOR_DIR}/tensorboard --tensorboard-log-interval 1 --log-timers-to-tensorboard --log-memory-to-tensorboard --use-wandb --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${WANDB_EXP_NAME} --wandb-save-dir ${MONITOR_DIR}/wandb"
        ;;
    both)
        MONITOR_ARGS+=" --tensorboard-dir ${MONITOR_DIR}/tensorboard --tensorboard-log-interval 1 --log-timers-to-tensorboard --log-memory-to-tensorboard --use-wandb --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${WANDB_EXP_NAME} --wandb-save-dir ${MONITOR_DIR}/wandb"
        ;;
    none)
        ;;
    *)
        echo "Unsupported MONITOR_BACKEND=${MONITOR_BACKEND}; use tensorboard, wandb, both, or none." >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname "${LOG_FILE}")" "${MONITOR_DIR}"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-ascend-coc
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1e-4 \
    --lr-decay-style cosine \
    --min-lr 1e-5 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH}
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 36 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --norm-epsilon 1e-6 \
    --ckpt-format torch
"

DATA_ARGS="
    --handler-name GeneralPretrainHandler \
    --workers 4 \
    --data-path $DATA_PATH \
    --split 98,1,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_ITERS} \
    --eval-interval ${EVAL_INTER} \
    --eval-iters 10 
"

# --no-load-optim \
# --no-load-rng

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $MONITOR_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    --transformer-impl local \
    > ${LOG_FILE} 2>&1 < /dev/null &

# --enable-hf2mg-convert \
# --model-type-hf qwen3 \

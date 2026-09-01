#!/bin/bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

cd /dpc-zhouy/zhouy/MindSpeed-LLM/
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh

export PRINT_TRAINING_SAMPLE=${PRINT_TRAINING_SAMPLE:-1}
export PRINT_TRAINING_SAMPLE_MAX_TOKENS=${PRINT_TRAINING_SAMPLE_MAX_TOKENS:-8192}
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2
export GLOO_SOCKET_IFNAME=enp66s0f5

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
# CKPT_LOAD_DIR="/dpc-zhouy/zhouy/ckpts/scillm_demo_hf"
CKPT_LOAD_DIR="/dpc-zhouy/zhouy/ckpts/SciLLM-Qwen3-8B"
CKPT_SAVE_DIR="/dpc-zhouy/zhouy/ckpts/scillm_dpo_demo_mc"
DATA_PATH="/dpc-zhouy/zhouy/data/finetune_dataset/orca_rlhf"
TOKENIZER_PATH="/dpc-zhouy/zhouy/ckpts/scillm_demo_hf"

TP=8
PP=1
MBS=2
GBS=16

SEQ_LENGTH=8192
TRAIN_ITERS=4000
EVAL_ITERS=20
SAVE_ITERS=2000

TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
LOG_FILE="./logs_run/dpo_scillm_qwen3_8b_sl${SEQ_LENGTH}_nn${NNODES}rank${NODE_RANK}tp${TP}pp${PP}mbs${MBS}gbs${GBS}_${TIMESTAMP}.log"
MONITOR_BACKEND=${MONITOR_BACKEND:-wandb}  # tensorboard, wandb, both, or none
MONITOR_DIR=${MONITOR_DIR:-./monitoring/scillm_qwen3_8b_sl${SEQ_LENGTH}_nn${NNODES}tp${TP}pp${PP}mbs${MBS}gbs${GBS}}
NPU_PEAK_TFLOPS=${NPU_PEAK_TFLOPS:-402.5}  # Override this with the BF16 peak of the actual NPU.
AI_CORE_SAMPLE_INTERVAL=${AI_CORE_SAMPLE_INTERVAL:-0.1}
WANDB_PROJECT=${WANDB_PROJECT:-mindspeed-llm}
WANDB_EXP_NAME=${WANDB_EXP_NAME:-dpo-scillm-qwen3-8b-sl${SEQ_LENGTH}-nn${NNODES}-tp${TP}-pp${PP}}


MONITOR_ARGS="
    --log-throughput \
    --log-mfu \
    --theoretical-device-tflops ${NPU_PEAK_TFLOPS} \
"

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

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --use-flash-attn \
    --num-layers 36 \
    --hidden-size 4096  \
    --use-rotary-position-embeddings \
    --num-attention-heads 32 \
    --ffn-hidden-size 12288 \
    --max-position-embeddings 32768 \
    --seq-length ${SEQ_LENGTH} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --disable-bias-linear \
    --train-iters ${TRAIN_ITERS} \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --untie-embeddings-and-output-weights \
    --num-query-groups 8 \
    --min-lr 1.25e-7 \
    --lr 1.25e-6 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16 \
    --ckpt-format torch
"

DATA_ARGS="
    --handler-name AlpacaStyleInstructionHandler \
    --is-instruction-dataset \
    --enable-thinking true \
    --prompt-type qwen3 \
    --workers 4 \
    --data-path $DATA_PATH \
    --split 98,1,1 \
    --npu-deterministic \
"

CKPT_ARGS=""

# --enable-hf2mg-convert \
#    --model-type-hf qwen3

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${SAVE_ITERS} \
    --eval-interval ${EVAL_ITERS} \
    --eval-iters 10 \
"

TUNE_ARGS="
    --finetune \
    --stage dpo \
    --dpo-loss-type sigmoid \
    --is-pairwise-dataset \
    --no-pad-to-seq-lengths
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    $CKPT_ARGS \
    $MONITOR_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --transformer-impl local \
    > ${LOG_FILE} 2>&1 < /dev/null &

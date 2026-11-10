#!/bin/bash

cd /dpc-zhouy/zhouy/MindSpeed-LLM/
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
source /dpc-zhouy/usr/local/Ascend/cann/set_env.sh
source /dpc-zhouy/usr/local/Ascend/nnal/atb/set_env.sh
# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=enp66s0f1

MODEL_NAME="SciOWL-Instruct-30B-A3B"
# please fill these path configurations
TOKENIZER_PATH="/dpc-zhouy/zhouy/ckpts/Qwen3-30B-A3B"
# CHECKPOINT="/dpc-zhouy/zhouy/ckpts/SciLLM-base-8B"
CHECKPOINT="/dpc-zhouy/zhouy/ckpts/${MODEL_NAME}"

# Change for multinode config
MASTER_ADDR=10.1.0.239
MASTER_PORT=6007
NNODES=2
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


MAX_NEW_TOKEN=1024
TP=1
PP=1
EP=16
CP=1
SEQ_LENGTH=4096
CP_TYPE='ulysses_cp_algo'
ROUTER_BALANCING_TYPE='aux_loss'
MOE_TOKEN_DISPATCHER_TYPE='alltoall'

TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')
ITERATION=$(tr -d '[:space:]' < "${CHECKPOINT}/latest_checkpointed_iteration.txt")
LOG_FILE="./logs_generate_c3/${MODEL_NAME}_rank${NODE_RANK}_iter_${ITERATION}_${TIMESTAMP}.log"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mkdir -p "$(dirname "${LOG_FILE}")"

torchrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size 1 \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --context-parallel-size ${CP} \
        --context-parallel-algo ${CP_TYPE} \
        --kv-channels 128 \
        --qk-layernorm \
        --norm-topk-prob \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --num-layers 48 \
        --hidden-size 2048 \
        --ffn-hidden-size 6144 \
        --num-attention-heads 32 \
        --tokenizer-type PretrainedFromHF \
        --make-vocab-size-divisible-by 1 \
        --padded-vocab-size 151936 \
        --rotary-base 1000000 \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --position-embedding-type rope \
        --normalization RMSNorm \
        --norm-epsilon 1e-6 \
        --swiglu \
        --attention-softmax-in-fp32 \
        --no-gradient-accumulation-fusion \
        --group-query-attention \
        --num-query-groups 4 \
        --ckpt-format torch \
        --recompute-activation-function \
        --seq-length ${SEQ_LENGTH} \
        --max-new-tokens ${MAX_NEW_TOKEN} \
        --max-position-embeddings 32768 \
        --load ${CHECKPOINT} \
        --no-chat-template \
        --use-fused-rotary-pos-emb \
        --use-rotary-position-embeddings \
        --recompute-method uniform \
        --recompute-granularity full \
        --recompute-num-layers 1 \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
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
        --num-experts 128 \
        --moe-router-topk 8 \
        --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
        --moe-ffn-hidden-size 768 \
        --moe-grouped-gemm \
        --moe-permutation-async-comm \
        --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE} \
        --moe-layer-freq -1 \
        --first-k-dense-replace -1 \
        --moe-aux-loss-coeff 0.001 \
        --moe-zero-memory level1 \
        --moe-fb-overlap \
        --exit-on-missing-checkpoint \
        --transformer-impl local \
        --ckpt-format torch \
        --prompt-type qwen3 \
        --task chat \
       | tee ${LOG_FILE}


# #!/bin/bash

# cd /dpc-zhouy/zhouy/MindSpeed-LLM/
# source /dpc-zhouy/zhouy/miniconda3/bin/activate
# conda activate py311
# source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
# # The number of parameters is not aligned
# export CUDA_DEVICE_MAX_CONNECTIONS=1

# # please fill these path configurations
# TOKENIZER_PATH="/dpc-zhouy/zhouy/forge/baseline/forge_tokenizer"
# CHECKPOINT="/dpc-zhouy/zhouy/ckpts/SciLLM-8B"

# # Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
# NNODES=1
# NODE_RANK=0
# NPUS_PER_NODE=8
# WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# TP=8
# PP=1
# SEQ_LENGTH=4096

# DISTRIBUTED_ARGS="
#     --nproc_per_node $NPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "

# torchrun $DISTRIBUTED_ARGS inference.py \
#        --use-mcore-models \
#        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
#        --qk-layernorm \
#        --tensor-model-parallel-size ${TP} \
#        --pipeline-model-parallel-size ${PP} \
#        --num-layers 36 \
#        --hidden-size 4096  \
#        --num-attention-heads 32  \
#        --ffn-hidden-size 12288 \
#        --max-position-embeddings ${SEQ_LENGTH} \
#        --seq-length ${SEQ_LENGTH} \
#        --disable-bias-linear \
#        --group-query-attention \
#        --num-query-groups 8 \
#        --swiglu \
#        --use-fused-swiglu \
#        --normalization RMSNorm \
#        --norm-epsilon 1e-6 \
#        --use-fused-rmsnorm \
#        --position-embedding-type rope \
#        --rotary-base 1000000 \
#        --use-fused-rotary-pos-emb \
#        --make-vocab-size-divisible-by 1 \
#        --padded-vocab-size 52000 \
#        --micro-batch-size 1 \
#        --max-new-tokens 256 \
#        --tokenizer-type PretrainedFromHF  \
#        --tokenizer-name-or-path ${TOKENIZER_PATH} \
#        --tokenizer-not-use-fast \
#        --hidden-dropout 0 \
#        --attention-dropout 0 \
#        --untie-embeddings-and-output-weights \
#        --no-gradient-accumulation-fusion \
#        --attention-softmax-in-fp32 \
#        --seed 42 \
#        --load ${CHECKPOINT} \
#        --exit-on-missing-checkpoint \
#        --transformer-impl local \
#        --ckpt-format torch \
#        --task chat \
#        | tee logs/generate_mcore_qwen3_all_8b_chat-test.log
#!/usr/bin/env bash
set -euo pipefail
# Run on EVERY participating node with its own NODE_RANK, same other variables.
export MINDSPEED_REPO="/dpc-zhouy/zhouy/MindSpeed-LLM"

# Parent directory containing iter_0002000, iter_0062000, etc.
export CHECKPOINT_ROOT="/dpc-zhouy/zhouy/ckpts/SciLLM-base-30B-A3B"
# export STEP=2000

export TOKENIZER_PATH="/dpc-zhouy/zhouy/ckpts/Qwen3-30B-A3B"

# Tokenized JSONL produced by prepare_inputs.py
export ROUTE_INPUT="/dpc-zhouy/zhouy/MindSpeed-LLM/moe_routing_analysis/four_domain_620_tokenized.jsonl"

# Choose a new output directory for each run
export ROUTE_OUTPUT="/dpc-zhouy/zhouy/MindSpeed-LLM/moe_routing_analysis/routing_results_new/step_${STEP}"

# Actual IP address of the first participating node
export MASTER_ADDR="10.16.201.31"

export GLOO_SOCKET_IFNAME=enp66s0f5

# cd /dpc-zhouy/zhouy/mindspeed_routing_analysis
# NODE_RANK=0 bash run_routing.sh

mkdir -p "${ROUTE_OUTPUT}"

NNODES=${NNODES:-2}
# NODE_RANK=${NODE_RANK:-0}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-6017}
EP=${EP:-16}
ROUTE_LENGTH=${ROUTE_LENGTH:-50}

if (( NNODES * NPUS_PER_NODE != EP )); then
    echo "This runner requires NNODES * NPUS_PER_NODE == EP (${EP})." >&2; exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo 'Invalid NODE_RANK' >&2; exit 2
fi
BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MINDSPEED_REPO
# export PYTHONPATH="${BUNDLE_DIR}:${MINDSPEED_REPO}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
# Activate your existing training environment and source CANN before this script.
cd "${MINDSPEED_REPO}"
exec torchrun \
  --nnodes "${NNODES}" --nproc_per_node "${NPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
  "${BUNDLE_DIR}/analyze_routing.py" \
  --route-input "${ROUTE_INPUT}" --route-output "${ROUTE_OUTPUT}" \
  --load "${CHECKPOINT_ROOT}" --ckpt-step "${STEP}" \
  --no-load-optim --no-load-rng --ckpt-format torch \
  --use-mcore-models \
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
  --transformer-impl local --distributed-backend nccl \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 --expert-tensor-parallel-size 1 \
  --expert-model-parallel-size "${EP}" \
  --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 \
  --num-attention-heads 32 --num-query-groups 4 --kv-channels 128 \
  --group-query-attention --qk-layernorm \
  --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 \
  --moe-grouped-gemm --moe-token-dispatcher-type alltoall \
  --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.001 \
  --moe-layer-freq -1 --first-k-dense-replace -1 --norm-topk-prob \
  --moe-zero-memory disable \
  --tokenizer-type PretrainedFromHF --tokenizer-name-or-path "${TOKENIZER_PATH}" \
  --padded-vocab-size 151936 --make-vocab-size-divisible-by 1 \
  --seq-length "${ROUTE_LENGTH}" --max-position-embeddings 4096 \
  --position-embedding-type rope --use-rotary-position-embeddings --rotary-base 1000000 \
  --normalization RMSNorm --norm-epsilon 1e-6 --swiglu \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --attention-dropout 0.0 --hidden-dropout 0.0 --attention-softmax-in-fp32 \
  --use-flash-attn --use-fused-rmsnorm --use-fused-swiglu --use-fused-rotary-pos-emb \
  --no-masked-softmax-fusion --no-gradient-accumulation-fusion \
  --bf16 --micro-batch-size 1 --global-batch-size "${EP}" --train-iters 1 \
  --seed 42 "$@"

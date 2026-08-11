# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
  --load-model-type hf \
  --save-model-type mg \
  --model-type-hf deepseek4 \
  --load-dir ../ckpts/DeepSeek-V4-Flash-Base-bf16 \
  --save-dir ../ckpts/DeepSeek-V4-Flash-Base-bf16-mcore \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 4 \
  --target-expert-parallel-size 32 \
  --noop-layers 43 \
  --mtp-num-layers 1 \
  --moe-grouped-gemm
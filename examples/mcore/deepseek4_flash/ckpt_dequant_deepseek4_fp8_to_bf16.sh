# pip install torchao

python tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py \
  --input_fp8_hf_path ../ckpts/DeepSeek-V4-Flash-Base \
  --output_hf_path ../ckpts/DeepSeek-V4-Flash-Base-bf16 \
  --quant_type bfloat16

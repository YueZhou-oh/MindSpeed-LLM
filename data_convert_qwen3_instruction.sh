source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir /dpc-zhouy/zhouy/data/finetune_dataset

# python ./preprocess_data.py \
#     --input ../data/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
#     --tokenizer-name-or-path /dpc-zhouy/zhouy/ckpts/scillm_demo_hf/ \
#     --output-prefix /dpc-zhouy/zhouy/data/finetune_dataset/alpaca \
#     --handler-name AlpacaStyleInstructionHandler \
#     --tokenizer-type PretrainedFromHF \
#     --workers 4 \
#     --log-interval 1000 \
#     --enable-thinking true \
#     --prompt-type qwen3

# 若使用Alpaca多轮对话数据集需要增加以下参数
# -map-keys '{"prompt":"instruction","query":"input","response":"output", "history":"history"}'
# 多轮对话建议使用Sharegpt格式数据集

# Sharegpt多轮对话数据集示例

# python ./preprocess_data.py \
# 	--input /dpc-zhouy/zhouy/post-training-data/sharegpt-pack/hermes_de_sharegpt.json \
# 	--tokenizer-name-or-path /dpc-zhouy/zhouy/ckpts/scillm_demo_hf/ \
# 	--output-prefix /dpc-zhouy/zhouy/data/finetune_dataset/sharegpt_pack \
# 	--handler-name SharegptStyleInstructionHandler \
# 	--tokenizer-type PretrainedFromHF \
# 	--workers 4 \
# 	--seq-length 8192 \
# 	--log-interval 1000 \
# 	--prompt-type qwen3 \
#     --enable-thinking true \
# 	--pack \
# 	--neat-pack \
# 	--map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}'

python ./preprocess_data.py \
        --input /dpc-zhouy/zhouy/post-training-data/orca_rlhf.jsonl \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-not-use-fast \
        --tokenizer-name-or-path /dpc-zhouy/zhouy/ckpts/scillm_demo_hf/ \
        --output-prefix /dpc-zhouy/zhouy/data/finetune_dataset/orca_rlhf \
        --workers 4 \
        --log-interval 1000 \
        --handler-name AlpacaStylePairwiseHandler \
        --prompt-type qwen3 \
        --seq-length 8192 \
        --map-keys '{"prompt":"question", "query":"", "system":"system"}'
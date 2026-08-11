# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ../data/alpaca_dsv4

python ./preprocess_data.py \
    --input ../data/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ../ckpts/DeepSeek-V4-Flash-Base-bf16 \
    --output-prefix ../data/alpaca_dsv4 \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type deepseek4 \
    --enable-thinking true
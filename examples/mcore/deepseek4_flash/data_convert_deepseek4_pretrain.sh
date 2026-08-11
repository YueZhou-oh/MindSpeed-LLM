# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ../data/alpaca_dsv4

python ./preprocess_data.py \
    --input ../data/enwiki-2026-05-01-text/enwiki-2026-05-01-p1134786p3947636.parquet \
    --tokenizer-name-or-path ../ckpts/DeepSeek-V4-Flash-Base-bf16  \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ../data/enwiki_260501_dsv4 \
    --json-keys wikitext \
    --workers 128 \
    --log-interval 1000
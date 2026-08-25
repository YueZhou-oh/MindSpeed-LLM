source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
cd /dpc-zhouy/zhouy/redpajama-data-v2
ansible-playbook \
    -i ./ansible_cmd/hosts.ini \
    ./ansible_cmd/dispatch_redpajama_download.yml \
    --forks 8 \
    -e shared_downloader=/dpc-zhouy/zhouy/redpajama-data-v2/download_redpajama_v2_en_head_middle_8way.sh
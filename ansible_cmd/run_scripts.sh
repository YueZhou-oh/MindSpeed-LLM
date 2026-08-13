source /dpc-zhouy/zhouy/miniconda3/bin/activate 
conda activate py311
ansible all -m shell -a "free -h" -i ./ansible_cmd/hosts.ini

ansible all -m shell -a "ps -ef|grep pretrain" -i ./ansible_cmd/hosts.ini

ansible all -m shell -a "pgrep -f pretrain_gpt | xargs -r kill -9" -i ./ansible_cmd/hosts.ini

ansible all -m shell -a "ps -ef|grep python" -i ./ansible_cmd/hosts.ini

ansible all -m command -a "who" -i ./ansible_cmd/hosts.ini

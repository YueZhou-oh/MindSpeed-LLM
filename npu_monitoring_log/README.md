python npu_monitor.py --once
python npu_monitor.py > /dpc-zhouy/zhouy/npu_monitoring_log/logs/$(hostname -I | awk '{print $1}')_tmp.log 2>&1 < /dev/null &

ansible all -m shell -a "date" -i ./hosts.ini
ansible all -i ./hosts.ini -m shell -a '
cd /dpc-zhouy/zhouy/npu_monitoring_log || exit 1
mkdir -p logs
IP=$(hostname -I)
IP=${IP%% *}

nohup /dpc-zhouy/zhouy/miniconda3/envs/py311/bin/python3 \
    npu_monitor.py \
    --output-root /dpc-zhouy/zhouy/npu_monitoring_log/logs \
    --ip "$IP" \
    --interval 60 \
    > "/dpc-zhouy/zhouy/npu_monitoring_log/logs/${IP}_tmp.log" \
    2>&1 < /dev/null &
'

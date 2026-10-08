#!/usr/bin/env bash
# nohup ./hourly_scp.sh >/dev/null 2>&1 &

DEST="/dpc-zhouy/zhouy/MindSpeed-LLM"
REMOTE_USER="root"
REMOTE_HOST="10.42.206.110"
REMOTE_PATH="/dpc-zhouy/zhouy/MindSpeed-LLM/monitoring_c3"
INTERVAL=600
LOG_FILE="./hourly_scp.log"

while true; do
    echo "[$(date '+%F %T')] Starting SCP transfer" >> "$LOG_FILE"

    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" "$DEST" >> "$LOG_FILE" 2>&1

    sleep 60

    wandb sync monitoring_c3/offline_SciLLM-base-30B-A3B_stage1_nn14tp1pp2ep8cp1mbs1gbs1792/wandb/wandb/latest-run >> "$LOG_FILE" 2>&1
    echo "[$(date '+%F %T')] Finish wandb sync" >> "$LOG_FILE"
    sleep "$INTERVAL"
done
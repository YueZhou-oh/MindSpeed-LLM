#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MASTER_SCRIPT=${1:-${MASTER_SCRIPT:-${SCRIPT_DIR}/scratch_forge_core_8b_4K_main.sh}}
WORKER_IP_FILE=${2:-${WORKER_IP_FILE:-${SCRIPT_DIR}/worker_ips.txt}}
OUTPUT_DIR=${3:-${OUTPUT_DIR:-${SCRIPT_DIR}/worker_rank_scripts}}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f ${MASTER_SCRIPT} ]] || die "Master script not found: ${MASTER_SCRIPT}"
[[ -f ${WORKER_IP_FILE} ]] || die "Worker IP file not found: ${WORKER_IP_FILE}"

mapfile -t WORKER_IPS < <(
    sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e '/^$/d' "${WORKER_IP_FILE}"
)

(("${#WORKER_IPS[@]}" > 0)) || die "No worker IP was found in ${WORKER_IP_FILE}"

declare -A SEEN_IPS=()
for ip in "${WORKER_IPS[@]}"; do
    [[ ${ip} =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die "Invalid IPv4 address: ${ip}"
    IFS=. read -r o1 o2 o3 o4 <<< "${ip}"
    for octet in "${o1}" "${o2}" "${o3}" "${o4}"; do
        ((10#${octet} <= 255)) || die "Invalid IPv4 address: ${ip}"
    done
    [[ -z ${SEEN_IPS[${ip}]+x} ]] || die "Duplicate worker IP: ${ip}"
    SEEN_IPS["${ip}"]=1
done

NNODES=$(("${#WORKER_IPS[@]}" + 1))
mkdir -p "${OUTPUT_DIR}"

for index in "${!WORKER_IPS[@]}"; do
    rank=$((index + 1))
    worker_script="${OUTPUT_DIR}/worker_rank${rank}.sh"
    tmp_script="${worker_script}.tmp"

    awk -v nnodes="${NNODES}" -v rank="${rank}" '
        BEGIN { nnodes_done = 0; rank_done = 0 }
        !nnodes_done && /^NNODES=/ {
            print "NNODES=" nnodes
            nnodes_done = 1
            next
        }
        !rank_done && /^NODE_RANK=/ {
            print "NODE_RANK=" rank
            rank_done = 1
            next
        }
        { print }
        END {
            if (!nnodes_done || !rank_done) {
                exit 42
            }
        }
    ' "${MASTER_SCRIPT}" > "${tmp_script}" || {
        status=$?
        rm -f "${tmp_script}"
        if ((status == 42)); then
            die "The master script must contain NNODES=... and NODE_RANK=... lines."
        fi
        exit "${status}"
    }

    chmod 0755 "${tmp_script}"
    mv -f "${tmp_script}" "${worker_script}"
    printf 'rank=%d ip=%s script=%s\n' "${rank}" "${WORKER_IPS[index]}" "${worker_script}"
done

echo "Generated ${#WORKER_IPS[@]} worker scripts; NNODES=${NNODES}."

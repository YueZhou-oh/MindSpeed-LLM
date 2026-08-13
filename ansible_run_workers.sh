#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKER_IP_FILE=${WORKER_IP_FILE:-${SCRIPT_DIR}/worker_ips.txt}
MASTER_SCRIPT=${MASTER_SCRIPT:-${SCRIPT_DIR}/scratch_forge_core_8b_4K_main.sh}
GENERATOR_SCRIPT=${GENERATOR_SCRIPT:-${SCRIPT_DIR}/generate_worker_scripts.sh}
GENERATED_SCRIPT_DIR=${GENERATED_SCRIPT_DIR:-${SCRIPT_DIR}/worker_rank_scripts}

SSH_USER=${SSH_USER:-root}
SSH_KEY_FILE=${SSH_KEY_FILE:-}
REMOTE_WORKDIR=${REMOTE_WORKDIR:-/dpc-zhouy/zhouy/MindSpeed-LLM}
# Shared filesystem: workers run the generated scripts in place (no copy).
REMOTE_SCRIPT_DIR=${REMOTE_SCRIPT_DIR:-${GENERATED_SCRIPT_DIR}}
ANSIBLE_FORKS=${ANSIBLE_FORKS:-20}
SKIP_GENERATE=${SKIP_GENERATE:-0}

TIMESTAMP=$(date '+%Y-%m-%d-%H-%M-%S')

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -f ${WORKER_IP_FILE} ]] || die "Worker IP file not found: ${WORKER_IP_FILE}"
[[ -x ${GENERATOR_SCRIPT} ]] || die "Generator is missing or not executable: ${GENERATOR_SCRIPT}"
[[ ${REMOTE_WORKDIR} != *[[:space:]]* ]] || die "REMOTE_WORKDIR cannot contain whitespace."
[[ ${REMOTE_SCRIPT_DIR} != *[[:space:]]* ]] || die "REMOTE_SCRIPT_DIR cannot contain whitespace."

mapfile -t WORKER_IPS < <(
    sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
        -e '/^$/d' "${WORKER_IP_FILE}"
)
(("${#WORKER_IPS[@]}" > 0)) || die "No worker IP was found in ${WORKER_IP_FILE}"
NNODES=$(("${#WORKER_IPS[@]}" + 1))

if [[ ${SKIP_GENERATE} != 1 ]]; then
    "${GENERATOR_SCRIPT}" "${MASTER_SCRIPT}" "${WORKER_IP_FILE}" "${GENERATED_SCRIPT_DIR}"
fi

for index in "${!WORKER_IPS[@]}"; do
    rank=$((index + 1))
    [[ -f ${GENERATED_SCRIPT_DIR}/worker_rank${rank}.sh ]] || \
        die "Generated worker script not found: ${GENERATED_SCRIPT_DIR}/worker_rank${rank}.sh"
done

RUNTIME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/forge-ansible.XXXXXX")
trap 'rm -rf -- "${RUNTIME_DIR}"' EXIT
INVENTORY_FILE=${RUNTIME_DIR}/inventory.ini
PLAYBOOK_FILE=${RUNTIME_DIR}/run_workers.yml
ANSIBLE_CONFIG_FILE=${RUNTIME_DIR}/ansible.cfg

{
    echo "[workers]"
    for index in "${!WORKER_IPS[@]}"; do
        rank=$((index + 1))
        printf 'worker%d ansible_host=%s node_rank=%d remote_rank_script=%s/worker_rank%d.sh remote_log=%s/logs_launch/%s/launcher_rank%d.log\n' \
            "${rank}" "${WORKER_IPS[index]}" "${rank}" \
            "${REMOTE_SCRIPT_DIR}" "${rank}" \
            "${REMOTE_WORKDIR}" "${TIMESTAMP}" "${rank}"
    done
    echo
    echo "[workers:vars]"
    printf 'ansible_user=%s\n' "${SSH_USER}"
    printf 'nnodes=%d\n' "${NNODES}"
    printf 'remote_workdir=%s\n' "${REMOTE_WORKDIR}"
    printf 'timestamp=%s\n' "${TIMESTAMP}"
    echo "ansible_python_interpreter=/dpc-zhouy/zhouy/miniconda3/envs/py311/bin/python3"
    if [[ -n ${SSH_KEY_FILE} ]]; then
        printf 'ansible_ssh_private_key_file=%s\n' "${SSH_KEY_FILE}"
    fi
} > "${INVENTORY_FILE}"

{
    echo "[defaults]"
    printf 'inventory = %s\n' "${INVENTORY_FILE}"
    printf 'forks = %s\n' "${ANSIBLE_FORKS}"
    echo "host_key_checking = False"
    echo "retry_files_enabled = False"
    echo "interpreter_python = auto_silent"
    echo
    echo "[ssh_connection]"
    echo "pipelining = True"
    echo "ssh_args = -o BatchMode=yes -o ConnectTimeout=100 -o ServerAliveInterval=150 -o ServerAliveCountMax=20"
} > "${ANSIBLE_CONFIG_FILE}"

cat > "${PLAYBOOK_FILE}" <<'YAML'
---
- name: Execute all worker scripts asynchronously
  hosts: workers
  gather_facts: false
  strategy: free
  tasks:
    - name: Ensure launcher log directory exists
      ansible.builtin.file:
        path: "{{ remote_workdir }}/logs_launch/{{ timestamp }}"
        state: directory
        mode: "0755"

    - name: Submit worker training without waiting for completion
      ansible.builtin.shell:
        cmd: >-
          nohup env NNODES={{ nnodes }} NODE_RANK={{ node_rank }}
          bash {{ remote_rank_script | quote }}
          > {{ remote_log | quote }} 2>&1 < /dev/null &
        chdir: "{{ remote_workdir }}"
      async: 60
      poll: 0
      changed_when: true
YAML

echo "Checking SSH and Python connectivity to all workers ..."
ANSIBLE_CONFIG=${ANSIBLE_CONFIG_FILE} ansible workers -m ansible.builtin.ping -o

echo "Submitting worker commands on shared scripts ..."
ANSIBLE_CONFIG=${ANSIBLE_CONFIG_FILE} ansible-playbook "${PLAYBOOK_FILE}"

echo "Submitted ${#WORKER_IPS[@]} workers asynchronously; NNODES=${NNODES}."
echo "Remote logs: ${REMOTE_WORKDIR}/logs_launch/${TIMESTAMP}/launcher_rank<N>.log"

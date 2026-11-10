#!/usr/bin/env bash
set -euo pipefail

# Download only CC-MAIN-2024* Parquet files from:
# openbmb/Ultra-FineWeb/data/ultrafineweb_en_v1_4
#
# Dependency:
#   pip install -U huggingface_hub hf_xet
#
# Usage:
#   bash download_ultrafineweb_en_v1_4_2024_range.sh START_INDEX END_INDEX
#
# Four-node split:
#   node 1: bash download_ultrafineweb_en_v1_4_2024_range.sh 1 1280
#   node 2: bash download_ultrafineweb_en_v1_4_2024_range.sh 1281 2560
#   node 3: bash download_ultrafineweb_en_v1_4_2024_range.sh 2561 3840
#   node 4: bash download_ultrafineweb_en_v1_4_2024_range.sh 3841 5120

if [[ $# -ne 2 ]]; then
	    echo "Usage: $0 START_INDEX END_INDEX" >&2
	        exit 2
fi

START_INDEX_RAW="$1"
END_INDEX_RAW="$2"
TOTAL_FILES=5120

if [[ ! "$START_INDEX_RAW" =~ ^[0-9]+$ || ! "$END_INDEX_RAW" =~ ^[0-9]+$ ]]; then
	    echo "ERROR: START_INDEX and END_INDEX must be integers." >&2
	        exit 2
fi

START_INDEX=$((10#$START_INDEX_RAW))
END_INDEX=$((10#$END_INDEX_RAW))

if (( START_INDEX < 1 || END_INDEX > TOTAL_FILES || START_INDEX > END_INDEX )); then
	    echo "ERROR: require 1 <= START_INDEX <= END_INDEX <= $TOTAL_FILES" >&2
	        exit 2
fi

DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-/home/pcl_user/Ultra-FineWeb}"
LOG_ROOT="${LOG_ROOT:-${DOWNLOAD_ROOT}/download_logs/en_v1_4_2024}"
PYTHON_BIN="${PYTHON_BIN:-/home/pcl_user/miniconda3/bin/python}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
MAX_RETRIES="${MAX_RETRIES:-4}"

if [[ ! -x "$PYTHON_BIN" ]]; then
	    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
	        exit 1
fi

if [[ ! "$DOWNLOAD_WORKERS" =~ ^[1-9][0-9]*$ || ! "$MAX_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
	    echo "ERROR: DOWNLOAD_WORKERS and MAX_RETRIES must be positive integers." >&2
	        exit 2
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

mkdir -p "$DOWNLOAD_ROOT" "$LOG_ROOT"

NODE_NAME="$(hostname -s)"
START_TIME="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_ROOT}/${NODE_NAME}_${START_INDEX}_${END_INDEX}_${START_TIME}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Repository: openbmb/Ultra-FineWeb"
echo "Restricted prefix: data/ultrafineweb_en_v1_4/CC-MAIN-2024"
echo "Node: $NODE_NAME"
echo "Global range: $START_INDEX-$END_INDEX (inclusive)"
echo "Files: $((END_INDEX - START_INDEX + 1))"
echo "Download root: $DOWNLOAD_ROOT"
echo "Workers: $DOWNLOAD_WORKERS"
echo "Log: $LOG_FILE"

"$PYTHON_BIN" - \
	    "$START_INDEX" \
	        "$END_INDEX" \
		    "$DOWNLOAD_ROOT" \
		        "$DOWNLOAD_WORKERS" \
			    "$MAX_RETRIES" <<'PY'
import concurrent.futures
import os
import random
import sys
import threading
import time
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is required. Install it with: "
        "pip install -U huggingface_hub hf_xet"
    ) from exc


start_index = int(sys.argv[1])
end_index = int(sys.argv[2])
download_root = Path(sys.argv[3]).resolve()
workers = int(sys.argv[4])
max_retries = int(sys.argv[5])

repo_id = "openbmb/Ultra-FineWeb"
repo_type = "dataset"
token = os.environ.get("HF_TOKEN") or None
print_lock = threading.Lock()

snapshots = (
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-18",
    "CC-MAIN-2024-22",
    "CC-MAIN-2024-26",
    "CC-MAIN-2024-30",
    "CC-MAIN-2024-33",
    "CC-MAIN-2024-38",
    "CC-MAIN-2024-42",
    "CC-MAIN-2024-46",
    "CC-MAIN-2024-51",
)
files_per_snapshot = 512


def remote_filename(global_index):
    zero_based = global_index - 1
    snapshot = snapshots[zero_based // files_per_snapshot]
    local_part = zero_based % files_per_snapshot + 1
    basename = (
        f"ultrafineweb-en-{snapshot}-"
        f"part-{local_part:04d}-of-{files_per_snapshot}.parquet"
    )
    return f"data/ultrafineweb_en_v1_4/{snapshot}/{basename}"


def download_one(global_index):
    filename = remote_filename(global_index)
    expected_path = download_root / filename

    if expected_path.is_file() and expected_path.stat().st_size > 0:
        with print_lock:
            print(
                f"SKIP global_index={global_index:04d} path={expected_path}",
                flush=True,
            )
        return global_index, str(expected_path), None

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with print_lock:
                print(
                    f"GET global_index={global_index:04d} "
                    f"attempt={attempt}/{max_retries} file={filename}",
                    flush=True,
                )

            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=str(download_root),
                token=token,
                force_download=False,
            )

            path = Path(local_path)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"download returned an empty or missing file: {path}")

            with print_lock:
                print(
                    f"DONE global_index={global_index:04d} "
                    f"size={path.stat().st_size} path={path}",
                    flush=True,
                )
            return global_index, str(path), None

        except Exception as exc:
            last_error = exc
            with print_lock:
                print(
                    f"RETRY global_index={global_index:04d} "
                    f"attempt={attempt}/{max_retries} "
                    f"error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

            if attempt < max_retries:
                delay = min(60.0, 5.0 * (2 ** (attempt - 1)))
                time.sleep(delay + random.uniform(0.0, 2.0))

    error = f"{type(last_error).__name__}: {last_error}"
    return global_index, None, error


indices = range(start_index, end_index + 1)
total = end_index - start_index + 1
completed = 0
failures = []

print(f"First remote file: {remote_filename(start_index)}", flush=True)
print(f"Last remote file:  {remote_filename(end_index)}", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(download_one, index): index
        for index in indices
    }

    for future in concurrent.futures.as_completed(futures):
        index = futures[future]
        try:
            _, local_path, error = future.result()
        except Exception as exc:
            local_path = None
            error = f"{type(exc).__name__}: {exc}"

        if error:
            failures.append((index, error))
            print(
                f"FAILED global_index={index:04d} error={error}",
                file=sys.stderr,
                flush=True,
            )
        else:
            completed += 1

        print(
            f"PROGRESS completed={completed} "
            f"failed={len(failures)} total={total}",
            flush=True,
        )

print(f"SUMMARY completed={completed} failed={len(failures)} total={total}")

if failures:
    print(
        "FAILED_GLOBAL_INDICES="
        + ",".join(f"{index:04d}" for index, _ in failures),
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

echo "Download range completed successfully: $START_INDEX-$END_INDEX"


#!/usr/bin/env bash

# Download RedPajama-Data-V2 English head+middle data, split into 8
# independent workers by Common Crawl snapshot.
#
# Components downloaded for every base tag:
#   1. documents
#   2. quality_signals
#   3. minhash
#   4. duplicates
#
# Usage:
#   bash download_redpajama_v2_en_head_middle_8way.sh WORKER_ID OUTPUT_ROOT [PARALLEL_DOWNLOADS]
#
# Examples:
#   bash download_redpajama_v2_en_head_middle_8way.sh 0 /data/redpajama_v2 16
#   bash download_redpajama_v2_en_head_middle_8way.sh 7 /data/redpajama_v2 16
#
# WORKER_ID must be 0 through 7. All workers may use the same OUTPUT_ROOT;
# their snapshot sets do not overlap.

set -uo pipefail

readonly BASE_URL="https://data.together.xyz/redpajama-data-v2/v1.0.0"
readonly LANG="en"
readonly PARTITION="head_middle"

WORKER_ID="${1:-}"
OUTPUT_ROOT="${2:-}"
PARALLEL_DOWNLOADS="${3:-${PARALLEL_DOWNLOADS:-8}}"

usage() {
    printf 'Usage: %s WORKER_ID OUTPUT_ROOT [PARALLEL_DOWNLOADS]\n' "$0" >&2
    printf '  WORKER_ID: 0-7\n' >&2
    printf '  Example: %s 0 /data/redpajama_v2 16\n' "$0" >&2
}

if [[ ! "$WORKER_ID" =~ ^[0-7]$ ]]; then
    usage
    exit 2
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    usage
    exit 2
fi

if [[ ! "$PARALLEL_DOWNLOADS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'PARALLEL_DOWNLOADS must be a positive integer: %s\n' "$PARALLEL_DOWNLOADS" >&2
    exit 2
fi

if ! command -v wget >/dev/null 2>&1; then
    printf 'Error: wget is required but was not found.\n' >&2
    exit 127
fi

if ! command -v xargs >/dev/null 2>&1; then
    printf 'Error: xargs is required but was not found.\n' >&2
    exit 127
fi

# 84 snapshots divided into four groups of 11 and four groups of 10.
SNAPSHOTS_0=(
    2014-15 2014-23 2014-35 2014-41 2014-42 2014-49 2014-52
    2015-14 2015-22 2015-27 2015-32
)

SNAPSHOTS_1=(
    2015-35 2015-40 2015-48
    2016-07 2016-18 2016-22 2016-26 2016-30 2016-36 2016-40 2016-44
)

SNAPSHOTS_2=(
    2016-50
    2017-04 2017-09 2017-17 2017-22 2017-26 2017-30 2017-34 2017-39 2017-43 2017-47
)

SNAPSHOTS_3=(
    2017-51
    2018-05 2018-09 2018-13 2018-17 2018-22 2018-26 2018-30 2018-34 2018-39 2018-43
)

SNAPSHOTS_4=(
    2018-47 2018-51
    2019-04 2019-09 2019-13 2019-18 2019-22 2019-26 2019-30 2019-35
)

SNAPSHOTS_5=(
    2019-39 2019-43 2019-47 2019-51
    2020-05 2020-10 2020-16 2020-24 2020-29 2020-34
)

SNAPSHOTS_6=(
    2020-40 2020-45 2020-50
    2021-04 2021-10 2021-17 2021-21 2021-25 2021-31 2021-39
)

SNAPSHOTS_7=(
    2021-43 2021-49
    2022-05 2022-21 2022-27 2022-33 2022-40 2022-49
    2023-06 2023-14
)

declare -n SNAPSHOTS="SNAPSHOTS_${WORKER_ID}"

readonly LISTINGS_DIR="${OUTPUT_ROOT}/listings"
readonly LOG_DIR="${OUTPUT_ROOT}/logs"
readonly FAILED_LOG="${LOG_DIR}/worker_${WORKER_ID}_failed.tsv"
readonly RUN_LOG="${LOG_DIR}/worker_${WORKER_ID}.log"

mkdir -p "$LISTINGS_DIR" "$LOG_DIR"
# Failures are per invocation. Existing .part files preserve resumable data.
: >"$FAILED_LOG"
touch "$RUN_LOG"

log() {
    local message
    message="$(date '+%Y-%m-%d %H:%M:%S') [worker ${WORKER_ID}] $*"
    printf '%s\n' "$message" | tee -a "$RUN_LOG"
}

download_listing() {
    local snapshot="$1"
    local tag="${LANG}-${snapshot}-${PARTITION}"
    local destination="${LISTINGS_DIR}/${tag}.txt"
    local partial="${destination}.part"
    local url="${BASE_URL}/listings/${tag}.txt"

    if [[ -s "$destination" ]]; then
        printf '%s\n' "$destination"
        return 0
    fi

    if wget \
        --continue \
        --tries=20 \
        --timeout=60 \
        --waitretry=5 \
        --no-verbose \
        --output-document="$partial" \
        "$url"; then
        mv "$partial" "$destination"
        printf '%s\n' "$destination"
        return 0
    fi

    printf '%s\t%s\t%s\n' "listing" "$snapshot" "$url" >>"$FAILED_LOG"
    return 1
}

download_one() {
    local component="$1"
    local base_tag="$2"
    local suffix
    local url
    local destination
    local partial

    case "$component" in
        documents)
            suffix=".json.gz"
            ;;
        quality_signals)
            suffix=".signals.json.gz"
            ;;
        minhash|duplicates)
            suffix=".${component}.parquet"
            ;;
        *)
            printf 'Unknown component: %s\n' "$component" >&2
            return 0
            ;;
    esac

    url="${BASE_URL}/${component}/${base_tag}${suffix}"
    destination="${OUTPUT_ROOT}/${component}/${base_tag}${suffix}"
    partial="${destination}.part"

    # A non-empty final file is treated as complete. Interrupted downloads use
    # the .part file and are resumed by wget --continue.
    if [[ -s "$destination" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "$destination")"

    if wget \
        --continue \
        --tries=20 \
        --timeout=60 \
        --waitretry=5 \
        --no-verbose \
        --output-document="$partial" \
        "$url"; then
        mv "$partial" "$destination"
    else
        # Keep the partial file for a future retry. Record failures without
        # stopping the remaining multi-million-file download.
        printf '%s\t%s\t%s\n' "$component" "$base_tag" "$url" >>"$FAILED_LOG"
    fi

    return 0
}

export BASE_URL OUTPUT_ROOT FAILED_LOG
export -f download_one

log "Starting ${#SNAPSHOTS[@]} snapshots with ${PARALLEL_DOWNLOADS} parallel downloads"
log "Snapshots: ${SNAPSHOTS[*]}"

for snapshot in "${SNAPSHOTS[@]}"; do
    log "Fetching listing for ${snapshot}"

    if ! listings_file="$(download_listing "$snapshot")"; then
        log "Listing failed for ${snapshot}; continuing with the next snapshot"
        continue
    fi

    base_tag_count="$(awk 'NF {count++} END {print count+0}' "$listings_file")"
    log "Downloading ${snapshot}: ${base_tag_count} base tags x 4 components"

    # The official head_middle listing contains both en_head and en_middle base
    # tags. Each pair below becomes one independent xargs job.
    awk 'NF {print $0}' "$listings_file" |
        while IFS= read -r base_tag; do
            printf '%s\t%s\n' documents "$base_tag"
            printf '%s\t%s\n' quality_signals "$base_tag"
            printf '%s\t%s\n' minhash "$base_tag"
            printf '%s\t%s\n' duplicates "$base_tag"
        done |
        xargs -r -P "$PARALLEL_DOWNLOADS" -n 2 \
            bash -c 'download_one "$1" "$2"' _

    log "Finished download pass for ${snapshot}"
done

failure_count="$(awk 'NF {count++} END {print count+0}' "$FAILED_LOG")"

if (( failure_count > 0 )); then
    log "Completed with ${failure_count} recorded failures; rerun the same worker to retry"
    log "Failure list: ${FAILED_LOG}"
    exit 1
fi

log "Completed successfully"

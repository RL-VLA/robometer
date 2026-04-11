#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

#==
# Helpers
#==

display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

version_gte() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

check_versions() {
    if ! command -v docker &> /dev/null; then
        echo "[ERROR] Docker is not installed." >&2; exit 1
    fi
    if ! command -v apptainer &> /dev/null; then
        echo "[ERROR] Apptainer is not installed." >&2; exit 1
    fi
    docker_version=$(docker --version | awk '{ print $3 }' | tr -d ',')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')
    if version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO] Docker ${docker_version} and Apptainer ${apptainer_version} — OK."
    else
        display_warning "Docker ${docker_version} / Apptainer ${apptainer_version} are untested versions."
    fi
}

check_sif_exists() {
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/robometer.tar ]"; then
        echo "[ERROR] robometer.tar not found on $CLUSTER_LOGIN:$CLUSTER_SIF_PATH" >&2
        echo "[ERROR] Push it first: bash cluster/cluster_interface.sh push" >&2
        exit 1
    fi
}

#==
# Usage
#==

help() {
    echo ""
    echo "usage: $(basename "$0") [-h] <command> [<args>...]"
    echo ""
    echo "Commands:"
    echo "  build               Build Docker image and convert to Apptainer sandbox."
    echo "  push                Scp the Apptainer tar to the cluster."
    echo "  job [args...]       Rsync code to cluster and submit a SLURM job."
    echo "                      train-datasets=<key>  override train dataset key"
    echo "                      eval-datasets=<key>   override eval dataset key"
    echo "                      exp-name=<name>       override experiment name"
    echo "                      Any other args are forwarded as Hydra overrides to train.py."
    echo "                      Examples:"
    echo "                        bash cluster/cluster_interface.sh job"
    echo "                        bash cluster/cluster_interface.sh job train-datasets=robometer_dataset_cloth_tshirt_cloth training.max_steps=500"
    echo "  data                Upload processed_datasets/ and Robometer-4B/ to the cluster (run once before first job)."
    echo "  log                 Rsync output from the latest run dir on the cluster."
    echo ""
    echo "Prerequisites:"
    echo "  cp cluster/.env.cluster.example cluster/.env.cluster  # then fill in your details"
    echo ""
}

#==
# Arg parsing
#==

while getopts ":h" opt; do
    case ${opt} in
        h) help; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; help; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -lt 1 ]; then
    echo "[ERROR] A command is required." >&2; help; exit 1
fi

command=$1; shift

#==
# Commands
#==

case $command in

    build)
        check_versions

        echo "[INFO] Building Docker image robometer:latest ..."
        cd "$REPO_ROOT"
        docker build -t robometer:latest -f cluster/Dockerfile .
        echo "[INFO] Docker build complete."

        exports_dir="$SCRIPT_DIR/exports"
        mkdir -p "$exports_dir"
        rm -rf "$exports_dir/robometer.sif" "$exports_dir/robometer.tar"

        echo "[INFO] Converting to Apptainer sandbox ..."
        cd "$exports_dir"
        APPTAINER_NOHTTPS=1 apptainer build --sandbox robometer.sif docker-daemon://robometer:latest

        echo "[INFO] Creating tar archive ..."
        tar -cvf robometer.tar robometer.sif
        echo "[INFO] Build complete: $exports_dir/robometer.tar"
        ;;

    push)
        source "$SCRIPT_DIR/.env.cluster"

        exports_dir="$SCRIPT_DIR/exports"
        if [ ! -f "$exports_dir/robometer.tar" ]; then
            echo "[ERROR] $exports_dir/robometer.tar not found." >&2
            echo "[ERROR] Build first: bash cluster/cluster_interface.sh build" >&2
            exit 1
        fi

        echo "[INFO] Ensuring remote SIF directory exists ..."
        ssh "$CLUSTER_LOGIN" "mkdir -p $CLUSTER_SIF_PATH"

        echo "[INFO] Copying image to cluster ..."
        scp "$exports_dir/robometer.tar" "$CLUSTER_LOGIN:$CLUSTER_SIF_PATH/robometer.tar"
        echo "[INFO] Push complete."
        ;;

    job)
        source "$SCRIPT_DIR/.env.cluster"
        check_sif_exists

        # Parse known keys; everything else is a Hydra override
        JOB_TRAIN_DATASETS=""
        JOB_EVAL_DATASETS=""
        JOB_EXP_NAME=""
        HYDRA_OVERRIDES=()
        for arg in "$@"; do
            case "$arg" in
                train-datasets=*) JOB_TRAIN_DATASETS="${arg#train-datasets=}" ;;
                eval-datasets=*)  JOB_EVAL_DATASETS="${arg#eval-datasets=}" ;;
                exp-name=*)       JOB_EXP_NAME="${arg#exp-name=}" ;;
                *)                HYDRA_OVERRIDES+=("$arg") ;;
            esac
        done

        timestamp=$(date +"%Y%m%d_%H%M%S")
        CLUSTER_RUN_DIR="${CLUSTER_REPO_DIR}_${timestamp}"

        echo "[INFO] Creating remote run directory: $CLUSTER_RUN_DIR"
        ssh "$CLUSTER_LOGIN" "mkdir -p $CLUSTER_RUN_DIR"

        echo "[INFO] Syncing code to cluster ..."
        rsync -rh \
            --exclude=".git" \
            --exclude="output" \
            --exclude="logs" \
            --exclude="cluster/exports" \
            --exclude="robometer_dataset" \
            --exclude="processed_datasets" \
            --exclude="/datasets" \
            --exclude="wandb" \
            --exclude="*.hdf5" \
            --exclude="*.pkl" \
            --exclude="*.sif" \
            "$REPO_ROOT/" "$CLUSTER_LOGIN:$CLUSTER_RUN_DIR/"

        echo "[INFO] Submitting job — train=$JOB_TRAIN_DATASETS eval=$JOB_EVAL_DATASETS exp=$JOB_EXP_NAME overrides=${HYDRA_OVERRIDES[*]}"
        ssh "$CLUSTER_LOGIN" \
            "BATCH_TRAIN_DATASETS='$JOB_TRAIN_DATASETS' \
             BATCH_EVAL_DATASETS='$JOB_EVAL_DATASETS' \
             BATCH_EXP_NAME='$JOB_EXP_NAME' \
             bash -l '$CLUSTER_RUN_DIR/cluster/submit_job_slurm.sh' '$CLUSTER_RUN_DIR' ${HYDRA_OVERRIDES[*]}"

        echo "[INFO] Job submitted. Run dir: $CLUSTER_LOGIN:$CLUSTER_RUN_DIR"
        ;;

    data)
        source "$SCRIPT_DIR/.env.cluster"

        echo "[INFO] Syncing processed datasets to cluster ..."
        ssh "$CLUSTER_LOGIN" "mkdir -p $CLUSTER_PROCESSED_DATASETS_PATH"
        rsync -rh --progress \
            "$REPO_ROOT/processed_datasets/" \
            "$CLUSTER_LOGIN:$CLUSTER_PROCESSED_DATASETS_PATH/"

        echo "[INFO] Data upload complete."
        ;;

    log)
        source "$SCRIPT_DIR/.env.cluster"
        if [ -z "${CLUSTER_LOG_DIR:-}" ]; then
            CLUSTER_LOG_DIR=$(ssh "$CLUSTER_LOGIN" "ls -td ${CLUSTER_REPO_DIR}_* 2>/dev/null | head -1")
            if [ -z "$CLUSTER_LOG_DIR" ]; then
                echo "[ERROR] No run directories found matching ${CLUSTER_REPO_DIR}_*" >&2; exit 1
            fi
            CLUSTER_LOG_DIR="$CLUSTER_LOG_DIR/output"
        fi
        echo "[INFO] Syncing output from $CLUSTER_LOGIN:$CLUSTER_LOG_DIR ..."
        mkdir -p "$REPO_ROOT/output"
        rsync -rh "$CLUSTER_LOGIN:$CLUSTER_LOG_DIR/" "$REPO_ROOT/output/"
        echo "[INFO] Sync complete → $REPO_ROOT/output/"
        ;;

    *)
        echo "[ERROR] Unknown command: $command" >&2; help; exit 1
        ;;
esac

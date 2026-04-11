#!/usr/bin/env bash
# Runs on the cluster login node (called via ssh from cluster_interface.sh).
# Usage: bash submit_job_slurm.sh <CLUSTER_RUN_DIR> [hydra_overrides...]
#
# $1 = CLUSTER_RUN_DIR (absolute path on cluster)
# $2..N = Hydra overrides forwarded to train.py

CLUSTER_RUN_DIR="$1"

source "$CLUSTER_RUN_DIR/cluster/.env.cluster"

# Allow per-job overrides set by cluster_interface.sh
TRAIN_DATASETS="${BATCH_TRAIN_DATASETS:-$CLUSTER_TRAIN_DATASETS}"
EVAL_DATASETS="${BATCH_EVAL_DATASETS:-$CLUSTER_EVAL_DATASETS}"
EXP_NAME="${BATCH_EXP_NAME:-$CLUSTER_EXP_NAME}"

JOBFILE=$(mktemp "$CLUSTER_RUN_DIR/job_XXXXXX.sh")
cat <<EOT > "$JOBFILE"
#!/bin/bash

#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=${CLUSTER_RUN_DIR}/%x_%j.log
#SBATCH -N1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${CLUSTER_EMAIL}
#SBATCH --job-name="rbm-$(date +"%m%d_%H%M")"

BATCH_TRAIN_DATASETS="${TRAIN_DATASETS}" \
BATCH_EVAL_DATASETS="${EVAL_DATASETS}" \
BATCH_EXP_NAME="${EXP_NAME}" \
bash "${CLUSTER_RUN_DIR}/cluster/run_apptainer.sh" "${CLUSTER_RUN_DIR}" $(printf '%q ' "${@:2}")
EOT

echo "[INFO] Generated $JOBFILE:"
cat "$JOBFILE"

DEPENDENCY_FLAG=""
if [ -n "${BATCH_DEPENDENCY:-}" ]; then
    DEPENDENCY_FLAG="--dependency=${BATCH_DEPENDENCY}"
fi

SBATCH_OUT=$(sbatch $DEPENDENCY_FLAG "$JOBFILE")
echo "$SBATCH_OUT"
rm "$JOBFILE"

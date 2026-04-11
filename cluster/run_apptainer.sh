#!/usr/bin/env bash
# Runs on the compute node inside the SLURM job.
# Usage: bash run_apptainer.sh <CLUSTER_RUN_DIR> [hydra_overrides...]
#
# $1 = CLUSTER_RUN_DIR (absolute path, persistent storage)
# $2..N = extra Hydra overrides

CLUSTER_RUN_DIR="$1"

echo "[INFO] run_apptainer.sh: CLUSTER_RUN_DIR=$CLUSTER_RUN_DIR, args=${*:2}"

source "$CLUSTER_RUN_DIR/cluster/.env.cluster"

TRAIN_DATASETS="${BATCH_TRAIN_DATASETS:-$CLUSTER_TRAIN_DATASETS}"
EVAL_DATASETS="${BATCH_EVAL_DATASETS:-$CLUSTER_EVAL_DATASETS}"
EXP_NAME="${BATCH_EXP_NAME:-$CLUSTER_EXP_NAME}"

# Copy code to fast local TMPDIR on compute node
echo "[INFO] Copying code to TMPDIR ..."
cp -r "$CLUSTER_RUN_DIR" "$TMPDIR/robometer"

# Extract Apptainer image
echo "[INFO] Extracting Apptainer image ..."
tar -xf "$CLUSTER_SIF_PATH/robometer.tar" -C "$TMPDIR"

# Persistent output directory
mkdir -p "$CLUSTER_RUN_DIR/output"

WANDB_KEY="${WANDB_KEY:-$(cat ~/.wandb_api_key 2>/dev/null || echo '')}"

echo "[INFO] Launching Apptainer container ..."
apptainer exec \
    --nv \
    -B "$TMPDIR/robometer":/workspace:rw \
    -B "$CLUSTER_RUN_DIR/output":/workspace/logs:rw \
    -B "$CLUSTER_PROCESSED_DATASETS_PATH":/processed_datasets:ro \
    -B "$CLUSTER_MODEL_PATH":/model:ro \
    "$TMPDIR/robometer.sif" \
    bash -c "
        export PYTHONPATH=/workspace:\$PYTHONPATH
        export VIRTUAL_ENV=/opt/venv
        export PATH=/opt/venv/bin:\$PATH
        export ROBOMETER_PROCESSED_DATASETS_PATH=/processed_datasets
        export WANDB_API_KEY='$WANDB_KEY'
        export WANDB_ENTITY=alfred11x-georgia-institute-of-technology
        export WANDB_PROJECT=robometer
        export HF_HOME=/workspace/logs/.hf_cache
        export TRANSFORMERS_CACHE=/workspace/logs/.hf_cache
        cd /workspace
        python train.py \
            model.base_model_id=/model \
            model.use_peft=true \
            model.train_progress_head=true \
            model.train_preference_head=true \
            data.train_datasets=[$TRAIN_DATASETS] \
            data.eval_datasets=[$EVAL_DATASETS] \
            training.load_from_checkpoint=/model \
            model.trust_remote_code=true \
            training.output_dir=/workspace/logs \
            training.exp_name=$EXP_NAME \
            training.per_device_train_batch_size=2 \
            training.gradient_accumulation_steps=4 \
            training.learning_rate=2e-5 \
            training.warmup_ratio=0.1 \
            training.weight_decay=0.01 \
            training.max_steps=1000 \
            training.eval_steps=50 \
            training.custom_eval_steps=50 \
            custom_eval.eval_types=[reward_alignment] \
            custom_eval.reward_alignment=[$EVAL_DATASETS] \
            logging.wandb_entity=alfred11x-georgia-institute-of-technology \
            logging.wandb_project=robometer \
            \"\$@\"
    " -- "${@:2}"

echo "[INFO] Job complete."

if ${REMOVE_CODE_COPY_AFTER_JOB:-false}; then
    echo "[INFO] Removing code copy: $TMPDIR/robometer"
    rm -rf "$TMPDIR/robometer"
fi

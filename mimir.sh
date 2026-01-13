#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

MODELS=(
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-2.8b"
    "EleutherAI/pythia-6.9b"
    "EleutherAI/pythia-12b"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_MODELS=${#MODELS[@]}
MODEL_IDX=$TASK_ID
MODEL=${MODELS[$MODEL_IDX]}

echo "SLURM_ARRAY_TASK_ID = ${TASK_ID}"
echo "Running model = ${MODEL}"

python mimir.py \
    --model "${MODEL}" \
    --split "ngram_13_0.8"
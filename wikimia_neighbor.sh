#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

MODELS=(
    "state-spaces/mamba-1.4b-hf"
    "EleutherAI/pythia-6.9b"
    "EleutherAI/pythia-12b"
    "huggyllama/llama-13b"
    "huggyllama/llama-65b"
)

DATASETS=(
    "WikiMIA_length32"
    "WikiMIA_length64"
    "WikiMIA_length128"
    "WikiMIA_length32_paraphrased"
    "WikiMIA_length64_paraphrased"
    "WikiMIA_length128_paraphrased"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}

MODEL_IDX=$(( TASK_ID / NUM_DATASETS ))
DATASET_IDX=$(( TASK_ID % NUM_DATASETS ))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

INT8_FLAG=""
if [[ "${MODEL}" == *"llama-65b"* ]]; then
    INT8_FLAG="--int8"
fi

echo "SLURM_ARRAY_TASK_ID = ${TASK_ID}"
echo "Running model = ${MODEL}"
echo "Running dataset = ${DATASET}"
echo "int8            = ${INT8_FLAG:-off}"

python wikimia_neighbor.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    ${INT8_FLAG}
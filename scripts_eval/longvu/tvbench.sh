#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate longvu
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

WEIGHTS=$1
PATH_TYPE=${2:-"base"} # mention "lora" if loading LORA weights or leave as it is
OUTDIR=${3:-"outputs/eval"}

evaldb="tvbench"
base_model_name="longvu_qwen_7b"
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LongVU_Qwen2_7B"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}

if [[ "$PATH_TYPE" != "lora" ]]; then
  torchrun --nnodes "$SLURM_JOB_NUM_NODES" --nproc-per-node="$SLURM_GPUS_ON_NODE" -m eval.tvbench.tvbench_longvu.py \
    --model_name "cambrian_qwen" \
    --version "qwen" \
    --model_path ${WEIGHTS} \
    --sample_fps 1 \
    --output_dir ${OUTDIR}

else
  torchrun --nnodes "$SLURM_JOB_NUM_NODES" --nproc-per-node="$SLURM_GPUS_ON_NODE" -m eval.tvbench.tvbench_longvu.py \
    --model_name "cambrian_qwen" \
    --version "qwen" \
    --model_path ${BASE_WEIGHTS} \
    --model_path2 ${WEIGHTS} \
    --sample_fps 1 \
    --output_dir ${OUTDIR}

fi

#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate mllm2
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

WEIGHTS=$1
PATH_TYPE=${2:-"base"} # mention "lora" if loading LORA weights or leave as it is
OUTDIR=${3:-"outputs/eval"}
GPUID=${4:-0}
MAX_FRAMES=${5:-64}

evaldb="mvbench"
base_model_name="llavavideo_qwen_7b"
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LLaVA-Video-7B-Qwen2"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}

if [[ "$PATH_TYPE" != "lora" ]]; then
  CUDA_VISIBLE_DEVICES=$GPUID python -m eval.mvbench.mvbench \
    --base_model_name ${base_model_name} \
    --model-path ${WEIGHTS} \
    --output-dir ${OUTDIR} \
    --max_frames $MAX_FRAMES \

else
  CUDA_VISIBLE_DEVICES=$GPUID python -m eval.mvbench.mvbench \
    --base_model_name ${base_model_name} \
    --model-path ${BASE_WEIGHTS} \
    --model-path2 ${WEIGHTS} \
    --output-dir ${OUTDIR} \
    --max_frames $MAX_FRAMES \

fi

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

evaldb="mlvu"
base_model_name="llavavideo_qwen_7b"
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LLaVA-Video-7B-Qwen2"
DATA_ROOT="/datasets/video_llm/video_eval/MLVU"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}

if [[ "$PATH_TYPE" != "lora" ]]; then
    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.mlvu.choice_bench \
    --base_model_name ${base_model_name} \
    --model-path ${WEIGHTS} \
    --output_dir ${OUTDIR} \
    --data_root ${DATA_ROOT}

else
    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.mlvu.choice_bench \
    --base_model_name ${base_model_name} \
    --model-path ${BASE_WEIGHTS} \
    --model-path2 ${WEIGHTS} \
    --output_dir ${OUTDIR} \
    --data_root ${DATA_ROOT}

fi

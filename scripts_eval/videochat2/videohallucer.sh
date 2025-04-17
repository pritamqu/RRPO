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

eval_db="videohallucer"
base_model_name="videochat2_mistral_7b"
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/VideoChat2_stage3_Mistral_7B"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}

if [[ "$PATH_TYPE" != "lora" ]]; then

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.videohallucer.evaluate_videochat2 \
    --base_model_name ${base_model_name} \
    --model-path ${WEIGHTS} \
    --eval_obj \
    --eval_obj_rel \
    --eval_temporal \
    --eval_semantic \
    --eval_fact \
    --eval_nonfact \
    --output_dir_path ${OUTDIR}

else 

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.videohallucer.evaluate_videochat2 \
    --base_model_name ${base_model_name} \
    --model-path ${BASE_WEIGHTS} \
    --model-path2 ${WEIGHTS} \
    --eval_obj \
    --eval_obj_rel \
    --eval_temporal \
    --eval_semantic \
    --eval_fact \
    --eval_nonfact \
    --output_dir_path ${OUTDIR}

fi

python eval/videohallucer/evaluation_bias.py \
    --base_model_name ${base_model_name} \
    --result_dir ${OUTDIR}
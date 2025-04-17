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

evaldb='tempcompass'
base_model_name="videochat2_mistral_7b"
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/VideoChat2_stage3_Mistral_7B"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}

if [[ "$PATH_TYPE" != "lora" ]]; then
  
    for task_type in "multi-choice" "captioning" "caption_matching" "yes_no"; do
        CUDA_VISIBLE_DEVICES=$GPUID python -m eval.tempcompass.infer \
        --base_model_name ${base_model_name} \
        --model-path ${WEIGHTS} \
        --output_path ${OUTDIR} \
        --num-chunks 1 \
        --chunk-idx 0 \
        --task_type ${task_type}
    done

else

    for task_type in "multi-choice" "captioning" "caption_matching" "yes_no"; do
        CUDA_VISIBLE_DEVICES=$GPUID python -m eval.tempcompass.infer \
        --base_model_name ${base_model_name} \
        --model-path ${BASE_WEIGHTS} \
        --model-path2 ${WEIGHTS} \
        --output_path ${OUTDIR} \
        --num-chunks 1 \
        --chunk-idx 0 \
        --task_type ${task_type}
    done
fi

# evaluation
EVALDIR=${OUTDIR}/auto_eval_results
mkdir -p ${EVALDIR}
export $(grep -v '^#' .env | xargs)
GPT_MODEL="gpt-4o-mini-2024-07-18"

python -m eval.tempcompass.eval_multi-choice \
    --gpt_model ${GPT_MODEL} \
    --api_key ${OPENAI_API_KEY} \
    --input_dir ${OUTDIR} \
    --output_dir ${EVALDIR}

python -m eval.tempcompass.eval_captioning \
    --gpt_model ${GPT_MODEL} \
    --api_key ${OPENAI_API_KEY} \
    --input_dir ${OUTDIR} \
    --output_dir ${EVALDIR}

python -m eval.tempcompass.eval_caption_matching \
    --gpt_model ${GPT_MODEL} \
    --api_key ${OPENAI_API_KEY} \
    --input_dir ${OUTDIR} \
    --output_dir ${EVALDIR}

python -m eval.tempcompass.eval_yes_no \
    --gpt_model ${GPT_MODEL} \
    --api_key ${OPENAI_API_KEY} \
    --input_dir ${OUTDIR} \
    --output_dir ${EVALDIR}
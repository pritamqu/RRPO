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

base_model_name="videochat2_mistral_7b"
eval_db="vidhalluc"
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/VideoChat2_stage3_Mistral_7B"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}

if [[ "$PATH_TYPE" != "lora" ]]; then

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.vidhalluc.video_hallu_bqa \
        --base_model_name ${base_model_name} --model-path ${WEIGHTS} \
        --save_path ${OUTDIR}/bqa_preds.json

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.vidhalluc.video_hallu_mcq \
        --base_model_name ${base_model_name} --model-path ${WEIGHTS} \
        --save_path ${OUTDIR}/mcq_preds.json

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.vidhalluc.video_hallu_sth \
        --base_model_name ${base_model_name} --model-path ${WEIGHTS} \
        --save_path ${OUTDIR}/sth_preds.json

    else 

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.vidhalluc.video_hallu_bqa \
        --base_model_name ${base_model_name} --model-path ${BASE_WEIGHTS} \
        --model-path2 ${WEIGHTS} \
        --save_path ${OUTDIR}/bqa_preds.json

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.vidhalluc.video_hallu_mcq \
        --base_model_name ${base_model_name} --model-path ${BASE_WEIGHTS} \
        --model-path2 ${WEIGHTS} \
        --save_path ${OUTDIR}/mcq_preds.json

    CUDA_VISIBLE_DEVICES=$GPUID python -m eval.vidhalluc.video_hallu_sth \
        --base_model_name ${base_model_name} --model-path ${BASE_WEIGHTS} \
        --model-path2 ${WEIGHTS} \
        --save_path ${OUTDIR}/sth_preds.json

    fi


bqa_anno='/datasets/video_llm/video_eval/VidHalluc/ach_binaryqa.json'
mcq_anno='/datasets/video_llm/video_eval/VidHalluc/ach_mcq.json'
sth_anno='/datasets/video_llm/video_eval/VidHalluc/sth.json'

conda activate mllm2
python eval/vidhalluc/eval_bqa.py ${bqa_anno} ${OUTDIR}/bqa_preds.json
python eval/vidhalluc/eval_mcq.py ${OUTDIR}/mcq_preds.json
python eval/vidhalluc/eval_sth.py ${sth_anno} ${OUTDIR}/sth_preds.json

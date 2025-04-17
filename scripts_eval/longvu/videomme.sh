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

evaldb="videomme"
base_model_name="longvu_qwen_7b"
LORA_WEIGHTS_PATH="/h/pritam/pritam_ssd004/OUTPUTS/Video-LLM/Phase2/ViSA/longvu_qwen2_7B/"${LORA_WEIGHTS}
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LongVU_Qwen2_7B"
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 
OUTDIR=${OUTDIR}/${base_model_name}-${evaldb}/${JOBID}
mkdir -p ${OUTDIR}
AFILE_W_SUB=videomme_with_sub.json
AFILE_WO_SUB=videomme_without_sub.json

mkdir -p ${OUTDIR}
LOGFILE=${OUTDIR}/${evaldb}.log

if [[ "$PATH_TYPE" != "lora" ]]; then
   
    torchrun --nnodes "$SLURM_JOB_NUM_NODES" --nproc-per-node="$SLURM_GPUS_ON_NODE" -m eval.videomme.infer_longvu \
    --base_model_name ${base_model_name} \
    --model_path ${WEIGHTS} \
    --output-file ${OUTDIR}/${AFILE_W_SUB} \
    --output-file2 ${OUTDIR}/${AFILE_WO_SUB} \
    --use_subtitle \
    --max_frames $MAX_FRAMES

else   
    torchrun --nnodes "$SLURM_JOB_NUM_NODES" --nproc-per-node="$SLURM_GPUS_ON_NODE" -m eval.videomme.infer_longvu \
    --base_model_name ${base_model_name} \
    --model_path ${BASE_WEIGHTS} \
    --model_path2 ${WEIGHTS} \
    --output-file ${OUTDIR}/${AFILE_W_SUB} \
    --output-file2 ${OUTDIR}/${AFILE_WO_SUB} \
    --use_subtitle \
    --max_frames $MAX_FRAMES

fi


python -m eval.videomme.calculate \
    --results_file ${OUTDIR}/${AFILE_W_SUB} \
    --results_path ${OUTDIR}/score_with_subs.json

python -m eval.videomme.calculate \
    --response_path ${OUTDIR}/${AFILE_WO_SUB} \
    --results_path ${OUTDIR}/score_without_subs.json

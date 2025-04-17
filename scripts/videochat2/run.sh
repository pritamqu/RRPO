#!/bin/bash


# ========== Setup ==========
export HOME=/scratch/ssd004/scratch/pritam/ 
# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate videochat2
export MASTER_ADDR=$(hostname)
MPORT=$(shuf -i 6000-9999 -n 1)
NNODE=$SLURM_JOB_NUM_NODES
NUM_GPUS=$(nvidia-smi -L | wc -l)

export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
# wandb online
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 

# ========== Default ==========
LR=2e-5
LOSS_ALPHA=0.01
LOSS_BETA=0.9
CONFIG='config.py'
BATCH_SIZE=2
GRAD_ACC_STEPS=4

OUTPUT_DIR="./outputs/videochat2/"${JOBID}
data_path="./HF_DATA/internal/videochat2.json"
mkdir -p ${OUTPUT_DIR}

torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MPORT} \
    train_videochat2.py \
    $(dirname $0)/${CONFIG} \
    output_dir ${OUTPUT_DIR} \
    loss_alpha $LOSS_ALPHA \
    loss_beta $LOSS_BETA \
    batch_size $BATCH_SIZE \
    gradient_accumulation_steps $GRAD_ACC_STEPS \
    optimizer.lr $LR \
    scheduler.epochs 1 \
    scheduler.warmup_epochs 0.0 \
    scheduler.sched 'cosine' \
    data_path ${data_path} \
    model.lora_r 128 \
    model.lora_alpha 256 \
    model.lora_dropout 0.05

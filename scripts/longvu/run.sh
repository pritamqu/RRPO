#!/bin/bash


# ========== Setup ==========
export HOME=/scratch/ssd004/scratch/pritam/ 
# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate longvu


export MASTER_ADDR=$(hostname)
MPORT=$(shuf -i 6000-9999 -n 1)
# export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
# wandb online
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 

# ========== Default ==========
LOSS_ALPHA=0.05
LOSS_BETA=0.5
LR=5e-6
EPOCHS=1
DATA_FILE=$5
LR_SCH=cosine
MAX_FRAMES=16 # you can also set to 32 or more based on memory
LORA=True
LORA_R=128
LORA_ALPHA=256
BATCH_SIZE=1
GRAD_ACC_STEPS=8 # effective batch size 32 across all nodes

OUTPUT_DIR="./outputs/longvu/"${JOBID}
data_path="./HF_DATA/internal/longvu.json"
mkdir -p ${OUTPUT_DIR}
PREV_STAGE_CHECKPOINT="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LongVU_Qwen2_7B"
VERSION="qwen"
BASE_MODEL="cambrian_qwen"

torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --node_rank=$SLURM_PROCID \
    --rdzv_endpoint="$MASTER_ADDR:$MPORT" \
    train_longvu.py \
    --base_model_name $BASE_MODEL \
    --input_model_filename $PREV_STAGE_CHECKPOINT \
    --model_max_length 8192 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --version $VERSION \
    --mm_vision_select_layer "-2" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter False \
    --freeze_backbone False \
    --gradient_checkpointing True \
    --mm_projector_type sva \
    --image_token_len 144 \
    --query_num_list "[144]" \
    --lowres_token 8 \
    --video_fps 1 \
    --highres True \
    --drop_threshold 0.8 \
    --output_dir ${OUTPUT_DIR} \
    --output_model_filename ${OUTPUT_DIR} \
    --data_path ${data_path} \
    --image_folder "" \
    --logging_dir ${OUTPUT_DIR}/log \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --save_steps 50000 \
    --eval_steps 50000 \
    --logging_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --report_to "wandb" \
    --run_name ${JOBID} \
    --save_total_limit 1 \
    --resume True \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type ${LR_SCH} \
    --tf32 False \
    --group_by_modality_length False \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --lora_enable ${LORA} \
    --lora_bias 'lora_only' \
    --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --max_frames $MAX_FRAMES \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --loss_alpha $LOSS_ALPHA \
    --loss_beta $LOSS_BETA


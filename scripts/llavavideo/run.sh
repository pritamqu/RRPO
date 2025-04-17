#!/bin/bash


# ========== Setup ==========
export HOME=/scratch/ssd004/scratch/pritam/ 
# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate llava
module load cuda-12.1
export MASTER_ADDR=$(hostname)
MPORT=$(shuf -i 6000-9999 -n 1)
# export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
# wandb online
export TZ="America/New_York"
JOBID=$(date +"%Y%m%d%H%M%S%4N")
# JOBID=$SLURM_JOB_ID 

# ========== Default ==========
LOSS_ALPHA=0.01
LOSS_BETA=0.1
LR=5e-6
EPOCHS=1
LR_SCH=cosine
MAX_FRAMES=16 # you can also set to 32 or more based on memory
LORA=True
LORA_R=128
LORA_ALPHA=256
BATCH_SIZE=1
GRAD_ACC_STEPS=8 # effective batch size 32 across all nodes

OUTPUT_DIR="./outputs/llavavideo/"${JOBID}
data_path="./HF_DATA/internal/llavavideo.json"
mkdir -p ${OUTPUT_DIR}
PREV_STAGE_CHECKPOINT="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LLaVA-Video-7B-Qwen2"
PROMPT_VERSION="qwen_1_5"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --node_rank=$SLURM_PROCID \
    --rdzv_endpoint="$MASTER_ADDR:$MPORT" \
    train_llavavideo.py \
    --deepspeed $(dirname $0)/zero3.json \
    --model_name_or_path ${PREV_STAGE_CHECKPOINT} \
    --version $PROMPT_VERSION \
    --data_path ${data_path} \
    --image_folder '' \
    --video_folder '' \
    --mm_tunable_parts "mm_language_model_lora" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${JOBID} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type ${LR_SCH} \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 6 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound $MAX_FRAMES \
    --mm_newline_position grid \
    --add_time_instruction True \
    --mm_spatial_pool_stride 2 \
    --force_sample True \
    --lora_enable ${LORA} \
    --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --loss_alpha $LOSS_ALPHA \
    --loss_beta $LOSS_BETA

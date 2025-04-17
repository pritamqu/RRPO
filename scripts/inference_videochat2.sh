#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate videochat2
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/VideoChat2_stage3_Mistral_7B"
WEIGHTS_ROOT="/datasets/video_llm/RRPO_MODEL_WEIGHTS/LORA_WEIGHTS"

python inference.py \
    --base_model_name "videochat2_mistral_7b" \
    --model-path ${BASE_WEIGHTS} \
    --model-path2 ${WEIGHTS_ROOT}"/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" \
    --video_path "sample_video.mp4" \
    --question "Describe this video." \
    --model_max_length 1024

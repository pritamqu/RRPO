#!/bin/bash

# to change the .cache location
export HOME=/scratch/ssd004/scratch/pritam/ 

# load module
source /h/pritam/anaconda3/etc/profile.d/conda.sh
conda activate llava
BASE_WEIGHTS="/h/pritam/pritam_ssd004/.cache/huggingface/hub/LLaVA-Video-7B-Qwen2"
WEIGHTS_ROOT="/datasets/video_llm/RRPO_MODEL_WEIGHTS/LORA_WEIGHTS"

python inference.py \
    --base_model_name "llavavideo_qwen_7b" \
    --model-path ${BASE_WEIGHTS} \
    --model-path2 ${WEIGHTS_ROOT}"/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" \
    --video_path "sample_video.mp4" \
    --question "Describe this video." \
    --model_max_length 1024

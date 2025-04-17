

## Evaluation

**VideoChat2**

```
bash scripts_eval/videochat2/tvbench.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/tempcompas.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/videohallucer.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/vidhalluc.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/mvbench.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/videomme.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/mlvu.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
bash scripts_eval/videochat2/longvideobench.sh "path/VideoChat2_stage3_Mistral_7B-RRPO-16f-LORA" "lora" "outputs/videochat2-rrpo-16f"
```


**LLaVA-Video**

*16f variant*
```
bash scripts_eval/llavavideo/tvbench.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/tempcompas.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/videohallucer.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/vidhalluc.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/mvbench.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/videomme.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/mlvu.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
bash scripts_eval/llavavideo/longvideobench.sh "path/LLaVA-Video-7B-Qwen2-RRPO-16f-LORA" "lora" "outputs/llavavideo-rrpo-16f"
```

*32f variant*
```
bash scripts_eval/llavavideo/tvbench.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/tempcompas.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/videohallucer.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/vidhalluc.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/mvbench.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/videomme.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/mlvu.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
bash scripts_eval/llavavideo/longvideobench.sh "path/LLaVA-Video-7B-Qwen2-RRPO-32f-LORA" "lora" "outputs/llavavideo-rrpo-32f"
```

**LongVU**

```
bash scripts_eval/longvu/tvbench.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/tempcompas.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/videohallucer.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/vidhalluc.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/mvbench.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/videomme.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/mlvu.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
bash scripts_eval/longvu/longvideobench.sh "path/LongVU_Qwen2_7B-RRPO-16f-LORA" "lora" "outputs/longvu-rrpo-16f"
```

You can also use our merged weights and directly perform evaluations using off-the-shelf evaluation kits. 
Here is an example:

```
accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=pritamqu/LLaVA-Video-7B-Qwen2-RRPO-32f,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=96 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "llava_video_rrpo" \
    --output_path ./logs/

```

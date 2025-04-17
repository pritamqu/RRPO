
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import json
import os
from peft import get_peft_config, get_peft_model
from safetensors.torch import load_file
import imageio

def load_model(model_path, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2"):

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, 
                                                                None, 
                                                                model_name, 
                                                                device_map=device_map, 
                                                                torch_dtype=torch_dtype, 
                                                                attn_implementation=attn_implementation)

    
    model.eval()
    model.config.use_cache = True
    ## as per: https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/llava_vid.py
    model.config.mm_spatial_pool_mode = 'average'
    tokenizer.pad_token_id = 151643
    return tokenizer, model, image_processor


def load_video(video_path, max_frames_num=64, return_meta=False):

    """
    sample fixed number of frames based on max_frames_num
    """
    
    if video_path.lower().endswith(('.gif')):
        gif = imageio.get_reader(video_path)
        frame_indices = [i for i in range(0, len(gif))]
        interval = len(frame_indices) / float(max_frames_num)
        frame_indices = [int(interval * i) for i in range(max_frames_num)]
        video = []
        for index, frame in enumerate(gif): 
            if index in frame_indices:
                frame_copy = frame.copy()
                video.append(frame_copy.convert("RGB"))
        video = np.stack(video)
    elif video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        frame_indices = [i for i in range(0, len(vr))]
        interval = len(frame_indices) / float(max_frames_num)
        frame_indices = [int(interval * i) for i in range(max_frames_num)]
        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        video = np.stack(video)
    else:
        NotImplementedError('unsupported file: ', video_path)

    if return_meta:
        sec = [round(f / fps, 1) for f in frame_indices]
        return video, fps, frame_indices, sec
    return video

def get_model_output(model, 
                     tokenizer,
                     image_processor,
                     vid_path, num_frame, question, max_new_tokens=1024, temperature=0.0):
    
    gen_kwargs = {"do_sample": True if temperature > 0 else False, 
                "temperature": temperature, 
                "top_p": None, 
                "num_beams": 1, 
                "use_cache": True, 
                "max_new_tokens": max_new_tokens}
    
    frames = load_video(vid_path, max_frames_num=num_frame, return_meta=False)
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(model.device, dtype=torch.float16)   

    question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nGive a detailed caption of the video as if I am blind.<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output


def get_model_output_from_loaded_video(model, 
                     tokenizer,
                     image_processor,
                     frames, question, max_new_tokens=1024, temperature=0.0):
    
    gen_kwargs = {"do_sample": True if temperature > 0 else False, 
                "temperature": temperature, 
                "top_p": None, 
                "num_beams": 1, 
                "use_cache": True, 
                "max_new_tokens": max_new_tokens}
    
    # print(gen_kwargs)

    # frames = load_video(vid_path, max_frames_num=num_frame, return_meta=False)
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(model.device, dtype=torch.float16)   

    question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output

def load_lora_weights(lora_weight_path, model):
    print("loading LoRA weights")
    lora_config_dict=json.load(open(os.path.join(lora_weight_path, 'adapter_config.json')))
    # _=lora_config_dict.pop('layer_replication')
    lora_config=get_peft_config(lora_config_dict)
    model = get_peft_model(model, lora_config)
    # FIXME: load .bin file
    # state_dict = load_file(os.path.join(lora_weight_path, "adapter_model.safetensors"))
    non_lora = torch.load(os.path.join(lora_weight_path, "non_lora_trainables.bin"))
    if len(non_lora):
        raise ValueError('we should not have anything here...')
    state_dict = torch.load(os.path.join(lora_weight_path, "adapter_model.bin"))

    # for k in model.state_dict():
    #     if 'lora' in k.lower():
    #         print(k)

    # for k in state_dict:
    #     print(k)

    new_dict={}
    for k in state_dict:
        new_name='.'.join(k.split('.')[:-1])+'.default.'+k.split('.')[-1]
        new_dict[new_name]=state_dict[k]

    msg = model.load_state_dict(new_dict, strict=False)
    # print(msg)
    assert len(msg.unexpected_keys)==0
    for k in msg.missing_keys:
        assert 'lora' not in k, print(k)

    model=model.merge_and_unload()
    return model

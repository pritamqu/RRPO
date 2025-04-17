import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader
import imageio
import json
import os
import torch
from peft import get_peft_config, get_peft_model
from safetensors.torch import load_file
import pathlib
# import decord
# decord.bridge.set_bridge("torch")

def load_video(video_path, max_num_frames, return_meta=False):
    
    if video_path.lower().endswith(('.gif')):
        gif = imageio.get_reader(video_path)
        # frame_indices = np.array([i for i in range(0, min(len(gif), max_num_frames), round(fps),)])
        frame_indices = [i for i in range(0, len(gif), round(fps))] # at max we are using 1fps
        if max_num_frames==-1: # taking all
            max_num_frames=len(frame_indices)

        if len(frame_indices)>max_num_frames: 
            interval = len(frame_indices) / float(max_num_frames)
            frame_indices = [int(interval * i) for i in range(max_num_frames)]
        video = []
        for index, frame in enumerate(gif): 
            if index in frame_indices:
                frame_copy = frame.copy()
                video.append(frame_copy.convert("RGB"))
        video = np.stack(video)

    elif video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        # frame_indices = np.array([i for i in range(0, min(len(vr), max_num_frames), round(fps),)])
        frame_indices = [i for i in range(0, len(vr), round(fps))]
        if max_num_frames==-1: # taking all
            max_num_frames=len(frame_indices)

        if len(frame_indices)>max_num_frames:
            interval = len(frame_indices) / float(max_num_frames)
            frame_indices = [int(interval * i) for i in range(max_num_frames)]
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

def load_lora_weights(lora_weight_path, model):
    print(f"loading LoRA weights from {lora_weight_path}")
    lora_weight_path=list(pathlib.Path(lora_weight_path).glob("checkpoint-*")) # we are saving one checkoint
    assert len(lora_weight_path)==1, print(lora_weight_path)
    lora_weight_path=lora_weight_path[0]
    lora_config_dict=json.load(open(os.path.join(lora_weight_path, 'adapter_config.json')))

    lora_config=get_peft_config(lora_config_dict)
    # lora_config.target_modules=find_all_linear_names_exact(model)

    model = get_peft_model(model, lora_config)
    # load new weights
    # state_dict = torch.load(os.path.join(lora_weight_path, "adapter_model.bin"), "cpu")
    state_dict = load_file(os.path.join(lora_weight_path, "adapter_model.safetensors"))
    new_dict={}
    for k in state_dict:
        new_name='.'.join(k.split('.')[:-1])+'.default.'+k.split('.')[-1]
        new_dict[new_name]=state_dict[k]
        
    msg = model.load_state_dict(new_dict, strict=False)
    print(msg)
    assert len(msg.unexpected_keys)==0
    for k in msg.missing_keys:
        assert 'lora' not in k, print(k)

    model=model.merge_and_unload()

    return model

def get_model_output_from_loaded_video(model, 
                     tokenizer,
                     image_processor,
                     media, question, 
                     max_new_tokens=1024, 
                     temperature=0.0,
                     version='llama3'):

    model_config=model.config
    # media = load_video(vid_path, image_processor, model_config, num_frame)
    media = process_images(media, image_processor, model_config)
    media = [item.unsqueeze(0) for item in media]
    image_sizes = [media[0].shape[:2]]

    if getattr(model_config, "mm_use_im_start_end", False):
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    if "llama3" in version:
        input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=media,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output

def get_model_output(model, 
                     tokenizer,
                     image_processor,
                     vid_path, num_frame, question, 
                     max_new_tokens=1024, 
                     temperature=0.0,
                     version='llama3'):

    model_config=model.config
    media = load_video(vid_path, num_frame)
    media = process_images(media, image_processor, model_config)
    media = [item.unsqueeze(0) for item in media]
    image_sizes = [media[0].shape[:2]]

    if getattr(model_config, "mm_use_im_start_end", False):
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question

    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    
    if "llama3" in version:
        input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=media,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output

def get_model_output_language_only(model, 
                     tokenizer,
                     question, 
                     max_new_tokens=1024, 
                     temperature=0.0,
                     version='llama3'):

    # model_config=model.config
    # media = load_video(vid_path, image_processor, model_config, num_frame)
    # media = process_images(media, image_processor, model_config)
    # media = [item.unsqueeze(0) for item in media]
    # image_sizes = [media[0].shape[:2]]

    # if getattr(model_config, "mm_use_im_start_end", False):
    #         question = (
    #             DEFAULT_IM_START_TOKEN
    #             + DEFAULT_IMAGE_TOKEN
    #             + DEFAULT_IM_END_TOKEN
    #             + "\n"
    #             + question
    #         )
    # else:
    #     question = DEFAULT_IMAGE_TOKEN + "\n" + question

    question=question.replace(DEFAULT_IMAGE_TOKEN, '') # making sure language token is not there

    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    if "llama3" in version:
        input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos
      
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            # images=media,
            # image_sizes=image_sizes,
            do_sample=False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output

import datetime
import json
import logging
import os
import re
import shutil
import uuid
from itertools import chain
import argparse
import sys
sys.path.append('./')
import numpy as np
from collections import defaultdict
import pandas as pd
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord
from pyarrow import parquet as pq
from torch import distributed as dist
from tqdm import tqdm

from transformers.trainer_pt_utils import IterableDatasetShard
import subprocess
from longvu.load_longvu import get_model_output_from_loaded_video, load_lora_weights, load_pretrained_model

from eval.videomme.utils import load_subtitles, convert_time_to_frame
def extract_subtitles(subtitle_path, fps):
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames


def load_video(video_path, max_num_frames, return_meta=False):
    
    if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        # frame_indices = [i for i in range(0, len(vr), round(fps))]
        frame_indices = [i for i in range(0, len(vr), round(fps / sample_fps))]
        if max_num_frames!=-1 and len(frame_indices)>max_num_frames: # taking all
            interval = len(frame_indices) / float(max_num_frames)
            frame_indices = [int(interval * i) for i in range(max_num_frames)]

        # video = []
        # for frame_index in frame_indices:
        #     img = vr[frame_index].asnumpy()
        #     video.append(img)
        # video = np.stack(video)
        video = vr.get_batch(frame_indices).asnumpy()

        # print(video.shape)
    else:
        NotImplementedError('unsupported file: ', video_path)

    if return_meta:
        sec = [round(f / fps, 1) for f in frame_indices]
        return video, fps, frame_indices, sec
    return video

class InferenceDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        list_data_dict,
        video_load_fn,
        load_video_kwargs={},
    ) -> None:
        super(InferenceDataset, self).__init__()

        self.data = list_data_dict
        self.video_load_fn = video_load_fn
        self.load_video_kwargs = load_video_kwargs

    def __len__(self) -> int:
        return len(self.data)

    def get_sample(self, line):
        video_path = line[0]["video_path"]
        subtitle_path = line[0]["subtitle_path"]
        
        subtitle=''
        vid, fps, frame_indices, sec = self.video_load_fn(video_path, **self.load_video_kwargs)
        
        use_subtitle=True
        if use_subtitle:
            if os.path.isfile(subtitle_path):
                subtitle_by_frame =extract_subtitles(subtitle_path, fps)
                # choose subtitles based on frame indices or sec
                subtitle_by_frame_idx = []
                for frame_idx in frame_indices:
                    for idx, title in enumerate(subtitle_by_frame):
                        if frame_idx < title[1] and frame_idx >= title[0]:
                            subtitle_by_frame_idx.append(idx)
                subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

                textlist = []
                for idx in subtitle_by_frame_idx:
                    pattern = r'<font color="white" size=".72c">(.*?)</font>'
                    raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
                    try:
                        textlist.append(raw_text[0])
                    except:
                        continue
                subtitle = "\n".join(textlist)

        return line, vid, subtitle
    
    def __iter__(self):
        for line in self.data:
            yield self.get_sample(line)


def train(args) -> None:
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    from eval.ddp import init_distributed_mode
    init_distributed_mode(args)
    print('dist initialized...')

    if args.base_model_name == "longvu_qwen_7b":
        model_name="cambrian_qwen"
        version='qwen'
    elif args.base_model_name == "longvu_llama3_3b":
        model_name="cambrian_llama3"
        version='llama3'
    else:
        raise NameError()
    
    print('loading base model...')
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,  
        None,
        model_name,
        device_map=None,
    )

    if args.model_path2:
        model = load_lora_weights(args.model_path2, model)

    model.get_model().config.drop_threshold = 0.65
    model.config.use_cache = True
    model.cuda()


    load_video_kwargs=dict(max_num_frames=args.max_frames,
                        return_meta=True,
                        )
    video_load_fn=load_video
    
    gt_questions = pd.read_parquet(args.question_file, engine='pyarrow', )
    # gt_questions = json.load(open(args.question_file, "r"))
    samples=defaultdict(list)
    attrs=gt_questions.columns

    for i in range(len(gt_questions)):

        sample=gt_questions.iloc[i]
        tmp_dict={}
        for attr in attrs:
            if attr=='options':
                tmp_dict[attr]='\n'.join(sample[attr])
            else:
                tmp_dict[attr]=sample[attr]
    
        video_name = tmp_dict['videoID']
        tmp_dict['video_path'] = os.path.join(args.root_dir, 'data', video_name+'.mp4')
        tmp_dict['subtitle_path'] = os.path.join(args.root_dir, 'subtitle', video_name+'.srt')
        samples[tmp_dict['video_id']].append(tmp_dict)

    gt_questions = [samples[key] for key in samples]

    # wrap dataset
    dataset=InferenceDataset(list_data_dict=gt_questions, 
                        video_load_fn=video_load_fn, 
                        load_video_kwargs=load_video_kwargs)


    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()

    responses_with_sub=[]
    responses_without_sub=[]
    final_output_with_sub = [None] * world_size
    final_output_without_sub = [None] * world_size

    for batch in tqdm(shard_dataset):
        samples, vid, subtitle = batch
        #################### with subtitle ####################
        if args.use_subtitle:

            response_with_sub={}
            response_with_sub['video_id']=samples[0]['video_id']
            response_with_sub['duration']=samples[0]['duration']
            response_with_sub['domain']=samples[0]['domain']
            response_with_sub['sub_category']=samples[0]['sub_category']
            response_with_sub['questions']=[]
            for sample in samples:

                q = sample['question']
                options=sample['options']
                instruct = f"Question: {q}\n"
                instruct += "Options:\n"
                instruct += f"{options}\n"
                instruct += (
                    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                )

                # print(instruct)
                if subtitle:
                    prompt = subtitle + instruct
                else:
                    prompt = instruct
                
                temperature=0.0
                pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, vid, prompt, 
                                                        args.model_max_length, temperature=temperature, 
                                                        version=version)
                
                response_with_sub['questions'].append({
                    'question_id':sample['question_id'],
                    'task_type':sample['task_type'],
                    'question':sample['question'],
                    'options':sample['options'],
                    'answer':sample['answer'],
                    'response':pred,
                })

        #################### without subtitle ####################

        response_without_sub={}
        response_without_sub['video_id']=samples[0]['video_id']
        response_without_sub['duration']=samples[0]['duration']
        response_without_sub['domain']=samples[0]['domain']
        response_without_sub['sub_category']=samples[0]['sub_category']
        response_without_sub['questions']=[]
        for sample in samples:

            q = sample['question']
            options=sample['options']
            instruct = f"Question: {q}\n"
            instruct += "Options:\n"
            instruct += f"{options}\n"
            instruct += (
                "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            )

            prompt = instruct
            # print(instruct)
            temperature=0.0
            pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, vid, prompt, 
                                                    args.model_max_length, temperature=temperature, 
                                                    version=version)
            
        
            response_without_sub['questions'].append({
                'question_id':sample['question_id'],
                'task_type':sample['task_type'],
                'question':sample['question'],
                'options':sample['options'],
                'answer':sample['answer'],
                'response':pred,
            })


        if args.use_subtitle:
            responses_with_sub.append(response_with_sub)
        responses_without_sub.append(response_without_sub)
       

    dist.barrier()
    dist.all_gather_object(
        final_output_with_sub,
        responses_with_sub,
    )
    all_output_with_sub = list(chain(*final_output_with_sub))

    dist.all_gather_object(
        final_output_without_sub,
        responses_without_sub,
    )
    all_output_without_sub = list(chain(*final_output_without_sub))



    global_rank = dist.get_rank()
    if global_rank == 0:
        if args.use_subtitle:
            json.dump(all_output_with_sub, open(args.output_file, 'w'), indent=2)
        json.dump(all_output_without_sub, open(args.output_file2, 'w'), indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="longvu_qwen_7b")
    parser.add_argument('--model_path', default="./checkpoints/longvu_qwen")
    parser.add_argument('--model_path2', help='', default=None, type=str, required=False)
    parser.add_argument("--root-dir", type=str, default="/datasets/video_llm/video_eval/Video-MME")
    parser.add_argument("--question-file", type=str, default="/datasets/video_llm/video_eval/Video-MME/videomme/test-00000-of-00001.parquet")

    # parser.add_argument('--model_name', default="cambrian_qwen")
    # parser.add_argument('--version', default="qwen")
    # parser.add_argument('--data_path', default="/datasets/video_llm/video_eval/Video-MME")
    parser.add_argument('--use_subtitle', action="store_true")
    parser.add_argument('--output-file', help='with sub model results JSON.', 
                        # required=True, 
                        default=None)
    parser.add_argument('--output-file2', help='without sub model results JSON.', 
                        # required=True,
                        default=None)
    
    
    parser.add_argument("--model_max_length", type=int, required=False, default=16)
    parser.add_argument('--max_frames', type=int, default=-1, help='only used for longvu which can take infinitely long frames')
    parser.add_argument('--sample_fps', default=0.5, type=float)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    global sample_fps
    sample_fps = args.sample_fps

    print(f'max frames set to: {args.max_frames}')

    # if "llama3" in args.version:
    #     args.model_name = "cambrian_llama3"

    train(args)
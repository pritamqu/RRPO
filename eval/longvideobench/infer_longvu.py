import datetime
import json
import os
import re
import shutil
import uuid
from itertools import chain
import argparse
import sys
sys.path.append('./eval')
import numpy as np
from PIL import Image
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
from decord import cpu, VideoReader  
from torch import distributed as dist
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard
from longvu.load_longvu import load_lora_weights, get_model_output_from_loaded_video


def load_video(video_path, max_num_frames, return_meta=False):
    
    if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        # frame_indices = np.array([i for i in range(0, min(len(vr), max_num_frames), round(fps),)])
        frame_indices = [i for i in range(0, len(vr), round(fps/sample_fps))]
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



class LongVideoBenchDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 annotation_file,
                #  max_num_frames=256,
                #  insert_text=True,
                #  insert_frame=True,
                video_load_fn=None,
                load_video_kwargs=dict(),
                ):
        super().__init__()
        self.data_path = data_path
        # self.insert_text = insert_text

        with open(os.path.join(data_path, annotation_file)) as f:
            self.data = json.load(f)
        # self.max_num_frames = max_num_frames
        self.video_load_fn=video_load_fn
        self.load_video_kwargs=load_video_kwargs
        
    def __iter__(self):
        for line in self.data:
            yield self.get_sample(line)

    def get_sample(self, di):
        # di = self.data[index]    
        frames = self.video_load_fn(os.path.join(self.data_path, "videos", di["video_path"]), **self.load_video_kwargs)       
        inputs = []
        inputs += ["Question: " + di["question"]]
        inputs += ["Options:"]
        inputs += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(di["candidates"])]
        question = '\n'.join(inputs)

        return {"question": question, 
                "video": frames,
                "correct_choice": chr(ord("A")+di.get("correct_choice", -1)), "id": di["id"]}
    

    def __len__(self):
        return len(self.data)
    
    def get_id(self, index):
        return self.data[index]["id"]

def run_inference(args) -> None:

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

    # torch.distributed.barrier()
    print('loading base model...')
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,  
        None,
        model_name,
        device_map=None,
    )

    if args.model_path2:
        model = load_lora_weights(args.model_path2, model)

    model.get_model().config.drop_threshold = 0.8
    model.config.use_cache = True
    model.cuda()

    load_video_kwargs=dict(max_num_frames=args.max_frames, 
                        return_meta=False)
    dataset = LongVideoBenchDataset(data_path=args.root_dir, 
                                    annotation_file="lvb_val.json", 
                                    video_load_fn=load_video, 
                                    load_video_kwargs=load_video_kwargs)

    total_eval_samples=len(dataset)
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )
    torch.distributed.barrier()
    output = []
    final_output = [None] * world_size

    for sample in tqdm(shard_dataset):
        question = sample['question']
        video = sample['video']
        correct_choice = sample['correct_choice'][0]
        id = sample['id']

        answer_prompt="Respond with only the letter of the correct option.\n"
        question=question+answer_prompt

        pred=get_model_output_from_loaded_video(model, 
                                                tokenizer,
                                                image_processor,
                                                video,
                                                question,
                                                args.model_max_length, 
                                                temperature=0.0, 
                                                version=version)

        sample_set=dict(id=id, pred=pred, gt=correct_choice)
        
        output.append(
            sample_set
        )

    dist.barrier()
    dist.all_gather_object(
        final_output,
        output,
    )
    
    all_output = list(chain(*final_output))
    global_rank = dist.get_rank()
    if global_rank == 0:
        seen=[]
        pred_file = args.output_file
        ans_file = open(pred_file, "w")
        for i, x in enumerate(all_output):
            # remove duplicates due to ddp inference
            if x['id'] in seen:
                continue
            seen.append(x['id'])

            ans_file.write(json.dumps(x) + "\n")
            ans_file.flush()

        ans_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="longvu_qwen_7b")
    parser.add_argument('--model-path', default='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--root-dir", type=str, default='/datasets/video_llm/video_eval/LongVideoBench')
    parser.add_argument('--output_file', default='predictions/debug/response.jsonl')     
    parser.add_argument("--model_max_length", type=int, required=False, default=128)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--sample_fps', default=0.5, type=float)
    parser.add_argument('--max_frames', default=1000, type=int)

    args = parser.parse_args()

    global sample_fps
    sample_fps=args.sample_fps
    run_inference(args)

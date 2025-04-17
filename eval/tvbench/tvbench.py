import math
import os
import argparse
import json
import os
import json
import torch
import numpy as np
import torchvision.transforms as T
import imageio
import cv2
import re
from tqdm import tqdm
from eval.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor, IdentityTransform
)
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from PIL import Image
from videochat2.utils.basic_utils import set_deterministic
set_deterministic(seed=42)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

data_list = {
    "Action Count": ("action_count.json", "action_count", "video", False),
    "Object Count": ("object_count.json", "object_count", "video", False),
    "Action Sequence": ("action_sequence.json", "action_sequence", "video", True),  # has start & end
    "Object Shuffle": ("object_shuffle.json", "object_shuffle", "video", False),
    "Scene Transition": ("scene_transition.json", "scene_transition", "video", False),
    "Action Localization": ("action_localization.json", "action_localization", "video", True),  # has start & end
    "Action Antonym": ("action_antonym.json", "action_antonym", "video", False),
    "Unexpected Action": ("unexpected_action.json", "unexpected_action", "video", False),
    "Egocentric Sequence": ("egocentric_sequence.json", "egocentric_sequence", "video", False),
    "Moving Direction": ("moving_direction.json", "moving_direction", "video", False),
}


class TVBench_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        anno_root, video_root = data_dir+'/json', data_dir+'/video'
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(anno_root, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    # 'prefix': os.path.join(video_root, '/'.join(v[1].split('/')[1:])),
                    'prefix': os.path.join(video_root, v[1]),
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
        
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        if base_model_name in ['videochat2_mistral_7b']:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        
            for frame_index in frame_indices:
                img = Image.fromarray(vr[frame_index].numpy())
                images_group.append(img)
            torch_imgs = self.transform(images_group)
            return torch_imgs
        
        elif base_model_name in ['llavavideo_qwen_7b']:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
            for frame_index in frame_indices:
                images_group.append(vr[frame_index].asnumpy())
            images_group = np.stack(images_group)
            return images_group
        
        else:
            raise NotImplementedError(base_model_name)
        
    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        if base_model_name in ['videochat2_mistral_7b']:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
            for index, frame in enumerate(gif):
                if index in frame_indices:
                    img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    img = Image.fromarray(img)
                    images_group.append(img)
            torch_imgs = self.transform(images_group)
            return torch_imgs
        elif base_model_name in ['llavavideo_qwen_7b']:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
            for index, frame in enumerate(gif):
                if index in frame_indices:
                    img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    images_group.append(img) # noqa
            images_group = np.stack(images_group)
            return images_group
        else:
            raise NotImplementedError(base_model_name)
        
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        # frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        if base_model_name in ['videochat2_mistral_7b']:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) 
            for frame_index in frame_indices:
                img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
                images_group.append(img)
            torch_imgs = self.transform(images_group)
            return torch_imgs
        elif base_model_name in ['llavavideo_qwen_7b']:
            frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) 
            for frame_index in frame_indices:
                img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
                images_group.append(img.convert('RGB')) # noqa
            images_group = np.stack(images_group)
            return images_group
        else:
            raise NotImplementedError(base_model_name)
        

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        # # if not os.path.isfile(video_path) or not os.path.isdir(video_path):
        # if not os.path.exists(video_path):
        #     print(f"[Missing file] {video_path}")
        #     return self[(idx+1) % len(self)]
        
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }
    
# def check_ans(pred, gt):
#     flag = False
    
#     pred_list = pred.lower().split(' ')
#     pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
#     gt_list = gt.lower().split(' ')
#     gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
#     if gt_content[-1] == '.':
#         gt_content = gt_content[:-1]
    
#     if pred_option.replace('.', '') in gt_option:
#         flag = True
#     elif gt_option in pred_option:
#         flag = True
        
#     print(pred, gt, flag)
#     return flag

def check_ans(pred, gt):
    
    gt_pattern = r"\(?\[?([A-Z])\]?\)?\.?" # r"\((.*?)\)"
    pred_pattern = r"\(?\[?([A-Z])\]?\)?\.?" # r"\((.*?)\)"

    _gt=gt
    match=re.search(gt_pattern, _gt, re.IGNORECASE)
    _gt=match.group(1)

    _pred=pred
    match = re.search(pred_pattern, _pred, re.IGNORECASE)
    if match:
        _pred = match.group(1)
    else:
        print("[Error] failed to find the answer from raw text: ", _pred)
        _pred='failed'
    
    flag=False
    if _gt.lower()==_pred.lower():
        flag=True
        
    # print('PRED: ', pred, 'GT: ', gt, 'CORRECT: ', flag)
    return flag

def run_inference(args):

    # model
    if args.base_model_name == "videochat2_mistral_7b":
        from videochat2.load_videochat2 import load_model, \
            get_model_output_from_loaded_video, load_lora_weights
        model, resolution, num_frame = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

    elif args.base_model_name == "llavavideo_qwen_7b":
        from llava.load_llavavideo import load_model, \
            get_model_output_from_loaded_video, load_lora_weights
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        num_frame=args.max_frames
        resolution=336 # ignored

    else:
        raise NotImplementedError(args.base_model_name)


    dataset = TVBench_dataset(args.data_dir, data_list, num_segments=num_frame, resolution=resolution)   
    if args.base_model_name in ["llavavideo_qwen_7b"]:
        dataset.transform=IdentityTransform() # we will process the frames from outside

    from eval.ddp import make_dataloader
    loader = make_dataloader(dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    # for sample in tqdm(dataset):
    for sample in tqdm(loader):

        # remove extra batch dimension
        for key in sample:
            sample[key]=sample[key][0]


        task_type = sample['task_type']
        video = sample["video"]
        question = sample["question"]

        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        if args.base_model_name == "videochat2_mistral_7b":
            question_prompt="\nOnly give the best option."
            answer_prompt="Best option:("
            return_prompt='('
            pred=get_model_output_from_loaded_video(model, video, num_frame, resolution, 
                                    question, 
                                    args.model_max_length, 
                                device=args.device, 
                                system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
                                system_q=False,
                                system_llm=True,
                                question_prompt=question_prompt,
                                answer_prompt=answer_prompt,)
            pred = return_prompt + pred.strip().split('\n')[0]

        elif args.base_model_name == "llavavideo_qwen_7b":
            temperature=0.0
            pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, video, question, 
                                                    args.model_max_length, temperature=temperature)
            if pred in [f"{chr(k)}" for k in range(ord('A'), ord('Z')+1)]:
                pred=f"({pred})"
        
        else:
            raise NotImplementedError(args.base_model_name)

        gt = sample['answer']
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt): 
            acc_dict[task_type][0] += 1
            correct += 1
        # print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        # print(f"Total Acc: {correct / total * 100 :.2f}%")
        # print('-' * 30, task_type, '-' * 30)

    with open(os.path.join(args.output_dir, f'tvbench.json'), "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    print(final_res)

    with open(os.path.join(args.output_dir, f"upload_leaderboard.json"), "w") as f:
        json.dump(final_res, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument('--data_dir', default="/datasets/video_llm/video_eval/TVBench")
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output-dir', help='Directory to save the model results JSON.', required=True)
    # parser.add_argument("--trial", type=int, default=1)
    # parser.add_argument("--conv-mode", type=str, default="llava_v1")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    # parser.add_argument("--additional_prompt", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=64)

    args = parser.parse_args()

    global base_model_name
    base_model_name=args.base_model_name
    run_inference(args)

import torch
from torchvision import transforms
import json
from tqdm import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
from eval.ddp import make_dataloader
import argparse
import re

class MLVU(Dataset):
    def __init__(self, data_dir, data_list, 
                video_load_fn,
                load_video_kwargs):
        self.data_list = []
        self.video_load_fn=video_load_fn
        self.load_video_kwargs=load_video_kwargs
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'data': data
                })
        
    
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
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        video = self.video_load_fn(video_path, **self.load_video_kwargs)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video_path': video_path, 
            'video': video,
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }



# def check_ans(pred, gt):
#     flag = False

#     index=gt.index("(")
#     index2=gt.index(")")
#     gt_option=gt[index+1:index2]

#     if ")" in pred:
#         index3=pred.index(")")
#         pred=pred[index3-1:index3]

#     if pred==gt_option:
#         flag=True

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

def main(args):

    '''
    load your model
    '''
    # model
    if args.base_model_name == "videochat2_mistral_7b":
        from videochat2.load_videochat2 import load_model, \
            get_model_output_from_loaded_video, load_lora_weights, load_video
        
        model, resolution, num_frame = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        # for data loading
        load_video_kwargs=dict(num_segments=num_frame,
                        resolution=resolution,
                        )

    elif args.base_model_name == "llavavideo_qwen_7b":
        from llava.load_llavavideo import load_model, \
            get_model_output_from_loaded_video, load_lora_weights, load_video
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        num_frame=64
        resolution=336 # ignored
        load_video_kwargs=dict(max_frames_num=num_frame,
                        )

    else:
        raise NotImplementedError(args.base_model_name)

    data_list = {
        "count": ("4_count.json", f"{args.data_root}/MLVU/video/4_count", "video"),
        "ego": ("3_ego.json", f"{args.data_root}/MLVU/video/3_ego", "video"),
        "needle": ("2_needle.json", f"{args.data_root}/MLVU/video/2_needle", "video"),
        "order": ("5_order.json", f"{args.data_root}/MLVU/video/5_order", "video"),
        "plotQA": ("1_plotQA.json", f"{args.data_root}/MLVU/video/1_plotQA", "video"),
        "anomaly_reco": ("6_anomaly_reco.json", f"{args.data_root}/MLVU/video/6_anomaly_reco", "video"),
        "topic_reasoning": ("7_topic_reasoning.json", f"{args.data_root}/MLVU/video/7_topic_reasoning", "video")
    }
   
    data_dir = f"{args.data_root}/MLVU/json"
    save_path = f"{args.output_dir}/test_all_choice.json"
    result_path = f"{args.output_dir}/bench_all.json"

    
    
    dataset = MLVU(data_dir, data_list, 
                   video_load_fn=load_video,
                   load_video_kwargs=load_video_kwargs)
    loader = make_dataloader(dataset, num_workers=4)



    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    for example in tqdm(loader):
        for key in example:
            example[key]=example[key][0]

        video=example['video']
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        # video_path=example["video_path"]
        question=example["question"]

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
                                answer_prompt=answer_prompt,
                                )
            
            pred = return_prompt + pred.strip().split('\n')[0]

        elif args.base_model_name == "llavavideo_qwen_7b":
            temperature=0.0
            pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, video, question, 
                                                    args.model_max_length, temperature=temperature)
            
            if pred in [f"{chr(k)}" for k in range(ord('A'), ord('Z')+1)]:
                pred=f"({pred})"

        else:
            raise NotImplementedError(args.base_model_name)
        
        gt = example['answer']
        res_list.append({
            'pred': pred,
            'gt': gt,
            'question':example['question'],
            'question_type':example['task_type'],
            'video_path':example['video_path']
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        # print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        # print('-' * 30, task_type, '-' * 30)


    with open(save_path, "w") as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    total=0
    idx=0
    for k, v in acc_dict.items():
        idx+=1
        final_res[k] = v[0] / v[1] * 100  
        total+=final_res[k]
    final_res['Avg'] = total /idx 
    print(final_res)

    with open(result_path, "w") as f:
        json.dump(final_res, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=1024)
    parser.add_argument("--data_root", help="Directory containing the dataset", default="/datasets/video_llm/video_eval/MLVU")
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')

    args = parser.parse_args()
    main(args)

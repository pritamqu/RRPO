import math
import os
import argparse
import json
import os
import re
from tqdm import tqdm
import traceback
import pandas as pd
from collections import defaultdict
import torch
from videochat2.utils.basic_utils import set_deterministic
from eval.ddp import make_dataloader
set_deterministic(seed=42)

prompt_with_st='''This video's subtitles are listed below:
{subtitle}
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
{options}
'''
prompt_general='''{question}
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{options}
'''

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

from eval.videomme.utils import load_subtitles, convert_time_to_frame
def extract_subtitles(subtitle_path, fps):
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames


class InferenceDataset(torch.utils.data.Dataset):
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
    
    def __getitem__(self, i):
        line = self.data[i] 
        return self.get_sample(line)



def run_inference(args):

    # model
    if args.base_model_name == "videochat2_mistral_7b":
        from videochat2.load_videochat2 import load_model, \
            get_model_output_from_loaded_video, load_lora_weights, load_video_with_meta
        model, resolution, num_frame = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        # for data loading
        load_video_kwargs=dict(num_segments=num_frame,
                        return_meta=True,
                        resolution=resolution,
                        )
        video_load_fn=load_video_with_meta

    elif args.base_model_name == "llavavideo_qwen_7b":
        from llava.load_llavavideo import load_model, \
            get_model_output_from_loaded_video, load_lora_weights, load_video
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        num_frame=args.max_frames
        resolution=336 # ignored
        # for data loading
        load_video_kwargs=dict(max_frames_num=num_frame,
                        return_meta=True,
                        )

        video_load_fn=load_video

    else:
        raise NotImplementedError(args.base_model_name)

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

    loader = make_dataloader(dataset, num_workers=4)
    os.makedirs(os.path.dirname(args.output_file2), exist_ok=True)

    responses_with_sub=[]
    responses_without_sub=[]

    for batch in tqdm(loader):
        samples, vid, subtitle = batch
        # remove batch dimension from samples
        vid, subtitle = vid[0], subtitle[0]
        for sample in samples:
            for key in sample:
                sample[key]=sample[key][0]

        #################### with subtitle ####################
        if args.use_subtitle:

            response_with_sub={}
            response_with_sub['video_id']=samples[0]['video_id']
            response_with_sub['duration']=samples[0]['duration']
            response_with_sub['domain']=samples[0]['domain']
            response_with_sub['sub_category']=samples[0]['sub_category']
            response_with_sub['questions']=[]
            for sample in samples:

                question=sample['question']
                options=sample['options']
                subtitle=subtitle

                if subtitle:
                    prompt=prompt_with_st.format(subtitle=subtitle, question=question, options=options)
                else:
                    prompt=prompt_general.format(question=question, options=options)
                
                if args.base_model_name == "videochat2_mistral_7b":
                    # llm_message=get_model_output_from_loaded_video(model, vid, num_frame, resolution, prompt, args.model_max_length, 
                    #                     device=args.device, 
                    #                     system="", # having system worsen performance
                    #                     system_q=False,
                    #                     system_llm=False,
                    #                     )
                    
                    question_prompt="\nOnly give the best option."
                    answer_prompt="Best option:("
                    return_prompt='('
                    pred=get_model_output_from_loaded_video(model, vid, num_frame, resolution, prompt, args.model_max_length, 
                                        device=args.device, 
                                        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
                                        system_q=False,
                                        system_llm=True,
                                        question_prompt=question_prompt,
                                        answer_prompt=answer_prompt,)
                    pred = return_prompt + pred.strip().split('\n')[0]

                elif args.base_model_name == "llavavideo_qwen_7b":
                    temperature=0.0
                    pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, vid, prompt, 
                                                            args.model_max_length, temperature=temperature)
                    
                    if pred in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']:
                        pred=f"({pred})"

                else:
                    raise NotImplementedError(args.base_model_name)
                
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

            question=sample['question']
            options=sample['options']
            # subtitle=subtitle

            prompt=prompt_general.format(question=question, options=options)
            
            if args.base_model_name == "videochat2_mistral_7b":
                pred=get_model_output_from_loaded_video(model, vid, num_frame, resolution, prompt, args.model_max_length, 
                                    device=args.device, 
                                    system="",
                                    system_q=False,
                                    system_llm=False,
                                    )

            elif args.base_model_name == "llavavideo_qwen_7b":
                temperature=0.0
                pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, vid, prompt, 
                                                        args.model_max_length, temperature=temperature)
                
                if pred in [f"{chr(k)}" for k in range(ord('A'), ord('Z')+1)]:
                    pred=f"({pred})"

            else:
                raise NotImplementedError(args.base_model_name)
            
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
        
    if args.use_subtitle:
        json.dump(responses_with_sub, open(args.output_file, 'w'), indent=2)
    json.dump(responses_without_sub, open(args.output_file2, 'w'), indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument('--model-path', default='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--root-dir", type=str, default="/datasets/video_llm/video_eval/Video-MME")
    parser.add_argument("--question-file", type=str, default="/datasets/video_llm/video_eval/Video-MME/videomme/test-00000-of-00001.parquet")
    parser.add_argument('--output-file', help='with sub model results JSON.', 
                        # required=True, 
                        default=None)
    parser.add_argument('--output-file2', help='without sub model results JSON.', 
                        # required=True,
                        default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument('--use_subtitle', action="store_true")
    parser.add_argument("--max_frames", type=int, default=64)

    args = parser.parse_args()

    run_inference(args)
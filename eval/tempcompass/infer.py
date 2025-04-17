import os
import argparse
import json
import os
import torch
from tqdm import tqdm
from eval.ddp import make_dataloader
from videochat2.utils.basic_utils import set_deterministic
set_deterministic(seed=42)

answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""    # The answer "Generated Caption:" is already contained in the question
}



class InferenceDataset(torch.utils.data.Dataset):
    """basic dataset for inference"""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        list_data_dict,
        video_load_fn,
        load_video_kwargs={}
    ) -> None:
        super(InferenceDataset, self).__init__()
        self.data = list_data_dict
        self.video_load_fn = video_load_fn
        self.load_video_kwargs = load_video_kwargs

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, i):
        sample = self.data[i]
        assert len(sample)==2
        video_file = sample[0]
        data = sample[1]
        video = self.video_load_fn(video_file, **self.load_video_kwargs)
        return {'vid': video_file.split('/')[-1][:-4],
            'data': data,
            'video': video}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument('--model-path', default='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--root-dir", type=str, default='/datasets/video_llm/video_eval/TempCompass')
    parser.add_argument('--output_path', default='predictions/debug')     
    parser.add_argument('--task_type', default='multi-choice', choices=['multi-choice', 'captioning', 'caption_matching', 'yes_no'])
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=64)

    args = parser.parse_args()

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
        from llava.load_llavavideo import load_model, load_video, \
            get_model_output_from_loaded_video, load_lora_weights
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        num_frame=args.max_frames
        resolution=336 # ignored

        load_video_kwargs=dict(max_frames_num=num_frame,
                        return_meta=False,
                        )
        
    else:
        raise NotImplementedError(args.base_model_name)

    # Loading questions
    question_path = f"{args.root_dir}/questions/{args.task_type}.json"
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pred_file = f"{args.output_path}/{args.task_type}.json"
    # Loading existing predictions
    if os.path.isfile(pred_file):
        with open(f"{args.output_path}/{args.task_type}.json", 'r') as f:
            predictions = json.load(f)
    else:
        predictions = {}


    gt_questions=[]
    for vid, data in tqdm(input_datas.items()):
        video_path = os.path.join(args.root_dir, 'videos', f'{vid}.mp4')
        gt_questions.append([video_path, data])

    
    dataset = InferenceDataset(gt_questions, 
                                video_load_fn=load_video, 
                                load_video_kwargs=load_video_kwargs)

    loader = make_dataloader(dataset, num_workers=4)

    for idx, sample in enumerate(tqdm(loader)):
        # break
        # remove batch dimension from samples
        vid = sample['vid'][0] # this is just the video name
        video_tensor = sample['video'][0] # this is loaded video
        data = sample['data']
        for key in data:
            for qa_pair in data[key]:
                # print(qa_pair)
                qa_pair['question']=qa_pair['question'][0]
                qa_pair['answer']=qa_pair['answer'][0]



        if vid not in predictions:
            predictions[vid] = {}

            for dim, questions in data.items():
                predictions[vid][dim] = []
                for question in questions:
                    model_input = question['question'] + answer_prompt[args.task_type]
                    if args.base_model_name == "videochat2_mistral_7b":
                        output=get_model_output_from_loaded_video(model, video_tensor, num_frame, resolution, model_input, args.model_max_length, 
                                            device=args.device, 
                                            system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, answer the given question.\n",
                                            system_q=False,
                                            system_llm=True,)
                        # print(output)

                    elif args.base_model_name == "llavavideo_qwen_7b":
                        temperature=0.0
                        output=get_model_output_from_loaded_video(model, tokenizer, image_processor, video_tensor, model_input, 
                                                                args.model_max_length, temperature=temperature)
                        
                        if output in [f"{chr(k)}" for k in range(ord('A'), ord('Z')+1)]:
                            output=f"({output})"

                    else:
                        raise NotImplementedError(args.base_model_name)

                    predictions[vid][dim].append({'question': question['question'], 
                                                  'answer': question['answer'], 
                                                  'prediction': output})
            



            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=4)
import os
import argparse
import json
import os
import torch
from tqdm import tqdm
from eval.ddp import make_dataloader
from PIL import Image
from videochat2.utils.basic_utils import set_deterministic
set_deterministic(seed=42)

prompt_with_st='''This video's subtitles are listed below:
{subtitle}
'''

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
        
    def __getitem__(self, index):
        di = self.data[index]    
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
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument('--model-path', default='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--root-dir", type=str, default='/datasets/video_llm/video_eval/LongVideoBench')
    parser.add_argument('--output_file', default='predictions/debug/response.jsonl')     
    parser.add_argument("--model_max_length", type=int, required=False, default=128)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()

    # model
    if args.base_model_name == "videochat2_mistral_7b":
        from videochat2.load_videochat2 import load_model, load_video, \
            get_model_output_from_loaded_video, load_lora_weights
        model, resolution, num_frame = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        # for data loading
        load_video_kwargs=dict(num_segments=num_frame,
                            resolution=resolution,
                            # return_meta=False,
                            )

    elif args.base_model_name == "llavavideo_qwen_7b":
        from llava.load_llavavideo import load_model, load_video, \
            get_model_output_from_loaded_video, load_lora_weights
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        num_frame=64
        resolution=336 # ignored

        load_video_kwargs=dict(max_frames_num=num_frame,
                        )

    else:
        raise NotImplementedError(args.base_model_name)


    pred_file = args.output_file
    ans_file = open(pred_file, "w")

    dataset = LongVideoBenchDataset(data_path=args.root_dir, 
                                    annotation_file="lvb_val.json", 
                                    video_load_fn=load_video, 
                                    load_video_kwargs=load_video_kwargs)

    loader = make_dataloader(dataset, num_workers=2)
   
    for idx, sample in enumerate(tqdm(loader)):
        # break
        # remove batch dimension from samples
        question = sample['question'][0]
        video = sample['video'][0]
        correct_choice = sample['correct_choice'][0]

        id = sample['id'][0]
        
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
            # print(output)

        elif args.base_model_name == "llavavideo_qwen_7b":
            temperature=0.0
            answer_prompt="\nAnswer with the option's letter from the given choices directly."
            question=question+answer_prompt
            pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, video, question, 
                                                    args.model_max_length, temperature=temperature)
            
            if pred in [f"{chr(k)}" for k in range(ord('A'), ord('Z')+1)]:
                pred=f"({pred})"

        else:
            raise NotImplementedError(args.base_model_name)

        sample_set=dict(id=id, pred=pred, gt=correct_choice)
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
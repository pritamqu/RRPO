import argparse
import json
import os
import math
from tqdm import tqdm

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument('--model-path', default='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument('--save_path', help='results JSON.', default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=512)
    return parser.parse_args()

def run_inference(inputs, tokenizer, model, image_processor):

    video_path = inputs['video_path']

    all_video_pathes = []

    # Check if the video_path is a directory or a file
    if os.path.isdir(video_path):
        # If it's a directory, loop over all files in the directory
        for filename in os.listdir(video_path):
            # Load the video file
            cur_video_path = os.path.join(video_path, f"{filename}")
            all_video_pathes.append(os.path.join(video_path, cur_video_path))
    else:
        # If it's a file, just process the video
        all_video_pathes.append(video_path) 

    # import pdb;pdb.set_trace()
    for video_path in all_video_pathes:

        question = inputs['question']
        
        # Check if the video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
            
        # FIXME: finish setup
        # run inference
        if args.base_model_name == "videochat2_mistral_7b":
            system_msg="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, answer the question.\n"
            outputs=get_model_output(model, 
                                     video_path, 
                                     num_frame, 
                                     resolution, 
                                     question, 
                                     args.model_max_length, 
                                     device='cuda', 
                                     system=system_msg,
                                     system_q=False,
                                     system_llm=True,
                                     question_prompt='',
                                     answer_prompt=None,)

        elif args.base_model_name == "llavavideo_qwen_7b":
            outputs=get_model_output(model, 
                     tokenizer,
                     image_processor,
                     video_path, 
                     num_frame=num_frame, 
                     question=question, 
                     max_new_tokens=args.model_max_length, 
                     temperature=0.0)

        elif args.base_model_name == "longvu_qwen_7b":
            outputs=get_model_output(model, 
                     tokenizer,
                     image_processor,
                     video_path, 
                     num_frame=-1, # take all frames at 1fps 
                     question=question, 
                     max_new_tokens=args.model_max_length, 
                     temperature=0.0,
                     version=version)

        else:
            raise NotImplementedError()
        
        # print(f"Question: {question}\n")
        # print(f"Response: {outputs}\n")
        outputs = outputs.strip()

        # sample_set["pred"] = outputs
        # ans_file.write(json.dumps(sample_set, ensure_ascii=False) + "\n")
        # ans_file.flush()

    # ans_file.close()
    return outputs

if __name__ == "__main__":
    args = parse_args()

    # model
    if args.base_model_name == "videochat2_mistral_7b":
        from videochat2.load_videochat2 import load_model, \
            get_model_output, load_lora_weights
        model, resolution, num_frame = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)

        tokenizer=None
        image_processor=None

    elif args.base_model_name == "llavavideo_qwen_7b":
        from llava.load_llavavideo import load_model, \
            get_model_output, load_lora_weights
        tokenizer, model, image_processor = load_model(args.model_path)
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)
        num_frame=64
        
    elif args.base_model_name == "longvu_qwen_7b":
        model_name="cambrian_qwen"
        version='qwen'
        from longvu.builder import load_pretrained_model as load_model
        from longvu.load_longvu import get_model_output, load_lora_weights
        tokenizer, model, image_processor, context_len = load_model(
                                        args.model_path,  
                                        None,
                                        model_name,
                                        device_map=None,
                                    )
        if args.model_path2:
            model = load_lora_weights(args.model_path2, model)
        
        model.config.use_cache = True
        model.cuda()

    else:
        raise NotImplementedError(args.base_model_name)

    # video_folder = '/path/to/your/video_folder'
    # videos = '/path/to/your/json_file'

    video_folder = '/datasets/video_llm/video_eval/VidHalluc/data_fps_5_res_480/STH/'
    videos = '/datasets/video_llm/video_eval/VidHalluc/sth.json'

    with open(videos, 'r') as f:
        videos_dict = json.load(f)
    
    results = {}
    for video_id, video_dict in tqdm(videos_dict.items()):
        if video_id in results:
            continue
        temp = {}
        video_path = None
        for root, dirs, files in os.walk(video_folder):
            for file in files:
                if file == f"{video_id}.mp4":
                    video_path = os.path.join(root, file)
                    break
            if video_path:
                break

        if not video_path:
            print(f"Video file {video_id}.mp4 not found.")
            continue
        
        question = "Watch the given video and determine if a scene change occurs. If no change occurs, respond: 'Scene change: No, Locations: None'. If there is a scene change, respond in the format: 'Scene change: Yes, Locations: from [location1] to [location2].'"

        # args.prompt = question
        # args.video_path = video_path
        # print(args.video_path)
        # print(question)
        
        model_answer = run_inference(dict(question=question, video_path=video_path), 
                                     tokenizer, model, image_processor)

        
        results[video_id] = model_answer
        # print(model_answer, '\n')
        with open(args.save_path, 'w') as f:
            json.dump(results, f, indent=4)
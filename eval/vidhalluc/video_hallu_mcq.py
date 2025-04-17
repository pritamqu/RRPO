import argparse
import json
import os
import math
from tqdm import tqdm
import numpy as np
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'


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
    if os.path.isdir(video_path):
        # If it's a directory, loop over all files in the directory
        for filename in os.listdir(video_path):
            # Load the video file
            cur_video_path = os.path.join(video_path, f"{filename}")
            all_video_pathes.append(os.path.join(video_path, cur_video_path))
    else:
        # If it's a file, just process the video
        all_video_pathes.append(video_path) 

    for video_path in all_video_pathes:
        question = inputs['question']
        choices = inputs['choices']
        options_string=""
        for key, value in choices.items():
            options_string += f"({key}) {value}\n"



        # sample_set["Q"] = question
        # sample_set["video_name"] = video_path
        
        # Check if the video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        
        # finish setup
        # run inference
        if args.base_model_name == "videochat2_mistral_7b":
            question=f"Question: {question}\nOptions:\n{options_string}"
            question_prompt="\nOnly give the best option."
            answer_prompt="Best option:("
            return_prompt='('
            system_msg="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
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
                                     question_prompt=question_prompt,
                                     answer_prompt=answer_prompt,)
            outputs = return_prompt + outputs.strip().split('\n')[0]

        elif args.base_model_name == "llavavideo_qwen_7b":
            answer_prompt=""
            question=f"Question: {question}\nOptions:\n{options_string}{answer_prompt}"
            outputs=get_model_output(model, 
                     tokenizer,
                     image_processor,
                     video_path, 
                     num_frame=num_frame, 
                     question=question, 
                     max_new_tokens=args.model_max_length, 
                     temperature=0.0)
            
        elif args.base_model_name == "longvu_qwen_7b":
            answer_prompt="Answer with the option's letter from the given choices directly and only give the best option."
            question=f"Question: {question}\nOptions:\n{options_string}{answer_prompt}"
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
    video_folder = '/datasets/video_llm/video_eval/VidHalluc/data_fps_5_res_480/ACH/'
    videos = '/datasets/video_llm/video_eval/VidHalluc/ach_mcq.json'

    with open(videos, 'r') as f:
        videos_dict = json.load(f)

    results = {}
    for index, video_dict in tqdm(videos_dict.items()):
        if index in results:
            continue
        temp = {}
        for clip_name, question_data in video_dict.items():
            video = f"{video_folder}{clip_name}.mp4"

            question = question_data['Question']
            choices = question_data['Choices']
            correct_answer = question_data['Correct Answer']

            # Call main()
            # Please select the correct answer (one or more options), 
            # inp = question + " Please select the correct answer (one or more options), only return your answer(s). (e.g., ABCD)" + "\nChoices:\n"
            # for key, value in choices.items():
            #     inp += f"{key}. {value}\n"
            # args.prompt = inp
            # args.video_path = video
            # print(args.video_path)
            # print(inp)

            # FIXME: do input setup inside run inference based on model's setup
            selected_choice = run_inference(dict(question=question, choices=choices, video_path=video),
                                            tokenizer, model, image_processor)
            # Store the results  
            temp[clip_name] = {
                'Question': question,
                'Choices': choices,
                'Correct Answer': correct_answer,
                'Model Answer': selected_choice,
            }
            # print(correct_answer, selected_choice)

        results[index] = temp
  
        with open(args.save_path, 'w') as f:
            json.dump(results, f, indent=4)
import os
import argparse
import json
import os
import torch
from tqdm import tqdm
from itertools import chain
from transformers.trainer_pt_utils import IterableDatasetShard
from eval.ddp import init_distributed_mode
from longvu.builder import load_pretrained_model
from longvu.load_longvu import get_model_output_from_loaded_video, load_lora_weights, load_video
from videochat2.utils.basic_utils import set_deterministic
set_deterministic(seed=42)

answer_prompt = {
    "multi-choice": "\nPlease directly give the best option:",
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "captioning": ""    # The answer "Generated Caption:" is already contained in the question
}


class InferenceDataset(torch.utils.data.IterableDataset):
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
    
    def __iter__(self):
        for sample in self.data:
            assert len(sample)==2
            video_file = sample[0]
            data = sample[1]
            video = self.video_load_fn(video_file, **self.load_video_kwargs)
            yield {'vid': video_file.split('/')[-1][:-4],
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
    parser.add_argument("--max_frames", type=int, default=-1, help='only used for longvu which can take infinitely long frames')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    init_distributed_mode(args)
    print('dist initialized...')

    if args.base_model_name == "longvu_qwen_7b":
        model_name="cambrian_qwen"
        version="qwen"
    elif args.base_model_name == "longvu_llama3_3b":
        model_name="cambrian_llama3"
        version="llama3"
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
        # from longvu.load_longvu import load_lora_weights
        model = load_lora_weights(args.model_path2, model)

    # model.get_model().config.drop_threshold = 0.8
    model.config.use_cache = True
    model.cuda()

    # for data loading
    num_frame=args.max_frames
    load_video_kwargs=dict(max_num_frames=num_frame, 
                        return_meta=False)

    # Loading questions
    question_path = f"{args.root_dir}/questions/{args.task_type}.json"
    with open(question_path, 'r') as f:
        input_datas = json.load(f)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    raw_output_file = f"{args.output_path}/{args.task_type}_raw.json" # remove this once script is stable
    pred_file = f"{args.output_path}/{args.task_type}.json"

    gt_questions=[]
    for vid, data in tqdm(input_datas.items()):
        video_path = os.path.join(args.root_dir, 'videos', f'{vid}.mp4')
        gt_questions.append([video_path, data])

    dataset = InferenceDataset(gt_questions, 
                                    video_load_fn=load_video, 
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
    output=[]
    final_output = [None] * world_size
    # for vid, data in tqdm(input_datas.items()):
    for idx, sample in enumerate(tqdm(shard_dataset)):

        vid = sample['vid'] # this is just the video name
        video = sample['video'] # this is loaded video
        data = sample['data']

        # predictions[vid] = {}
        sample_set = {'vid': vid, 'pred': {}}
        for dim, questions in data.items():
            # predictions[vid][dim] = []
            sample_set['pred'][dim] = []
            for question in questions:
                model_input = question['question'] + answer_prompt[args.task_type]
                pred=get_model_output_from_loaded_video(model, 
                                                            tokenizer,
                                                            image_processor,
                                                            video,
                                                            model_input,
                                                            args.model_max_length, 
                                                            temperature=0.0, 
                                                            version=version)

                sample_set['pred'][dim].append({
                                'question': question['question'], 
                                'answer': question['answer'], 
                                'prediction': pred})
                            
        # with open(pred_file, 'w') as f:
        #     json.dump(predictions, f, indent=4)   
                
        output.append(sample_set)


    torch.distributed.barrier()
    torch.distributed.all_gather_object(
        final_output,
        output,
    )

    all_output = list(chain(*final_output))

    global_rank = torch.distributed.get_rank()
    if global_rank == 0:   
        # save in the format needed for eval file
        predictions = {}
        for sample_set in all_output:
            vid=sample_set['vid']
            predictions[vid]={}
            for dim in sample_set['pred']:
                predictions[vid][dim]=sample_set['pred'][dim]

        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=4)   
                
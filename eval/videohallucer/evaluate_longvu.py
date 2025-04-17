import argparse
import json
import sys
import os
import numpy as np
import random
import uuid
from collections import defaultdict
from itertools import chain
from typing import Callable
from tqdm import tqdm
import torch
from .evaluation_utils import cal_score, setup_seed
from longvu.load_longvu import get_model_output_from_loaded_video, load_lora_weights, load_video, load_pretrained_model
from transformers.trainer_pt_utils import IterableDatasetShard
from eval.ddp import init_distributed_mode
DATA_DIR="/datasets/video_llm/video_eval/VideoHallucer"
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'



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
            basic_dict = sample.pop('basic')
            halluc_dict = sample.pop('hallucination')
            common_dict = {**sample}

            basic_video = self.video_load_fn(basic_dict['video_path'], **self.load_video_kwargs)
            halluc_video = self.video_load_fn(halluc_dict['video_path'], **self.load_video_kwargs)

            yield basic_dict, basic_video, halluc_dict, halluc_video, common_dict

    def __getitem__(self, i):
        sample = self.data[i]

        basic_dict = sample.pop('basic')
        halluc_dict = sample.pop('hallucination')
        common_dict = {**sample}

        basic_video = self.video_load_fn(basic_dict['video_path'], **self.load_video_kwargs)
        halluc_video = self.video_load_fn(halluc_dict['video_path'], **self.load_video_kwargs)

        return basic_dict, basic_video, halluc_dict, halluc_video, common_dict




def prepare_questions(
    qa_path,
    qa_type,
    video_dir_path,
):

    paired_qas = json.load(open(qa_path))
    for qa_dct in paired_qas:

        # basic
        basic = qa_dct["basic"]
        basic_question = basic["question"]
        basic_question = f"{basic_question}\nAnswer the question using 'yes' or 'no'."
        basic_video_path = os.path.join(video_dir_path, basic["video"])
        qa_dct["basic"]["predict"] = None
        # what we need for inference
        qa_dct["basic"]["video_path"] = basic_video_path
        qa_dct["basic"]["basic_question"] = basic_question

        # hallucination
        halluc = qa_dct["hallucination"]
        halluc_question = halluc["question"]
        halluc_question = f"{halluc_question}\nAnswer the question using 'yes' or 'no'."
        halluc_video_path = os.path.join(video_dir_path, halluc["video"])
        qa_dct["hallucination"]["predict"] = None
        # what we need for inference
        qa_dct["hallucination"]["video_path"] = halluc_video_path
        qa_dct["hallucination"]["halluc_question"] = halluc_question

        qa_dct["category"] = qa_type

    # print(qa_dct.keys())

    return paired_qas # list

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_name", type=str, default="videochat2_mistral_7b")
    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--model-path2', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=1024)
    parser.add_argument("--device", type=str, default='cuda')

    # parser.add_argument("--model_name", type=str,
    #                     default="", 
    #                     choices=["VideoChatGPT", "Valley2", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video",
    #                              "Gemini-1.5-pro", "GPT4O",
    #                              "LLaVA", "GPT4V", 
    #                              "Video-LLaMA-2-13B", "LLaMA-VID-13B", "PLLaVA-13B", 
    #                              "PLLaVA-34B", "LLaVA-NeXT-Video-34B"])

    parser.add_argument(
        "--output_dir_path", type=str, default="results",
    )

    # Per-dataset evaluation flags
    parser.add_argument(
        "--eval_obj_rel",
        action="store_true",
        default=False,
        help="Whether to evaluate on object&relation hallucination",
    )
    parser.add_argument(
        "--eval_temporal",
        action="store_true",
        default=False,
        help="Whether to evaluate on temporal hallucination.",
    )
    parser.add_argument(
        "--eval_semantic",
        action="store_true",
        default=False,
        help="Whether to evaluate on other semantic detail hallucination.",
    )
    parser.add_argument(
        "--eval_fact",
        action="store_true",
        default=False,
        help="Whether to evaluate on fact hallucination.",
    )
    parser.add_argument(
        "--eval_nonfact",
        action="store_true",
        default=False,
        help="Whether to evaluate on fact hallucination.",
    )
    parser.add_argument(
        "--detect_fact",
        action="store_true",
        default=False,
        help="Whether to detect factual and nonfactula knowledge.",
    )

    ## Object-Relation Dataset
    parser.add_argument(
        "--obj_rel_path",
        type=str,
        default="object_relation/object_relation.json",
    )
    parser.add_argument(
        "--obj_rel_video_dir_path",
        type=str,
        default="object_relation/videos",
    )
    ## Temporal Dataset
    parser.add_argument(
        "--temporal_path",
        type=str,
        default="temporal/temporal.json",
    )
    parser.add_argument(
        "--temporal_video_dir_path",
        type=str,
        default="temporal/videos",
    )
    ## Other Semantic Detail Dataset
    parser.add_argument(
        "--semantic_path",
        type=str,
        default="semantic_detail/semantic_detail.json",
    )
    parser.add_argument(
        "--semantic_video_dir_path",
        type=str,
        default="semantic_detail/videos",
    )
    ## External Fact Dataset
    parser.add_argument(
        "--fact_path",
        type=str,
        default="external_factual/external_factual.json",
    )
    parser.add_argument(
        "--fact_video_dir_path",
        type=str,
        default="external_factual/videos",
    )
    ## External Non-Fact Dataset
    parser.add_argument(
        "--nonfact_path",
        type=str,
        default="external_nonfactual/external_nonfactual.json",
    )
    parser.add_argument(
        "--nonfact_video_dir_path",
        type=str,
        default="external_nonfactual/videos",
    )
    ## Fact-Nonfact Detect Dataset
    parser.add_argument(
        "--factdet_path",
        type=str,
        default="fact_detect/fact_detect.json",
    )
    parser.add_argument(
        "--factdet_video_dir_path",
        type=str,
        default="fact_detect/videos",
    )

    parser.add_argument("--max_frames", type=int, default=-1, help='only used for longvu which can take infinitely long frames')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    setup_seed(seed=42)

    gt_questions = []
    if args.eval_obj_rel:
        gt_questions.extend(prepare_questions(
            qa_path=os.path.join(DATA_DIR, args.obj_rel_path),
            qa_type='obj_rel',
            video_dir_path=os.path.join(DATA_DIR, args.obj_rel_video_dir_path),
        ))

    if args.eval_temporal:
        gt_questions.extend(prepare_questions(
            qa_path=os.path.join(DATA_DIR, args.temporal_path),
            qa_type='temporal',
            video_dir_path=os.path.join(DATA_DIR, args.temporal_video_dir_path),
        ))

    if args.eval_semantic:
        gt_questions.extend(prepare_questions(
            qa_path=os.path.join(DATA_DIR, args.semantic_path),
            qa_type='semantic',
            video_dir_path=os.path.join(DATA_DIR, args.semantic_video_dir_path),
        ))
    
    if args.eval_fact:
        gt_questions.extend(prepare_questions(
            qa_path=os.path.join(DATA_DIR, args.fact_path),
            qa_type='fact',
            video_dir_path=os.path.join(DATA_DIR, args.fact_video_dir_path),
        ))

    if args.eval_nonfact:
        gt_questions.extend(prepare_questions(
            qa_path=os.path.join(DATA_DIR, args.nonfact_path),
            qa_type='nonfact',
            video_dir_path=os.path.join(DATA_DIR, args.nonfact_video_dir_path),
        ))
    
    if args.detect_fact:
        gt_questions.extend(prepare_questions(
            qa_path=os.path.join(DATA_DIR, args.factdet_path),
            qa_type='factdet',
            video_dir_path=os.path.join(DATA_DIR, args.factdet_video_dir_path),
        ))



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
        # from longvu.load_longvu import load_lora_weights
        model = load_lora_weights(args.model_path2, model)

    # model.get_model().config.drop_threshold = 0.8
    model.config.use_cache = True
    model.cuda()

    # for data loading
    num_frame=args.max_frames
    load_video_kwargs=dict(max_num_frames=num_frame, 
                            return_meta=False)

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
    output = []
    final_output = [None] * world_size


    # iterate over each sample in the ground truth file
    for idx, batch in enumerate(tqdm(shard_dataset)):

        basic_dict, basic_video, halluc_dict, halluc_video, common_dict = batch

        # each sample has one basic and one hall qs

        # basic
        basic_pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, basic_video, 
                                                basic_dict['basic_question'], args.model_max_length, 
                                                temperature=0.0, 
                                                version=version)
            
        basic_dict['predict']=basic_pred

        # halluc
        halluc_pred=get_model_output_from_loaded_video(model, tokenizer, image_processor, halluc_video, 
                                                halluc_dict['halluc_question'], args.model_max_length, 
                                                temperature=0.0, 
                                                version=version)
            
        halluc_dict['predict']=halluc_pred
        
        output.append({'basic': basic_dict, 'hallucination': halluc_dict, **common_dict})
        # ans_file.write(json.dumps(sample_set) + "\n")
        # ans_file.flush()

    # ans_file.close()
    torch.distributed.barrier()
    torch.distributed.all_gather_object(
        final_output,
        output,
    )

    all_output = list(chain(*final_output))
    global_rank = torch.distributed.get_rank()
    output_path = os.path.join(args.output_dir_path, "raw_responses.json")
    if global_rank == 0:
        with open(output_path, "w",) as f:
            json.dump(all_output, f, indent=4)

        # separate based on category 
        separate_paired_qas = defaultdict(list)
        for sample in all_output:
            separate_paired_qas[sample['category']].append(sample)

        final_result={}
        for category in separate_paired_qas:
            final_result[category]=cal_score(separate_paired_qas[category])

        final_acc = 0
        final_basic_acc = 0
        final_halluc_acc = 0
        eval_type = ""
        for halluc_type, result in final_result.items():
            eval_type += halluc_type + "_"
            final_basic_acc += result["basic_accuracy"]
            final_halluc_acc += result["halluc_accuracy"]
            final_acc += result["accuracy"]
        if len(final_result.keys()) != 0:
            final_acc = final_acc / len(final_result.keys())
            final_basic_acc = final_basic_acc / len(final_result.keys())
            final_halluc_acc = final_halluc_acc / len(final_result.keys())
            final_result["all"] = {
                "basic_accuracy": final_basic_acc,
                "halluc_accuracy": final_halluc_acc,
                "accuracy": final_acc,
            }

            # eval_result_path = os.path.join(args.output_dir_path, f"{eval_type}{args.model_name}_evaluation_results.json")
            eval_result_path = os.path.join(args.output_dir_path, f"{eval_type}_evaluation_results.json")
            with open(eval_result_path, "w") as jp:
                json.dump(final_result, jp, indent=4)
            print("="*20)
            print("Basic Accuracy: ", final_basic_acc)
            print("Hallucination Accuracy: ", final_halluc_acc)
            print("Final Accuracy: ", final_acc)
            if args.detect_fact:
                print("Fact Score: ", (final_basic_acc + final_halluc_acc)/2)
            print("="*20)

if __name__ == "__main__":
    main()
import datetime
import json
import os
import re
import shutil
import uuid
from itertools import chain
import argparse
import sys
sys.path.append('./')
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
from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord
from torch import distributed as dist
from tqdm import tqdm

from transformers.trainer_pt_utils import IterableDatasetShard
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'


def video_load(video_path, 
               image_processor, 
               model_config):

    if os.path.exists(video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = round(vr.get_avg_fps())
        # sample_fps=args.sample_fps # LongVU used sample_fps as 0.5 here! use strictly 1
        frame_idx = [
                i
                for i in range(0, len(vr), round(fps / sample_fps))
            ]
        if len(frame_idx) > args.max_frames:
            frame_idx = [
                frame_idx[i]
                for i in range(0, len(frame_idx), len(frame_idx) // args.max_frames)
            ]
        video = vr.get_batch(frame_idx).asnumpy()
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, image_processor, model_config)
        video = [item.unsqueeze(0) for item in video]
    else:
        video = np.zeros((1, 1024, 1024, 3)).astype(np.uint8)
        image_sizes = [(1024, 1024)]
        video = process_images(video, image_processor, model_config)

    return video, image_sizes


class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        load_video_kwargs: dict,
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data_path = data_path

        data_list = {
            "count": ("json/4_count.json", f"video/4_count", "video"),
            "ego": ("json/3_ego.json", f"video/3_ego", "video"),
            "needle": ("json/2_needle.json", f"video/2_needle", "video"),
            "order": ("json/5_order.json", f"video/5_order", "video"),
            "plotQA": ("json/1_plotQA.json", f"video/1_plotQA", "video"),
            "anomaly_reco": (
                "json/6_anomaly_reco.json",
                f"video/6_anomaly_reco",
                "video",
            ),
            "topic_reasoning": (
                "json/7_topic_reasoning.json",
                f"video/7_topic_reasoning",
                "video",
            ),
        }

        list_data_dict = []
        for k, v in data_list.items():
            with open(os.path.join(data_path, v[0]), "r") as f:
                json_data = json.load(f)
            for data in json_data:
                question, answer = self.qa_template(data)
                list_data_dict.append(
                    {
                        "task_type": k,
                        "video": os.path.join(self.data_path, v[1], data["video"]),
                        "video_name": data["video"],
                        "question": data["question"],
                        "prompt": question,
                        "answer": answer,
                    }
                )

        # self.data = list_data_dict
        self.data = [{**list_data_dict[i], 'qid':i} for i in range(len(list_data_dict))]
        self.load_video_kwargs = load_video_kwargs

    def __len__(self) -> int:
        return len(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question += (
            "Respond with only the letter (A, B, C or D) of the correct option.\n"
        )
        question = question.rstrip()
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer


    def get_sample(self, line):

        video_path = os.path.join(
            self.data_path,
            line["video"],
        )

        video, image_sizes = video_load(video_path, 
               self.load_video_kwargs['image_processor'], 
               self.load_video_kwargs['model_config'])

        return line, video, image_sizes

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        for line in self.data:
            yield self.get_sample(line)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __getitem__(self, i):
        return self.data[i]


def train(args) -> None:
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    from ddp import init_distributed_mode
    init_distributed_mode(args)
    print('dist initialized...')
    
    version = args.version
    model_name = args.model_name
    model_path = args.model_path

    # torch.distributed.barrier()
    print('loading base model...')
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,  # pyre-fixme
        None,
        model_name,
        device_map=None,
    )

    if args.model_path2:
        from longvu.load_longvu import load_lora_weights
        model = load_lora_weights(args.model_path2, model)

    model.get_model().config.dino_threshold = 0.82
    model.get_model().config.drop_threshold = 0.77
    model.config.use_cache = True
    model.cuda()
    dataset = EvalDataset(
        data_path=args.data_path,
        load_video_kwargs=dict(image_processor=image_processor, 
                               model_config=model.config)
    )
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

    for batch in tqdm(shard_dataset):
        line, video, image_sizes = batch
        video_name = line["video_name"]
        answer = line["answer"]
        qs = line["prompt"]
        task_type = line["task_type"]
        # video_path = os.path.join(
        #     args.data_path,
        #     line["video"],
        # )

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[version].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        if "llama3" in version:
            input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,  
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        if pred.endswith(stop_str):
            pred = pred[: -len(stop_str)]
            pred = pred.strip()
        pred = pred.replace("Answer", "")

        letters = ["A", "B", "C", "D"]

        pred_answer = re.findall("[\(\ \[]*([A-D])[\)\.\ \]]*", pred)

        try:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip("()")
            if pred_answer in letters:
                pred_idx = letters.index(pred_answer)
                pred = letters[pred_idx]
        except Exception as e:
            print(e)
            print("[ERROR to fetch answer] ", "gt: ", line["answer"], " pred: ", pred)
            pred = 'FAILED'
        
        # else:
        #     print("[Error] pred_answer: ", pred_answer, " pred: ", pred, flush=True)
        #     pred_idx = 2
        #     pred = letters[pred_idx]
        #     # pred='failed'


        ans_id = uuid.uuid4()
        output.append(
            {
                "question": line["question"],
                "prompt": qs,
                "answer": answer,
                "pred": pred,
                "task_type": task_type,
                "answer_id": str(ans_id),
                "model_id": model_name,
                "video_name": video_name,
                "metadata": {},
                "qid": line['qid'],
            }
        )

    dist.barrier()
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output))
    global_rank = dist.get_rank()
    if global_rank == 0:
        # if os.path.exists(args.output_dir):
        #     shutil.rmtree(args.output_dir)
        # os.mkdir(args.output_dir)

        with open(
            os.path.join(args.output_dir, "outputs.json"),
            "w",
        ) as f:
            json.dump(all_output, f)

        correct = 0
        total = 0
        acc_dict = {}
        seen=[]
        for output in all_output:
            # remove duplicates due to ddp inference
            if output['qid'] in seen:
                continue
            seen.append(output['qid'])

            pred = output["pred"]
            gt = output["answer"]
            task_type = output["task_type"]
            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]
            acc_dict[task_type][1] += 1
            total += 1

            if pred == gt:
                acc_dict[task_type][0] += 1
                correct += 1

        assert total==total_eval_samples, f"actual number of eval samples {total_eval_samples} - inferenced samples {total}"

        final_res = dict()
        total = 0
        idx = 0
        for k, v in acc_dict.items():
            idx += 1
            final_res[k] = v[0] / v[1] * 100
            total += final_res[k]
        final_res["Acc"] = total / idx
        print(final_res, flush=True)

        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(final_res, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./checkpoints/longvu_qwen")
    parser.add_argument('--model_path2', help='', default=None, type=str, required=False)
    parser.add_argument('--model_name', default="cambrian_qwen")
    parser.add_argument('--version', default="qwen")
    parser.add_argument('--data_path', default="/datasets/video_llm/video_eval/MLVU/MLVU")
    parser.add_argument('--output_dir', help='where the results will be stored', required=True)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--sample_fps', default=0.5, type=float)
    parser.add_argument('--max_frames', default=1000, type=int)

    args = parser.parse_args()
    global sample_fps
    sample_fps=args.sample_fps

    if "llama3" in args.version:
        args.model_name = "cambrian_llama3"

    train(args)

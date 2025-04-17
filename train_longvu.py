import copy
import datetime
import json
import os
import pathlib
import uuid
import random
import numpy as np
import math
import torch
import transformers
import torch.nn.functional as F

from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Optional, Sequence
from decord import cpu, VideoReader
from longvu import conversation as conversation_lib
from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from longvu.language_model.cambrian_llama import CambrianLlamaForCausalLM
from longvu.language_model.cambrian_qwen import CambrianQwenForCausalLM
from longvu.mm_datautils import (
    preprocess,
    preprocess_multimodal,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from longvu.rrpo_trainer import RRPOTrainer
from PIL import Image, ImageSequence
from torch import distributed as dist
from torch.utils.data import Dataset
from transformers import TrainerCallback
from longvu.utils import prepare_multimodal_data_signed
from longvu.mm_datautils import preprocess_qwen, preprocess_llama3, preprocess_llama_3_1, preprocess_llama_3_2
from longvu.losses import *
import imageio
import cv2
os.environ["WANDB_PROJECT"] = "RRPO" # for wandb logging


def sanity_check_token_len(tokenizer):
    assert len(tokenizer('</MASK>', add_special_tokens=False).input_ids) == 1
    assert len(tokenizer(' </MASK>', add_special_tokens=False).input_ids) == 2
    assert len(tokenizer('</MASK> ', add_special_tokens=False).input_ids) == 2

    assert len(tokenizer('<MASK>', add_special_tokens=False).input_ids) == 1
    assert len(tokenizer(' <MASK>', add_special_tokens=False).input_ids) == 2
    assert len(tokenizer('<MASK> ', add_special_tokens=False).input_ids) == 2

def process_masked_tokens(input_ids, tokenizer):

    assert input_ids.shape[0]==1
    input_ids = list(input_ids[0].numpy())

    tag_start=[]
    tag_start.append(tokenizer(' <MASK>', add_special_tokens=False).input_ids) # expected to be 2 tokens
    tag_start.append(tokenizer('<MASK> ', add_special_tokens=False).input_ids) # expected to be 2 tokens

    tag_end=[]
    tag_end.append(tokenizer(' </MASK>', add_special_tokens=False).input_ids) # expected to be 2 tokens
    tag_end.append(tokenizer('</MASK> ', add_special_tokens=False).input_ids) # expected to be 2 tokens

    mask_token_start= tokenizer('<MASK>', add_special_tokens=False).input_ids # expected to be 1 token
    mask_token_end = tokenizer('</MASK>', add_special_tokens=False).input_ids # expected to be 1 token

    new_input_ids=[]
    # new_attention_mask=[]
    signs=[]
    k=0
    sign=0
    flag=False
    while k<len(input_ids)-1:

        # start mask
        if [input_ids[k], input_ids[k+1]] in tag_start:
            sign+=1
            flag=True
            k=k+2
            continue
        elif [input_ids[k]]==mask_token_start:
            sign+=1
            flag=True
            k=k+1
            continue

        # end mask
        if [input_ids[k], input_ids[k+1]] in tag_end:
            flag=False
            k=k+2
            continue
        elif [input_ids[k]]==mask_token_end:
            flag=False
            k=k+1
            continue

        if flag:
            signs.append(sign)
        else:
            signs.append(0)

        new_input_ids.append(input_ids[k])

        k=k+1

    # the last one
    if k<len(input_ids):
        new_input_ids.append(input_ids[k])

    if flag:
        signs.append(sign)
    else:
        signs.append(0)

    return new_input_ids, signs

def preprocess_masked(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    # noqa
    preprocess_fn=None
    if conversation_lib.default_conversation.version == "llama3":
        preprocess_fn=preprocess_llama3
    if conversation_lib.default_conversation.version == "llama3_1":
        preprocess_fn=preprocess_llama_3_1
    if conversation_lib.default_conversation.version == "llama3_2":
        preprocess_fn=preprocess_llama_3_2
    if conversation_lib.default_conversation.version == "qwen":
        preprocess_fn=preprocess_qwen

    assert preprocess_fn is not None, f"Unknow conversation version: {conversation_lib.default_conversation.version}"

    ## just to make sure, we are handling MASK tags correctly
    ref_sources = copy.deepcopy(sources)
    assert ref_sources[0][2]['from']=='gpt-ref'
    ref_sources[0][2]['from']='gpt'
    sources = [sources[0][:-1]] # remove the last one which is 'gpt-ref'
    ref_sources = [ref_sources[0][0], ref_sources[0][2]] # creating a ref copy with unmasked responses
    ref_sources=[ref_sources]
    
    # obtain ref_input_ids for sanity
    ref_dict = preprocess_fn(ref_sources, tokenizer, has_image=has_image)
    ref_input_ids = list(ref_dict['input_ids'][0].numpy())
    mask_dict = preprocess_fn(sources, tokenizer, has_image=has_image)
    input_ids = mask_dict['input_ids']
    input_ids, signs = process_masked_tokens(input_ids, tokenizer=tokenizer)

    if input_ids==ref_input_ids:
        pass
    else:
        if len(input_ids)==len(ref_input_ids):
            ## checking whether the signs of the corresponding diff_ids belongs to same seq. 
            ## then we can use that, but if their signs are mixed, we have to discard them
            diff_ids = np.array(ref_input_ids) - np.array(input_ids)
            signs = np.array(signs)
            diff_ids = (diff_ids!=0).astype(int)
            one_indices = np.where(diff_ids == 1)[0]
            consecutive_groups = np.split(one_indices, np.where(np.diff(one_indices) != 1)[0] + 1)
            accurate=True
            for group in consecutive_groups:
                if len(set(signs[group])) != 1:
                    accurate=False

            if not accurate:
                # ignoring them in loss
                print("[Skipping] unable to handle MASK tags properly")
                signs=[0]*len(ref_input_ids)

        else:
            # if we can not confirm -- ignore that sample
            print("[Skipping] unable to handle MASK tags properly")
            signs=[0]*len(ref_input_ids)

    signs=torch.tensor(np.array(signs), dtype=torch.int64).unsqueeze(0)
    return dict(**ref_dict, signs=signs) # must use the ids got without mask tag
    
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for ViSA fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int
    image_aux_token_len_list: list  # pyre-fixme
    image_position: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        pos_instances = [dict(input_ids=instance["input_ids"], 
                              labels=instance["labels"], 
                              signs=instance["signs"], 
                              image_aux_list=instance["image_aux_list"], 
                              image_size=instance["image_size"], 
                              ) \
                         for instance in instances]
        neg_instances = [dict(input_ids=instance["neg_input_ids"], 
                              labels=instance["neg_labels"], 
                              signs=instance["neg_signs"], 
                              image_aux_list=instance["neg_image_aux_list"], 
                              image_size=instance["neg_image_size"], 
                              ) \
                         for instance in instances]
       
        pos_batch = self.__subcall__(pos_instances)
        neg_batch = self.__subcall__(neg_instances)

        return dict(**pos_batch,
                    neg_input_ids=neg_batch["input_ids"],
                    neg_labels=neg_batch["labels"],
                    neg_signs=neg_batch["signs"],
                    neg_attention_mask=neg_batch["attention_mask"],
                    neg_position_ids=neg_batch["position_ids"],
                    neg_image_aux_attention_masks_list=neg_batch['image_aux_attention_masks_list'],
                    neg_image_sizes=neg_batch['image_sizes'],
                    neg_images=neg_batch['images'],
                    )
    
    def __subcall__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:  # pyre-fixme

        image_token_len = self.image_token_len
        image_aux_token_len_list = self.image_aux_token_len_list
        image_position = self.image_position

        input_ids, labels, signs = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "signs")
        )
        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side

        # print_rank0("Pad token id is", self.tokenizer.pad_token_id)

        if padding_side == "left":
            input_ids = [(t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), "constant", self.tokenizer.pad_token_id,))for t in input_ids]
            labels = [(t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), "constant", IGNORE_INDEX)) for t in labels]
            signs = [(t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), "constant", IGNORE_INDEX)) for t in signs]
        else:
            input_ids = [(t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), "constant", self.tokenizer.pad_token_id,)) for t in input_ids]
            labels = [(t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad( t, (0, max_length - t.shape[0]), "constant", IGNORE_INDEX)) for t in labels]
            signs = [(t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad( t, (0, max_length - t.shape[0]), "constant", IGNORE_INDEX)) for t in signs]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        signs = torch.stack(signs)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # pyre-fixme
        # insert dummy image
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_ids_tmp = input_ids[i].clone()
                cur_input_ids_tmp[image_position + 1 :] = input_ids[
                    i, image_position:-1
                ]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp

                cur_labels_tmp = labels[i].clone()
                cur_labels_tmp[image_position + 1 :] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp

                cur_attention_mask_tmp = attention_mask[i].clone()
                cur_attention_mask_tmp[image_position + 1 :] = attention_mask[
                    i, image_position:-1
                ]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp

                cur_signs_tmp = signs[i].clone()
                cur_signs_tmp[image_position + 1 :] = signs[i, image_position:-1]
                cur_signs_tmp[image_position] = IGNORE_INDEX
                signs[i] = cur_signs_tmp


        image_sizes = [instance["image_size"] for instance in instances]
        (
            new_input_ids,
            new_labels,
            new_signs,
            new_attention_mask,
            new_position_ids,
            im_aux_attention_masks_list,
        ) = prepare_multimodal_data_signed(
            input_ids,
            labels,
            signs,
            attention_mask,
            image_sizes,
            image_token_len,
            image_aux_token_len_list,
            max_length,
        )
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            signs=new_signs,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            image_aux_attention_masks_list=im_aux_attention_masks_list,
        )

        batch["image_sizes"] = image_sizes
        if "image_aux_list" in instances[0]:
            image_aux_list = [instance["image_aux_list"] for instance in instances]
            image_aux_list = [
                list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)
            ]
            if all(
                x is not None and x.shape == image_aux_list[0][0].shape
                for x in image_aux_list[0]
            ):
                batch["images"] = [
                    torch.stack(image_aux) for image_aux in image_aux_list
                ]
            else:
                batch["images"] = image_aux_list

        return batch

class LazySupervisedDataset(Dataset):
    """Dataset for ViSA."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: None):
        super(LazySupervisedDataset, self).__init__()

        self.data_args = data_args
        data_path=data_args.data_path
        data_dict=json.load(open(data_path))
        print(f'total available samples: {len(data_dict)}')

        random.seed(42)
        random.shuffle(data_dict)
        list_data_dict, neg_list_data_dict = self.prepare_data_dict(data_dict)
        assert len(list_data_dict)==len(neg_list_data_dict)
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.neg_list_data_dict = neg_list_data_dict
        self.max_frames = data_args.max_frames

    def prepare_data_dict(self, data):
        pos_data=[]
        neg_data=[]
        
        for sample in data:

            conversation=[]
            conversation.append({'from': 'human', 
                                'value': sample['question'],
                                })
            conversation.append({'from': 'gpt', 
                                'value': sample['answer_masked']})
            conversation.append({'from': 'gpt-ref', # for verification purpose
                                'value': sample['answer']})
            
            pos_data.append({'conversations': conversation,
                            'id': sample['sample_id'],
                            'video': sample['media_path'],
                            })
            
            neg_conversation=[]
            neg_conversation.append({'from': 'human', 
                                'value': sample['question'],
                                })
            neg_conversation.append({'from': 'gpt', 
                                'value': sample['pred_masked']})
            neg_conversation.append({'from': 'gpt-ref', # for verification purpose
                                'value': sample['pred']})
            
            neg_data.append({'conversations': neg_conversation,
                            'id': sample['sample_id'],
                            'video': sample['media_path'],
                            })
            
        return pos_data, neg_data


    def __len__(self):
        return len(self.list_data_dict)

    def _has_image(self, sample: dict) -> bool:  # pyre-fixme
        if "image" in sample and not str(sample["image"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        if "video" in sample and not str(sample["video"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        return False
    
    def __pos_neg_getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        neg_sources = self.neg_list_data_dict[i]
        dat = sources
        assert sources['id']==neg_sources['id']
        if isinstance(i, int):
            sources = [sources]
            neg_sources = [neg_sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        assert len(neg_sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        has_image = self._has_image(dat)
        self.vid_or_image = None
        if has_image:
            if "image" in dat:
                self.vid_or_image='image'
                image_file = dat["image"]
                image_folder = self.data_args.image_folder
                processor_aux_list = self.data_args.image_processor_aux_list
                try:
                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
                except:
                    print(
                        "Not exist: ",
                        os.path.join(image_folder, image_file),
                        flush=True,
                    )
                    return self.__pos_neg_getitem__((i+1)% len(self))
                image_size = image.size
            else:
                self.vid_or_image='video'
                video_file = dat["video"]
                processor_aux_list = self.data_args.image_processor_aux_list
                video_file = os.path.join(self.data_args.image_folder, video_file)
                if os.path.exists(video_file):
                    try:
                        num_sample = self.max_frames 
                        
                        if video_file.endswith(".gif"):
                            gif = imageio.get_reader(video_file)
                            frame_idx = [i for i in range(0, len(gif))]
                            if len(frame_idx)>num_sample:
                                interval = len(frame_idx) / float(num_sample)
                                frame_idx = [int(interval * i) for i in range(num_sample)]

                            images_group = list()
                            for index, frame in enumerate(gif): 
                                if index in frame_idx:
                                    img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                                    images_group.append(img)

                            if self.data_args.force_sample:
                                images_group=[images_group[i] for i in frame_idx]
                            
                            image=np.stack(images_group)
                            image_size = image[0].shape[:2]
                  
                        else:
                            vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                            if self.data_args.force_sample:
                                # load fixed number of frames
                                fps = float(vr.get_avg_fps())
                                frame_idx = [i for i in range(0, len(vr))]
                                interval = len(frame_idx) / float(num_sample)
                                frame_idx = [int(interval * i) for i in range(num_sample)]
                            else:
                                sample_fps = round(
                                    vr.get_avg_fps() / self.data_args.video_fps
                                )
                                frame_idx = [i for i in range(0, len(vr), sample_fps)]
                                if len(frame_idx)>num_sample:
                                    interval = len(frame_idx) / float(num_sample)
                                    frame_idx = [int(interval * i) for i in range(num_sample)]
                            image = vr.get_batch(frame_idx).asnumpy()
                            image_size = image[0].shape[:2]

                    except:
                        print("fail to load video: ", video_file, flush=True)
                        return self.__pos_neg_getitem__((i+1)% len(self))
                else:
                    print("Not exist: ", video_file, flush=True)
                    return self.__pos_neg_getitem__((i+1)% len(self))

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    # result.paste(pil_img, (0, 0))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    # result.paste(pil_img, (0, 0))
                    return result

            if self.data_args.image_aspect_ratio != "pad":
                raise NotImplementedError("Only pad is supported for now.")

            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                if not isinstance(image_aux, Image.Image):
                    frame_list = []
                    for frame in image_aux:
                        if not isinstance(frame, Image.Image):
                            frame = Image.fromarray(frame)
                        frame_aux = expand2square(
                            frame, tuple(int(x * 255) for x in processor_aux.image_mean)
                        ).resize((target_resolution, target_resolution))
                        frame_aux = processor_aux.preprocess(
                            frame_aux, return_tensors="pt"
                        )["pixel_values"][0]
                        frame_list.append(frame_aux)
                    image_aux = torch.stack(frame_list)
                else:
                    image_aux = expand2square(
                        image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                    image_aux = processor_aux.preprocess(
                        image_aux, return_tensors="pt"
                    )["pixel_values"][0]
                image_aux_list.append(image_aux)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
            neg_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in neg_sources]), self.data_args
            )

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
            neg_sources = copy.deepcopy([e["conversations"] for e in neg_sources])

        # add signs
        data_dict = preprocess_masked(sources, self.tokenizer, has_image=has_image)
        neg_data_dict = preprocess_masked(neg_sources, self.tokenizer, has_image=has_image)

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0],
                signs=data_dict["signs"][0],
            )
            neg_data_dict = dict(
                input_ids=neg_data_dict["input_ids"][0], labels=neg_data_dict["labels"][0],
                signs=neg_data_dict["signs"][0],
            )

        if (data_dict["labels"] != IGNORE_INDEX).sum() == 0:
            return self.__pos_neg_getitem__((i+1)% len(self))
        if (neg_data_dict["labels"] != IGNORE_INDEX).sum() == 0:
            return self.__pos_neg_getitem__((i+1)% len(self))
        
        if sum(data_dict["signs"])==0 or sum(neg_data_dict["signs"])==0:
            print("[Skipping] unable to handle MASK tags properly")
            return self.__pos_neg_getitem__((i+1)% len(self))

        # image exist in the data
        if has_image:
            data_dict["image_aux_list"] = image_aux_list  # pyre-fixme
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            image_list = []
            for processor_aux in processor_aux_list:
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                image_list.append(
                    torch.zeros(
                        3,
                        target_resolution,
                        target_resolution,
                    )
                )
            data_dict["image_aux_list"] = image_list
            image_size = (crop_size, crop_size)
        data_dict["image_size"] = image_size

        # sample_id=self.list_data_dict[i]['id']
        return dict(**data_dict, 
                    neg_input_ids=neg_data_dict["input_ids"],
                    neg_labels=neg_data_dict["labels"],
                    neg_signs=neg_data_dict["signs"],
                    # sample_id=sample_id
                    )
    
    def __getitem__(self, i):
        pos_neg_dict=self.__pos_neg_getitem__(i)
        pos_neg_dict['neg_image_aux_list'] = copy.deepcopy(pos_neg_dict['image_aux_list'])
        pos_neg_dict['neg_image_size']=copy.deepcopy(pos_neg_dict['image_size'])

        return pos_neg_dict

def make_visa_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_args=data_args,
                                )
    
    data_collator_kwargs = {
        "tokenizer": tokenizer,
    }

    if hasattr(data_args, "image_token_len"):
        data_collator_kwargs["image_token_len"] = data_args.image_token_len

    if hasattr(data_args, "vision_tower_aux_token_len_list"):
        data_collator_kwargs["image_aux_token_len_list"] = (
            data_args.vision_tower_aux_token_len_list
        )
    else:
        data_collator_kwargs["image_aux_token_len_list"] = [data_args.image_token_len]

    if hasattr(data_args, "image_position"):
        data_collator_kwargs["image_position"] = data_args.image_position


    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)  # pyre-fixme

    return dict(
            train_dataset=train_dataset, 
            eval_dataset=None, 
            data_collator=data_collator
            )

@dataclass
class ModelArguments:
    base_model_name: Optional[str] = field(default=None)
    input_model_filename: Optional[str] = field(default=None)
    output_model_filename: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    grid_size: Optional[int] = field(default=8)
    vision_tower_type: Optional[str] = field(default="sam")
    mm_hidden_size: Optional[int] = field(default=256)

    # cambrian
    vision_tower_aux_list: Optional[str] = field(
        default='["siglip/CLIP-ViT-SO400M-14-384", "facebook/dinov2-giant-res378"]'
    )
    vision_tower_aux_token_len_list: Optional[str] = field(default="[576, 576]")
    image_token_len: Optional[int] = field(default=576)
    num_query_group: Optional[int] = field(default=1)
    query_num_list: Optional[str] = field(default="[576]")
    connector_depth: Optional[int] = field(default=3)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    num_of_vision_sampler_layers: Optional[int] = field(default=10)
    start_of_vision_sampler_layers: Optional[int] = field(default=0)
    stride_of_vision_sampler_layers: Optional[int] = field(default=3)

    is_st_sampler: bool = field(default=False)
    highres_connect: bool = field(default=False)
    highres: bool = field(default=False)
    connect_layer: Optional[int] = field(default=2)
    lowres_token: Optional[int] = field(default=8)
    dino_threshold: float = field(default=0.83)
    drop_threshold: float = field(default=0.8)
    frame_pos: bool = field(default=False)
    is_image_newline: bool = field(default=True)


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_position: Optional[int] = field(default=91)
    image_folder: Optional[str] = field(default=None)
    uniform_sample: bool = field(default=False)
    image_aspect_ratio: str = "square"
    num_points: int = field(default=0)
    video_fps: float = field(default=1)
    use_subtitle: bool = field(default=True)
    max_frames: int = field(default=100)
    force_sample: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_text_decoder: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    unfreeze_mm_image_decoder: bool = field(default=False)

    mm_vision_sampler_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    model_max_length: Optional[int] = field(default=8192)

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    # mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    loss_alpha: Optional[float] = 0.5
    loss_beta: Optional[float] = 0.1


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


def setup_model(data_args, model_args, training_args):

    # model_name='cambrian_qwen'
    model_name=model_args.base_model_name
    # assert model_name in ['cambrian_qwen', 'cambrian_llama3_2', 'cambrian_llama3']

    bnb_model_from_pretrained_args = {}

    if model_args.vision_tower_aux_list is not None:
        if "cambrian" in model_name:
            if "qwen" in model_name:
                model = CambrianQwenForCausalLM.from_pretrained(
                    model_args.input_model_filename,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args,
                )
            else:
                model = CambrianLlamaForCausalLM.from_pretrained(
                    model_args.input_model_filename,
                    **bnb_model_from_pretrained_args,
                )
        else:
            raise NotImplementedError(
                f"{model_args.model_name_or_path} is not supported yet"
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.input_model_filename,
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.input_model_filename,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if (
        model_args.version == "llama3"
        or model_args.version == "llama3_1"
        or model_args.version == "llama3_2"
    ):
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif model_args.version == "qwen":
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        raise NotImplementedError(model_args.version)
    
    print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    if model_args.vision_tower_aux_list is not None:
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        model_args.vision_tower_aux_token_len_list = json.loads(
            model_args.vision_tower_aux_token_len_list
        )
        model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=None,  # FSDP or not, flag should be the same as None to avoid creation error
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()

        if not training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(
                        dtype=torch.bfloat16, device=training_args.device
                    )
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [  
                vision_tower_aux.image_processor
                for vision_tower_aux in vision_tower_aux_list
            ]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio  
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position  
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end  
        data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token  

        
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            tune_modules = [
                "mm_projector",
                "pos_emb",
                "vision_sampler",
                "vision_sampler_layers",
                "vision_query",
                "image_newline",
            ]
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    param.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter  
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.image_token_len = model_args.image_token_len = (  
            model_args.image_token_len
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr  
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr  
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr  
        training_args.use_im_start_end = model_args.mm_use_im_start_end  
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = (
            data_args.vision_tower_aux_token_len_list
        ) = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = model_args.image_token_len
        model.config.is_st_sampler = model_args.is_st_sampler  
        data_args.image_token_len = model_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)


    model.to(torch.bfloat16)

    def convert_bn_to_float(model):
        if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
            return model.float()
        for child_name, child in model.named_children():
            model.add_module(child_name, convert_bn_to_float(child))
        return model

    model = convert_bn_to_float(model)

    return model, tokenizer, data_args, model_args, training_args


def find_all_linear_names_exact(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 
                           'vision_tower', 
                           'vision_resampler', 
                           'vision_sampler', 
                           'lm_head',]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    print('LORA selection')
    print(sorted(list(lora_module_names)))

    return list(lora_module_names)


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global_rank = get_global_rank()
    local_rank = get_local_rank()

    torch.distributed.barrier()

    # make a copy to load ref model; safe coding
    ref_model_args=copy.deepcopy(model_args)
    ref_data_args=copy.deepcopy(data_args)
    ref_training_args=copy.deepcopy(training_args)

    print('Loading online model')
    model, tokenizer, data_args, model_args, training_args = \
        setup_model(data_args, model_args, training_args)

    special_tokens = {'additional_special_tokens': ['<MASK>', '</MASK>']}
    tokenizer.add_special_tokens(special_tokens)
    sanity_check_token_len(tokenizer)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names_exact(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.to(torch.bfloat16)

    # enable setup to train other modules if required
    # control grad if required 
    print("params with grad on...")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)

    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(
        p.numel() for p in model.get_model().parameters() if p.requires_grad
    )
    print(f'# params: trainable/toral: {round(trainable_params/10**6)} M/{round(total_params/10**9, 4)} B')

    training_args.output_dir = model_args.output_model_filename
    model_args.local_dir = model_args.output_model_filename

    os.environ[f"FSDP_USE_ORIG_PARAMS"] = "true"
    training_args.fsdp_config["use_orig_params"] = True

    print('Loading reference model')
    ref_training_args.lora_enable = False
    ref_model = setup_model(ref_data_args, ref_model_args, ref_training_args)[0]
    
    for n,p in ref_model.named_parameters():
        p.requires_grad = False

    data_module = make_visa_data_module(tokenizer=tokenizer, data_args=data_args)
    loss_fn = RRPO(alpha=training_args.loss_alpha, 
                    beta=training_args.loss_beta, 
                    )
    
    trainer = RRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        ref_model=ref_model,
        loss_fn=loss_fn,
        callbacks=[],
        **data_module,
    )

    if trainer.is_fsdp_enabled:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            StateDictType,
            BackwardPrefetch,
            ShardingStrategy,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import functools
        
        def get_block_class_from_model(model: torch.nn.Module, block_class_name: str) -> torch.nn.Module:
            """Get the class of a block from a model, using the block's class name."""
            for module in model.modules():
                if module.__class__.__name__ == block_class_name:
                    return module.__class__
            raise ValueError(f"Could not find block class {block_class_name} in model {model}")

        print(('sharding reference model'))
        wrap_class = get_block_class_from_model(trainer.ref_model, 
                                                trainer.args.fsdp_transformer_layer_cls_to_wrap)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=trainer.args.local_rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )
        trainer.ref_model = FSDP(trainer.ref_model, **shared_fsdp_kwargs)
    else:
        trainer.ref_model.to(trainer.args.device)
        trainer.ref_model.eval()

    ## no resume
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()
    trainer.save_state() 

    print('removing files wont be required...')
    torch.distributed.barrier()
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        ckptdir=list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        ckptdir=ckptdir[0]
        delete_files=('pytorch_model_fsdp.bin', 'optimizer.bin', 'scheduler.pt', 
                'rng_state_0.pth', 'rng_state_1.pth', 'rng_state_2.pth', 'rng_state_3.pth', 
                )

        import glob
        files=glob.glob(os.path.join(ckptdir, '*'))
        for file in files:
            if file.endswith(delete_files):
                os.remove(file)
                print("[Removed] ", file)


if __name__ == "__main__":
    train()

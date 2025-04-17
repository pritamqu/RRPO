import logging
import os
import json
import random
import torch
import re
import numpy as np
from dataset.it_dataset_mistral import ITVidTrainDataset_mistral, ITImgTrainDataset_mistral
from dataset.video_utils import VIDEO_READER_FUNCS
from torchvision import transforms
from torchvision.transforms import InterpolationMode
logger = logging.getLogger(__name__)

def setup_transformation(resolution, 
                         input_mean=[0.48145466, 0.4578275, 0.40821073], 
                         input_std=[0.26862954, 0.26130258, 0.27577711],
                         ):

    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
    normalize = transforms.Normalize(input_mean, input_std)

    transformation = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )

    return transformation

class SupervisedDataset(ITVidTrainDataset_mistral):
    media_type = "video"

    def __init__(
        self, ann_file, transform, num_frames=4, 
        video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", start_token="<Video>", end_token="</Video>",
        add_second_msg=False,
        random_shuffle=True,
        return_question_instruction=False, # if True, return instruction with instruciton
        dynamic_config=None, # config for dynamic resolution finetuning
    ):
        super().__init__(
            ann_file, transform, 
            system=system,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
            return_question_instruction=return_question_instruction,
            dynamic_config=dynamic_config,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            msg = ""
            clip = None
            if "start" in ann and "end" in ann:
                clip = [ann["start"], ann["end"]]

            video, index, sec = self.load_and_transform_media_data_video(
                index, ann["image"], return_fps=True, clip=clip,
                dynamic_config=self.dynamic_config
            )
            if self.add_second_msg and sec is not None:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            conversation, instruction = self.process_qa(ann["qa"], msg)
            return video, conversation, instruction, index
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)
        
    def __len__(self):
        return len(self.anno)

# dataset with dynamic sign tensors
class RRPODataset(ITVidTrainDataset_mistral):
    media_type = "video"

    def __init__(
        self, ann_file, transform, num_frames=4, 
        video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", start_token="<Video>", end_token="</Video>",
        add_second_msg=False,
        random_shuffle=True, # this shuffles the conversation orders not across samples
        return_question_instruction=False, # if True, return instruction with instruciton
        dynamic_config=None, # config for dynamic resolution finetuning
        tokenizer=None, 
    ):
        super().__init__(
            ann_file, transform, 
            system=system,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
            return_question_instruction=return_question_instruction,
            dynamic_config=dynamic_config,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

        self.tokenizer=tokenizer

    def get_anno(self, index, dict_key="QA"):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index][dict_key]
        sample_id = self.anno[index]["sample_id"]

        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa,
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa, "sample_id": sample_id}
        return anno
    
        
    def __getitem__(self, index):
        try:
            pos = self.get_anno(index, "QA")
            neg = self.get_anno(index, "QA_NEG")

            sample_id = pos["sample_id"]
            video_path = pos["image"]

            # load video once
            msg = ""
            clip = None
            video, index, sec = self.load_and_transform_media_data_video(
                index, video_path, return_fps=True, clip=clip,
                dynamic_config=self.dynamic_config
            )
            if self.add_second_msg and sec is not None:
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "

            pos_conversation, pos_instruction = self.process_qa(pos["qa"], msg)
            neg_conversation, _ = self.process_qa(neg["qa"], msg)
            instruction=pos_instruction
            
            return (video, 
                    pos_conversation,
                    neg_conversation,
                    instruction, 
                    index)
        
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {video_path}")
            index = (index+1) % len(self)
            return self.__getitem__(index)

    def __len__(self):
        return len(self.anno)
    



import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy

anno_root_it = "/datasets/video_llm/VideoChat2-IT"

# ============== pretraining datasets=================
available_corpus = dict(
    # image
    caption_coco=[
        f"{anno_root_it}/image/caption/coco/train.json", 
        "your_data_path/coco_caption",
    ],
    caption_coco_100k=[
        f"{anno_root_it}/image/caption/coco/train_100k.json", 
        "your_data_path/coco_caption",
    ],
    caption_llava=[
        f"{anno_root_it}/image/caption/llava/train.json", 
        "your_data_path/coco_caption",
    ],
    caption_minigpt4=[
        f"{anno_root_it}/image/caption/minigpt4/train.json", 
        "your_data_path/minigpt4/image",
    ],
    caption_paragraph_captioning=[
        f"{anno_root_it}/image/caption/paragraph_captioning/train.json", 
        "your_data_path/m3it/image-paragraph-captioning",
    ],
    caption_textcaps=[
        f"{anno_root_it}/image/caption/textcaps/train.json", 
        "your_data_path/m3it/textcap",
    ],
    classification_imagenet=[
        f"{anno_root_it}/image/classification/imagenet/train.json", 
        "your_data_path/m3it/imagenet",
    ],
    classification_coco_itm=[
        f"{anno_root_it}/image/classification/coco_itm/train.json", 
        "your_data_path/m3it/coco-itm",
    ],
    conversation_llava=[
        f"{anno_root_it}/image/conversation/llava/train.json", 
        "your_data_path/coco_caption",
    ],
    reasoning_clevr=[
        f"{anno_root_it}/image/reasoning/clevr/train.json", 
        "your_data_path/m3it/clevr",
    ],
    reasoning_visual_mrc=[
        f"{anno_root_it}/image/reasoning/visual_mrc/train.json", 
        "your_data_path/m3it/visual-mrc",
    ],
    reasoning_llava=[
        f"{anno_root_it}/image/reasoning/llava/train.json", 
        "your_data_path/coco_caption",
    ],
    vqa_vqav2=[
        f"{anno_root_it}/image/vqa/vqav2/train.json", 
        "your_data_path/m3it/vqa-v2",
    ],
    vqa_gqa=[
        f"{anno_root_it}/image/vqa/gqa/train.json", 
        "your_data_path/m3it/gqa",
    ],
    vqa_okvqa=[
        f"{anno_root_it}/image/vqa/okvqa/train.json", 
        "your_data_path/m3it/okvqa",
    ],
    vqa_a_okvqa=[
        f"{anno_root_it}/image/vqa/a_okvqa/train.json", 
        "your_data_path/m3it/a-okvqa",
    ],
    vqa_viquae=[
        f"{anno_root_it}/image/vqa/viquae/train.json", 
        "your_data_path/m3it/viquae",
    ],
    vqa_ocr_vqa=[
        f"{anno_root_it}/image/vqa/ocr_vqa/train.json", 
        "your_data_path/m3it/ocr-vqa",
    ],
    vqa_text_vqa=[
        f"{anno_root_it}/image/vqa/text_vqa/train.json", 
        "your_data_path/m3it/text-vqa",
    ],
    vqa_st_vqa=[
        f"{anno_root_it}/image/vqa/st_vqa/train.json", 
        "your_data_path/m3it/st-vqa",
    ],
    vqa_docvqa=[
        f"{anno_root_it}/image/vqa/docvqa/train.json", 
        "your_data_path/m3it/docvqa",
    ],
    # new image instruction
    reasoning_science_qa=[
        f"{anno_root_it}/image/reasoning/science_qa/train.json", 
        "your_data_path/m3it/science-qa",
    ],
    vqa_infovqa=[
        f"{anno_root_it}/image/vqa/infovqa/train_gpt.json", 
        "your_data_path/ocr_data/InfoVQA/infographicVQA_train_v1.0_images",
    ],
    vqa_ai2d=[
        f"{anno_root_it}/image/vqa/ai2d/train.json", 
        "your_data_path/ai2diagram/ai2d/images",
    ],
    vqa_chart_qa=[
        f"{anno_root_it}/image/vqa/chart_qa/train.json", 
        "your_data_path/chartqa/ChartQA Dataset/train/png",
    ],
    vqa_dvqa_80k=[
        f"{anno_root_it}/image/vqa/dvqa/train_80k.json", 
        "your_data_path/DVQA/images",
    ],
    grounding_coco=[
        f"{anno_root_it}/image/grounding/coco/train.json", 
        "your_data_path/videollava/llava_image_tune/coco",
    ],
    grounding_vg=[
        f"{anno_root_it}/image/grounding/vg/train.json", 
        "your_data_path/videollava/llava_image_tune/vg",
    ],
    conversation_lvis_instruct4v=[
        f"{anno_root_it}/image/conversation/lvis_instruct4v/train.json", 
        "your_data_path",
    ],
    caption_sharegpt4v_420k=[
        f"{anno_root_it}/image/caption/sharegpt4v/train_420k.json", 
        "your_data_path/sharegpt4v/data",
    ],
    # video
    caption_textvr=[ # 398x224, 3 fps
        f"{anno_root_it}/video/caption/textvr/train.json", 
        "/datasets/video_llm/video_instruction_tuning/textvr/TextVR_data/Video_subset_1fps",
        "video"
    ],
    caption_videochat=[ # 398x224, 1 fps
        f"{anno_root_it}/video/caption/videochat/train.json", 
        "/datasets/video_llm/video_instruction_tuning/webvid_subset_1fps_224/videos",
        "video"
    ],
    caption_webvid=[ # 398x224, 1 fps
        f"{anno_root_it}/video/caption/webvid/train.json", 
        "/datasets/video_llm/video_instruction_tuning/webvid_subset_1fps_224/videos",
        "video"
    ],
    caption_webvid_80k=[ # 398x224, 1 fps
        f"{anno_root_it}/video/caption/webvid/train_80k.json", 
        "/datasets/video_llm/video_instruction_tuning/webvid_subset_1fps_224/videos",
        "video"
    ],
    caption_youcook2=[ # 1280x720, 24 fps
        f"{anno_root_it}/video/caption/youcook2/train.json", 
        "/datasets/video_llm/video_instruction_tuning/youcook2/split_videos_subset_1fps_256",
        "video"
    ],
    caption_smit=[ # 1280x720, 30 fps
        f"{anno_root_it}/video/caption/s_mit/train.json", 
        "/datasets/video_llm/video_instruction_tuning/s_mit/smit_subset_1fps_256",
        "video"
    ],
    caption_smit_40k=[
        f"{anno_root_it}/video/caption/s_mit/train_40k.json", 
        "/datasets/video_llm/video_instruction_tuning/s_mit/smit_subset_1fps_256",
        "video"
    ],
    classification_k710=[ # 456x256, 30 fps
        f"{anno_root_it}/video/classification/k710/train.json", 
        "/datasets/video_llm/video_instruction_tuning/kinetics_subset_1fps_256",
        "video"
    ],
    classification_ssv2=[ # 320x240, 12 fps
        f"{anno_root_it}/video/classification/ssv2/train.json", 
        "/datasets/video_llm/video_instruction_tuning/ssv2_subset_1fps_240",
        "video"
    ],
    conversation_videochat1=[ # 398x224, 1 fps
        f"{anno_root_it}/video/conversation/videochat1/train.json", 
        "/datasets/video_llm/video_instruction_tuning/webvid_subset_1fps_224/videos",
        "video"
    ],
    conversation_videochat2=[ # 1280x720, 30 fps
        f"{anno_root_it}/video/conversation/videochat2/train.json", 
        "/datasets/video_llm/video_instruction_tuning/videochat/videos_subset_1fps_256",
        "video"
    ],
    conversation_videochatgpt=[ # 320x240, 1 fps
        f"{anno_root_it}/video/conversation/videochatgpt/train.json", 
        "/datasets/video_llm/ActivityNet/train_val_1fps",
        "video"
    ],
    reasoning_next_qa=[ # 480x360, 24 fps
        f"{anno_root_it}/video/reasoning/next_qa/train.json", 
        "/datasets/video_llm/video_instruction_tuning/NextQA/nextqa_subset_1fps",
        "video"
    ],
    reasoning_clevrer_qa=[ # 480x320, 25 fps
        f"{anno_root_it}/video/reasoning/clevrer_qa/train.json", 
        "/datasets/video_llm/video_instruction_tuning/clevrer/video_train_subset_1fps",
        "video"
    ],
    reasoning_clevrer_mc=[ # 480x320, 25 fps
        f"{anno_root_it}/video/reasoning/clevrer_mc/train.json",  
        "/datasets/video_llm/video_instruction_tuning/clevrer/video_train_subset_1fps",
        "video"
    ],
    vqa_ego_qa=[ # 568x320, 30 fps
        f"{anno_root_it}/video/vqa/ego_qa/train.json", 
        "/datasets/video_llm/video_instruction_tuning/egoqa/split_videos_subset_1fps_256",
        "video"
    ],
    vqa_tgif_frame_qa=[ # 245x245, 10 fps
        f"{anno_root_it}/video/vqa/tgif_frame_qa/train.json", 
        "/datasets/video_llm/video_instruction_tuning/tgif/gifs_subset_1fps",
        "video"
    ],
    vqa_tgif_transition_qa=[ # 245x245, 10 fps
        f"{anno_root_it}/video/vqa/tgif_transition_qa/train.json", 
        "/datasets/video_llm/video_instruction_tuning/tgif/gifs_subset_1fps",
        "video"
    ],
    vqa_webvid_qa=[ # 398x224, 1 fps
        f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        "/datasets/video_llm/video_instruction_tuning/webvid_subset_1fps_224/videos",
        "video"
    ],
    vqa_webvid_qa_30k=[ # 398x224, 1 fps
        f"{anno_root_it}/video/vqa/webvid_qa/train_30k.json", 
        "/datasets/video_llm/video_instruction_tuning/webvid_subset_1fps_224/videos",
        "video",
    ],
    # new video instruction
    caption_sharegptvideo_300k=[
        f"{anno_root_it}/video/caption/sharegptvideo/train_300k.json", 
        "your_data_path/LLaVA_DPO/train_300k",
        "video",
        "img", # read from image
    ],
    vqa_sharegptvideo_240k=[
        f"{anno_root_it}/video/vqa/sharegptvideo/train_240k.json", 
        "your_data_path/LLaVA_DPO/train_300k",
        "video",
        "img", # read from image
    ],
    caption_vidln_kinetics=[
        f"{anno_root_it}/video/caption/vidln/kinetics_train.json", 
        "",
        "video",
    ],
    caption_vidln_oops=[
        f"{anno_root_it}/video/caption/vidln/oops_train.json", 
        "your_data_path/oops/oops_video/train",
        "video",
    ],
    caption_vidln_ovis=[
        f"{anno_root_it}/video/caption/vidln/ovis_train.json", 
        "your_data_path/ovis/train",
        "video",
        "img", # read from image
    ],
    caption_vidln_uvo_sparse=[
        f"{anno_root_it}/video/caption/vidln/uvo_sparse_train.json", 
        "your_data_path/UVO/uvo_videos_sparse",
        "video",
    ],
    caption_vidln_uvo_dense=[
        f"{anno_root_it}/video/caption/vidln/uvo_dense_train.json", 
        "your_data_path/UVO/uvo_videos_dense",
        "video",
    ],
    caption_favd=[
        f"{anno_root_it}/video/caption/favd/train.json", 
        "your_data_path/favd",
        "video",
    ],
    grounding_didemo=[
        f"{anno_root_it}/video/grounding/didemo/train.json", 
        "your_data_path/DiDeMo",
        "video",
    ],
    # text
    conversation_sharegpt=[
        f"{anno_root_it}/text/sharegpt/train.json", 
        "",
        "text",
    ],
)


available_corpus["videochat2_instruction"] = [
    available_corpus["caption_coco"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid"],
    available_corpus["caption_youcook2"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
]


# add smit
available_corpus["videochat2_instruction_new"] = [
    available_corpus["caption_coco_100k"],
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["classification_imagenet"],
    available_corpus["classification_coco_itm"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"],
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_webvid_80k"],
    available_corpus["caption_youcook2"],
    available_corpus["caption_smit"],
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_frame_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa"],
]


# add more high-quality data
available_corpus["videochat2_instruction_hd"] = [
    # image
    available_corpus["caption_llava"],
    available_corpus["caption_minigpt4"],
    available_corpus["caption_paragraph_captioning"],
    available_corpus["caption_textcaps"],
    available_corpus["conversation_llava"],
    available_corpus["reasoning_clevr"],
    available_corpus["reasoning_visual_mrc"],
    available_corpus["reasoning_llava"],
    available_corpus["vqa_vqav2"],
    available_corpus["vqa_gqa"],
    available_corpus["vqa_okvqa"],
    available_corpus["vqa_a_okvqa"],
    available_corpus["vqa_viquae"],
    available_corpus["vqa_ocr_vqa"], 
    available_corpus["vqa_text_vqa"],
    available_corpus["vqa_st_vqa"],
    available_corpus["vqa_docvqa"],
    # new image instruction
    available_corpus["reasoning_science_qa"], 
    available_corpus["vqa_infovqa"], 
    available_corpus["conversation_lvis_instruct4v"],
    available_corpus["vqa_ai2d"],
    available_corpus["vqa_chart_qa"],
    available_corpus["vqa_dvqa_80k"],
    available_corpus["caption_sharegpt4v_420k"],
    available_corpus["grounding_coco"],
    available_corpus["grounding_vg"],
    # video
    available_corpus["caption_textvr"],
    available_corpus["caption_videochat"],
    available_corpus["caption_youcook2"],
    available_corpus["caption_smit_40k"], # decrease
    available_corpus["classification_k710"],
    available_corpus["classification_ssv2"],
    available_corpus["conversation_videochat1"],
    available_corpus["conversation_videochat2"],
    available_corpus["conversation_videochatgpt"],
    available_corpus["reasoning_next_qa"],
    available_corpus["reasoning_clevrer_qa"],
    available_corpus["reasoning_clevrer_mc"],
    available_corpus["vqa_ego_qa"],
    available_corpus["vqa_tgif_transition_qa"],
    available_corpus["vqa_webvid_qa_30k"], # decrease
    # new video instruction
    available_corpus["caption_sharegptvideo_300k"],
    available_corpus["vqa_sharegptvideo_240k"],
    available_corpus["caption_vidln_kinetics"],
    available_corpus["caption_vidln_oops"],
    available_corpus["caption_vidln_ovis"],
    available_corpus["caption_vidln_uvo_sparse"],
    available_corpus["caption_vidln_uvo_dense"],
    available_corpus["caption_favd"],
    available_corpus["grounding_didemo"],
    # test
    available_corpus["conversation_sharegpt"],
]

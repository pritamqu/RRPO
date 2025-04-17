import os
import argparse
import json
from typing import List, Dict, Optional, Union
import re

CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    pred_answer = re.findall("\(?\[?([A-D])\]?\)?\.?", s)

    try:
        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip("()")
        return pred_answer
    except Exception as e:
        # FIXME: its safe to print the errors
        # print(e)
        # print(f"no answer found from: {s}")
        return ""
    
    # if len(s.split()) > 10 and not re.search("[ABCD]", s):
    #     return ""
    # matches = re.search(r'[ABCD]', s)
    # if matches is None:
    #     return ""
    # return matches[0]


def eval_your_results(
        args
    ):

    response_path=args.response_path
    results_path=args.results_path
    video_types="short,medium,long"
    gt_answer_key = "answer"
    your_answer_key = "response"
    skip_missing=False

    if response_path.endswith('.json'):
        your_results=json.load(open(response_path, 'r'))
    elif response_path.endswith('.jsonl'):
        your_results=[json.loads(q) for q in open(response_path, 'r')]
    else:
        raise NotImplementedError(f'unknown file type {response_path}')

    if isinstance(video_types, str):
        video_types = video_types.split(",")

    q_type_dict = {}
    v_type_dict = {}
    v_sub_type_dict = {}

    for video_type in video_types:

        # Filter your results based on video types
        your_results_video_type = [item for item in your_results if item["duration"] == video_type]

        # Task Categories
        q_type_dict[video_type] = {}
        for q_type in TASK_CATEGORIES:
            q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

        # Video categories
        v_type_dict[video_type] = {}
        for v_type in CATEGORIES:
            v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}
        
        v_sub_type_dict[video_type] = {}
        for v_sub_type in SUB_CATEGORIES:
            v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

        if not skip_missing:
            # Check if the number of files in your results and ground truth are the same
            assert len(your_results_video_type) == 300, f"Number of files in {video_type} is not 300. Check if there are missing files."

        for item in your_results_video_type:
            if skip_missing and item["missing"]:
                continue

            # Get the video category, sub category and question category
            video_category = item["domain"]
            video_sub_category = item["sub_category"]
            
            questions = item["questions"]

            for question in questions:
                q_type = question["task_type"]

                # Get the ground truth and your response
                gt_answer = question[gt_answer_key]
                response = question[your_answer_key]

                # Extract the answer from the response
                extration = extract_characters_regex(response)
    
                if extration != "":
                    q_type_dict[video_type][q_type]["answered"] += 1
                    q_type_dict[video_type][q_type]["correct"] += extration == gt_answer

                    v_type_dict[video_type][video_category]["answered"] += 1
                    v_type_dict[video_type][video_category]["correct"] += extration == gt_answer

                    v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
                    v_sub_type_dict[video_type][video_sub_category]["correct"] += extration == gt_answer


    score_by_duration={}
    for video_type in video_types:

        total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
        total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
        _acc = 100 * total_correct / total_answered if total_answered > 0 else 0 
        score_by_duration[video_type+'_acc']=round(_acc, 1)

    total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
    total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types])

    _acc=100 * total_correct / total_answered if total_answered > 0 else 0
    score_by_duration['average_acc']=round(_acc, 1)

    # print(score_by_duration)
    if results_path:
        with open(results_path, "w") as f:
            json.dump(score_by_duration, f)

    return score_by_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    eval_your_results(args)



import os
import re
import json
import random        
import numpy as np
import torch
from tqdm import tqdm

def setup_seed(seed=428):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False


def cal_score(results):
    basic_acc = 0
    halluc_acc = 0
    acc = 0
    for result in results:
        
        basic_hit = 0
        halluc_hit = 0
        final_hit = 0

        basic_answer = result["basic"]["answer"]
        basic_predict = result["basic"]["predict"]
        basic_answer_pattern = r'\b('+basic_answer+ r')\b'
        if re.search(basic_answer_pattern, basic_predict, re.IGNORECASE):
            basic_hit = 1

        halluc_answer = result["hallucination"]["answer"]
        halluc_predict = result["hallucination"]["predict"]
        halluc_answer_pattern = r'\b('+halluc_answer+ r')\b'
        if re.search(halluc_answer_pattern, halluc_predict, re.IGNORECASE):
            halluc_hit = 1
        
        final_hit = int(basic_hit and halluc_hit)

        basic_acc += basic_hit
        halluc_acc += halluc_hit
        acc += final_hit
    
    scores = {
        "basic_accuracy": basic_acc / len(results),
        "halluc_accuracy": halluc_acc / len(results),
        "accuracy": acc / len(results)
    }

    return scores

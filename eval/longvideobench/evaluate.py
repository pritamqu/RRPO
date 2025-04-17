
import argparse
import json
import os
import re


def calc_mcq(args):

    try:
        results=json.load(open(args.response_file))
    except:
        results=[json.loads(q) for q in open(args.response_file, 'r')]

    pattern = r"\(?\[?([A-Za-z])\]?\)?\.?"
    acc=0
    for sample in results:
        _pred=sample['pred']
        match = re.search(pattern, _pred, re.IGNORECASE)
        if match:
            _pred = match.group(1)
        else:
            print("[Error] failed to find the answer from raw text: ", _pred)
            _pred='failed'

        _gt=sample['gt']

        if _pred.lower()==_gt.lower():
            acc+=1


    return {'acc': acc/len(results),}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file', help='Path to the ground truth file containing question.', required=True)
    return parser.parse_args()

if __name__=='__main__':
    args=parse_args()
    ans=calc_mcq(args)
    
    if args.response_file.endswith('.jsonl'):
        result_file=args.response_file[:-6]+'_result.json'
    elif args.response_file.endswith('.json'):
        result_file=args.response_file[:-5]+'_result.json'
    else:
        raise ValueError()
    
    json.dump(ans, open(result_file, 'w'), indent=2)

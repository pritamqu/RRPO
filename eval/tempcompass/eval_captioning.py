from .prompt_templates import *

import requests, json, os, argparse, random, time
from .utils.eval_utils import url, headers, print_result, process_gemini_caption, process_reka_caption
from tqdm import tqdm

qtype = "captioning"
base_prompt = caption_evaluation_prompt

def parse_llm_output(llm_output, gt_answer):

    if llm_output=="invalid_request_error" or not llm_output:
        eval_result = {"rating": -1, "chatgpt-answer": None, "chatgpt-reasoning": None}
        return eval_result
    
    eval_result = {}
    lines = llm_output.split("\n")

    for line in lines:
        line = line.strip()
        if "Reasoning" in line:
            eval_result["chatgpt-reasoning"] = line.replace("Reasoning:", "").strip()
        if "Answer" in line:
            eval_result["chatgpt-answer"] = line.replace("Answer:", "").strip()

    if not "chatgpt-answer" in eval_result:
        eval_result["chatgpt-answer"] = llm_output
    if not "chatgpt-reasoning" in eval_result:
        eval_result["chatgpt-reasoning"] = None

    # Check if the chatgpt answer is the ground-truth answer
    answer_counts = sum(eval_result["chatgpt-answer"].count(prefix) for prefix in ['A.', 'B.', 'C.', 'D.'])     # calculate the number of 'A.', 'B.', 'C.', 'D.' in chatgpt-answer
    if eval_result["chatgpt-answer"].split(". ")[0]==gt_answer.split(". ")[0] and answer_counts==1:
        eval_result["rating"] = 1
    else:
        eval_result["rating"] = 0
    return eval_result

def get_llm_output(client, gpt_model, prompt, sys_prompt=None, max_tokens=128):
    if sys_prompt is None:
        sys_prompt = "You are an AI assistant for question answering."
    data = {
        "max_tokens": max_tokens,
        "model": gpt_model,
        "temperature": 1.0,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [
            {
                "role": "system",
                "content": sys_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    llm_output=None
    # while llm_output is None:
    try:
        completion = client.chat.completions.create(**data)
        llm_output = completion.choices[0].message.content
        llm_output = llm_output.strip()
        # token_count = completion.usage.completion_tokens
        # print(llm_output)
    except Exception as e:
        print(e)

    return llm_output, None

def get_eval_result(client, gpt_model, prompt, mc_answer, maxtry=10):
    while True:
        try:
            llm_output, token_count = get_llm_output(client, gpt_model, prompt)
            eval_result = parse_llm_output(llm_output, gt_answer=mc_answer)
            # eval_result["token_count"] = token_count
            return eval_result
        except:
            if maxtry<=0:
                eval_result = {"chatgpt-reasoning": None, "chatgpt-answer": None, "rating": -1, "token_count": None}
                return eval_result
            maxtry -= 1
            print(f"Not success! {maxtry} retries remaining...")
            time.sleep(random.uniform(1, 2))

def main(predictions, eval_results, output_file, mc_questions):
    for id in tqdm(predictions):

        if id not in eval_results:
            eval_results[id] = {}

        for dim, preds in predictions[id].items():

            if dim in eval_results[id] and eval_results[id][dim] and len(preds)==len(eval_results[id][dim]):    # skip if the eval result already exists
                continue
            eval_results[id][dim] = []

            for pred in preds:
                if "prediction" not in pred and "response" in pred:
                    pred["prediction"] = pred["response"]

                if pred["prediction"] is None:  # In some cases the Video LLM may refuse to produce a response
                    eval_result = {"question": pred["question"], "gt-answer": pred["answer"], "video-llm-prediction": pred["prediction"], "match_success": False, "rating": 0}
                    eval_results[id][dim].append(eval_result)
                    continue

                pred["prediction"] = pred["prediction"].replace('</s>', '').strip()

                # if 'gemini' in args.video_llm:
                #     pred["prediction"] = process_gemini_caption(pred["prediction"])
                # if 'reka' in args.video_llm:
                #     pred["prediction"] = process_reka_caption(pred["prediction"])

                prompt = f"""{base_prompt}\nVideo Description:{pred["prediction"]}\nMulti-Choice Question:\n{mc_questions[id][dim][0]["question"]}\n"""

                eval_result = get_eval_result(client, gpt_model, prompt, mc_answer=mc_questions[id][dim][0]["answer"])

                eval_result["video-llm-prediction"] = pred["prediction"]
                eval_result["gt-answer"] = mc_questions[id][dim][0]["answer"]
                eval_results[id][dim].append(eval_result)

    with open(os.path.expanduser(output_file), "w") as f:
        json.dump(eval_results, f, indent=4)

    score = print_result(eval_results)

    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="./", help="path to the raw response")
    parser.add_argument('--output_dir', default="./", help="path to store eval results")
    parser.add_argument("--root-dir", type=str, default='/datasets/video_llm/video_eval/TempCompass')
    parser.add_argument("--gpt_model", default="gpt-3.5-turbo-0125",)
    parser.add_argument("--api_key", default="",)

    args = parser.parse_args()

    # global client, gpt_model
    gpt_model=args.gpt_model
    from openai import OpenAI
    client = OpenAI(api_key=args.api_key)

    input_file = f"{args.input_dir}/{qtype}.json"
    output_file = f"{args.output_dir}/{qtype}.json"
    result_file = f"{args.output_dir}/score_{qtype}.json"
    
    print('*'*10, qtype, '*'*10)

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Loading video-llm predictions and multi-choice questions
    with open(input_file, 'r') as f:
        predictions = json.load(f)

    # Loading already evaluated results
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {}

    # Loading multi-choice questions
    with open(os.path.join(args.root_dir, "questions/multi-choice.json"), 'r') as f:
        mc_questions = json.load(f)

    score = main(predictions, eval_results, output_file, mc_questions)

    with open(result_file, "w") as f:
        json.dump(score, f)
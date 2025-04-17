from .prompt_templates import *
from .utils.eval_utils import *
import json, os, argparse
from tqdm import tqdm

qtype = "yes_no"
base_prompt = yes_no_evaluation_prompt

def extract_pred(video_llm_output):
    # Extract the yes/no predction from the original video llm output
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "yes"
    elif video_llm_output.startswith("no"):
        return "no"
    else:
        return False

def main(predictions, eval_results, output_file, disable_llm):
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
                eval_result = {"question": pred["question"], "gt-answer": pred["answer"], "video-llm-prediction": pred["prediction"], "match_success": True}

                yes_no_pred = extract_pred(pred["prediction"])  # Some hand-crafted matching rules
                if yes_no_pred:
                    eval_result["rating"] = 1 if yes_no_pred==pred["answer"] else 0
                elif disable_llm:
                    eval_result["match_success"] = False    
                    eval_result["rating"] = 0               # Fail to match answer in the video-llm response. Directly set rating to 0
                else:
                    eval_result["match_success"] = False    # Fail to match answer in the video-llm response. Use ChatGPT to evaluate.
                    prompt = f"""{base_prompt}\nYes/No Question:\n{pred["question"]}\nGround-Truth Answer: {pred["answer"]}\nModel Prediction: {pred["prediction"]}"""
                    eval_result["chatgpt-response"], eval_result["rating"] = get_eval_result(client, gpt_model, prompt)

                eval_results[id][dim].append(eval_result)

    with open(os.path.expanduser(output_file), "w") as f:
        json.dump(eval_results, f, indent=4)

    score = print_result(eval_results)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="./", help="path to the raw response")
    parser.add_argument('--output_dir', default="./", help="path to store eval results")
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
        print(output_file)
        with open(output_file, 'r') as f:
            eval_results = json.load(f)
    else:
        eval_results = {}

    score = main(predictions, eval_results, output_file, disable_llm=False)

    with open(result_file, "w") as f:
        json.dump(score, f)
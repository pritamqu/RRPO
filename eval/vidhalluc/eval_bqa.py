### this script is shared by Chaoyu Li

import json
import sys

def compute(gt_data, pred_data):
    total_questions = 0
    correct_questions = 0
    for key in gt_data.keys():
        gt_questions = gt_data[key]
        pred_questions = pred_data.get(key, [])

        # If the key is missing in the prediction file, skip and print a warning
        if not pred_questions:
            print(f"Missing key in prediction file: {key}")
            continue

        # Iterate over each question for the current key
        for idx, gt_question in enumerate(gt_questions):
            # Check if there is a corresponding question in the prediction file
            if idx >= len(pred_questions):
                print(f"Missing question {idx + 1} for key {key} in prediction file")
                continue

            pred_question = pred_questions[idx]

            gt_answers = gt_question['a']
            pred_answers = pred_question.get('a', {})

            # Initialize a flag indicating whether all answers for this question are correct
            all_correct = True

            # Iterate over each clip's answer
            for clip_key, gt_answer in gt_answers.items():
                pred_answer = pred_answers.get(clip_key)

                # If the prediction answer is missing, print a warning and mark as incorrect
                if pred_answer is None:
                    print(f"Missing clip {clip_key} for question {idx + 1} with key {key} in prediction file")
                    all_correct = False
                    break

                # Standardize answers for comparison
                pred_answer = pred_answer.strip().split('.')[0].split(',')[0].strip().upper()
                gt_answer = gt_answer.strip().upper()

                # If the answers do not match, mark as incorrect
                if gt_answer != pred_answer:
                    all_correct = False
                    break

            total_questions += 1  # Increment the total question count

            # If all answers for the question are correct, increment the correct count
            if all_correct:
                correct_questions += 1
    return total_questions, correct_questions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <gt_file.json> <pred_file.json>")
        sys.exit(1)

    gt_file_path = sys.argv[1]
    pred_file_path = sys.argv[2]

    # Load ground truth file
    with open(gt_file_path, 'r', encoding='utf-8') as gt_file:
        gt_data = json.load(gt_file)

    # Load prediction file
    with open(pred_file_path, 'r', encoding='utf-8') as pred_file:
        pred_data = json.load(pred_file)

    total_questions, correct_questions = compute(gt_data, pred_data)
    if total_questions > 0:
        print(correct_questions, total_questions)
        accuracy = correct_questions / total_questions
        print(f"Accuracy: {accuracy:.4f}")

        result_file_path=pred_file_path.split('_preds.json')[0]+'_results.json'
        json.dump(dict(accuracy=accuracy), open(result_file_path, 'w'))
    else:
        print("No data available for accuracy computation.")

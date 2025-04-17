### this script is shared by Chaoyu Li

import json
import sys
import re


def extract_answer(model_answer):
    """Extract answer choices (A-D) from the model's response.

    Returns a set of uppercase letters found in the answer.
    """
    choices = re.findall(r'\b[A-D]\b', model_answer)
    if choices:
        return set([c.upper() for c in choices])
    return set()


def match_model_answer(data):
    """
    For cases where the model output does not include a direct answer letter,
    this function searches each question's "Choices" to find an option whose
    text appears in the model answer. The found choice key is stored in the
    "Matched Answer" field (or "None" if no match is found).
    """
    for key, clips in data.items():
        for clip_id, clip_data in clips.items():
            model_answer = clip_data.get("Model Answer", "").lower()
            choices = clip_data.get("Choices", {})
            matched_answer = None
            for choice_key, choice_value in choices.items():
                if choice_value.lower() in model_answer:
                    matched_answer = choice_key
                    break
            clip_data["Matched Answer"] = matched_answer if matched_answer else "None"


def compute_MCQ(data):
    """
    Compute the total number of MCQ questions and the number of correct answers.
    For each question, the ground truth "Correct Answer" is compared with the
    answer extracted from the model's "Model Answer" (or "Matched Answer" if no
    answer letter is directly found).
    """
    count = 0
    correct_count = 0
    for key, clips in data.items():
        for clip_id, clip_data in clips.items():
            count += 1
            correct_answer = clip_data.get('Correct Answer', "").strip().upper()
            correct_answer_set = set(re.findall(r'[A-D]', correct_answer))
            model_answer_set = extract_answer(clip_data.get('Model Answer', ""))
            # If no answer letter is found in the model's response, try using "Matched Answer"
            if not model_answer_set:
                matched = clip_data.get("Matched Answer", "").strip().upper()
                if matched and matched != "NONE":
                    model_answer_set = set([matched])
            if model_answer_set == correct_answer_set:
                correct_count += 1
    return count, correct_count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.json>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

    if not data:
        print("No data in input file.")
        sys.exit(1)

    # Update the "Matched Answer" field for each question
    match_model_answer(data)

    total_questions, correct_questions = compute_MCQ(data)
    if total_questions > 0:
        print(correct_questions, total_questions)
        accuracy = correct_questions / total_questions
        print(f"Accuracy: {accuracy:.4f}")

        result_file_path=file_path.split('_preds.json')[0]+'_results.json'
        json.dump(dict(accuracy=accuracy), open(result_file_path, 'w'))
    else:
        print("No data available for accuracy computation.")

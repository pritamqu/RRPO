### this script is shared by Chaoyu Li

import json
import re
import math
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Load the SimCSE model for sentence embeddings
simcse_model_name = "princeton-nlp/sup-simcse-roberta-large"
simcse_tokenizer = AutoTokenizer.from_pretrained(simcse_model_name)
simcse_model = AutoModel.from_pretrained(simcse_model_name)


def get_simcse_embedding(sentence):
    """Get the SimCSE embedding for a given sentence."""
    inputs = simcse_tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = simcse_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding


def calculate_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return cosine_similarity(embedding1, embedding2)[0][0]


def extract_scenes(description):
    """
    Extract the "from" and "to" scenes from a description.
    The description should match the pattern: 'from ... to ....'
    """
    pattern = r"from (.+?) to (.+?)\."
    match = re.match(pattern, description, re.IGNORECASE)
    if match:
        from_scene = match.group(1).strip()
        to_scene = match.group(2).strip()
        return from_scene, to_scene
    return None, None


def sigmoid(x):
    """Return the sigmoid of x."""
    return 1 / (1 + math.exp(-x))


def calculate_scene_similarity(model_scene, gt_scene, similarity_threshold_high=0.8, similarity_threshold_low=0.3):
    """
    Compute a similarity score between the model and ground truth scene.
    The score is 0 if the similarity is below the low threshold,
    otherwise it is scaled using a sigmoid function.
    """
    embedding_model = get_simcse_embedding(model_scene)
    embedding_gt = get_simcse_embedding(gt_scene)
    similarity = calculate_similarity(embedding_model, embedding_gt)
    if similarity <= similarity_threshold_low:
        score = 0.0
    else:
        score = (sigmoid(similarity) - sigmoid(similarity_threshold_low)) / (
                    sigmoid(1) - sigmoid(similarity_threshold_low))
    return score


def evaluate_description(model_description, gt_description, similarity_threshold_high=0.8,
                         similarity_threshold_low=0.3):
    """
    Evaluate the similarity of scene descriptions.
    It extracts the "from" and "to" scenes from both model and ground truth,
    computes individual similarity scores, and returns their sum.
    The maximum score per description is 2.
    """
    model_from, model_to = extract_scenes(model_description)
    gt_from, gt_to = extract_scenes(gt_description)
    if not model_from or not model_to or not gt_from or not gt_to:
        return 0.0  # Missing necessary scene descriptions
    from_score = calculate_scene_similarity(model_from, gt_from, similarity_threshold_high, similarity_threshold_low)
    to_score = calculate_scene_similarity(model_to, gt_to, similarity_threshold_high, similarity_threshold_low)
    total_score = from_score + to_score
    return total_score


def evaluate_model(video_samples, similarity_threshold_high=0.8, similarity_threshold_low=0.3):
    """
    Evaluate the model predictions using both classification and scene description metrics.

    For each sample:
      - The "GT Scene change" and "Model Scene change" fields are used for binary classification.
      - If both are "yes", the scene descriptions ("GT Locations" and "Model Locations") are evaluated.

    The final score is a weighted combination of a transformed MCC (classification score)
    and the normalized scene description accuracy.
    """
    ground_truth_labels = []
    model_predictions = []
    total_description_score = 0
    max_description_score = 0

    for sample in video_samples:
        gt = sample["GT Scene change"].lower()
        pred = sample["Model Scene change"].lower()
        ground_truth_labels.append(1 if gt == "yes" else 0)
        model_predictions.append(1 if pred == "yes" else 0)

        if gt == "yes" and pred == "yes":
            model_sentence = sample.get("Model Locations", "")
            gt_sentence = sample.get("GT Locations", "")
            if model_sentence and gt_sentence:
                desc_score = evaluate_description(model_sentence, gt_sentence, similarity_threshold_high,
                                                  similarity_threshold_low)
                total_description_score += desc_score
            max_description_score += 2  # Maximum score per evaluated sample
        elif gt == "yes" and pred == "no":
            # For false negatives, do not add to the maximum description score.
            pass

    # Compute confusion matrix components
    cm = confusion_matrix(ground_truth_labels, model_predictions, labels=[1, 0])
    if cm.shape == (2, 2):
        TP, FN = cm[0]
        FP, TN = cm[1]
    else:
        TP = FP = TN = FN = 0
        if ground_truth_labels.count(1) == 0:
            TN = cm[0][0]
        else:
            TP = cm[0][0]

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    mcc = matthews_corrcoef(ground_truth_labels, model_predictions)
    adjusted_mcc = ((mcc + 1) / 2) ** 2
    classification_score = adjusted_mcc

    normalized_description_score = total_description_score / max_description_score if max_description_score > 0 else 0
    overall_score = (classification_score * 0.6) + (normalized_description_score * 0.4)

    return {
        "MCC": mcc,
        "Classification Score": classification_score,
        "Description Accuracy": normalized_description_score,
        "Overall Score": overall_score
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <gt_file.json> <model_file.json>")
        sys.exit(1)

    gt_file_path = sys.argv[1]
    model_file_path = sys.argv[2]

    # Load the model JSON file
    try:
        with open(model_file_path, 'r') as model_file:
            model_data = json.load(model_file)
    except Exception as e:
        print(f"Error loading model file: {e}")
        sys.exit(1)

    # Load the ground truth JSON file
    try:
        with open(gt_file_path, 'r') as gt_file:
            gt_data = json.load(gt_file)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        sys.exit(1)

    # Merge the data into a new dictionary based on keys common to both files
    merged_data = {}
    for key in model_data.keys():
        if key in gt_data:
            model_text = model_data[key]
            # Extract the model's scene change information and locations
            if ", Locations: " in model_text:
                scene_change_part, locations_part = model_text.split(", Locations: ", 1)
                model_scene_change = scene_change_part.split(": ", 1)[-1].split(",")[0].strip()
                model_locations = locations_part.strip()
            else:
                model_scene_change = model_text.split(": ", 1)[-1].split(",")[0].strip()
                model_locations = ""
            gt_scene_change = gt_data[key]["Scene change"]
            gt_locations = gt_data[key]["Locations"]

            merged_data[key] = {
                "GT Scene change": gt_scene_change,
                "GT Locations": gt_locations,
                "Model Scene change": model_scene_change,
                "Model Locations": model_locations
            }

    # Convert the merged data to a list of samples for evaluation
    video_samples = list(merged_data.values())
    results = evaluate_model(video_samples, similarity_threshold_high=0.8, similarity_threshold_low=0.5)

    print(f"MCC: {results['MCC']:.4f}")
    print(f"Classification Score: {results['Classification Score']:.4f}")
    print(f"Description Accuracy: {results['Description Accuracy']:.4f}")
    print(f"Overall Score: {results['Overall Score']:.4f}")

    result_file_path=model_file_path.split('_preds.json')[0]+'_results.json'
    json.dump(results, open(result_file_path, 'w'))


if __name__ == "__main__":
    main()

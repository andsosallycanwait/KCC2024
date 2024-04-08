import json
import re
import string

import json

import json

def load_predictions(predictions_file):
    with open(predictions_file, 'r') as f:
        return json.load(f)

def load_eval_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)['data']
    cases = []
    for entry in data:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                cases.append({
                    'qas_id': qa['id'],
                    'question_text': qa['question'],
                    'context_text': paragraph['context'],
                    'answers': [answer['text'] for answer in qa['answers']]
                })
    return cases

def categorize_matches(cases, preds):
    exact_matches = []
    mismatches = []
    for case in cases:
        qas_id = case['qas_id']
        predicted_answer = preds.get(qas_id, "No prediction found").strip()
        answers = [ans.strip() for ans in case['answers']]
        # Check for exact match but exclude "CANNOTANSWER"
        if predicted_answer in answers and predicted_answer.lower() != "cannotanswer":
            exact_matches.append(case)
        else:
            mismatches.append(case)
    return exact_matches, mismatches

def print_cases(cases, preds, title):
    print(f"\n{title}:")
    for case in cases[:3]:  # Print first 3 examples
        qas_id = case['qas_id']
        predicted_answer = preds.get(qas_id, "No prediction found")
        print(f"\nQuestion ID: {qas_id}")
        print(f"Question: {case['question_text']}")
        print(f"Context: {case['context_text'][:100]}...")
        print(f"True Answers: {case['answers']}")
        print(f"Predicted Answer: {predicted_answer}")

predictions_file = '/home/nilu/SpanBERT/predictions.json'
dataset_filename = '/home/nilu/SpanBERT/doqa-cooking-dev-v2.1.json'

preds = load_predictions(predictions_file)
cases = load_eval_data(dataset_filename)

exact_matches, mismatches = categorize_matches(cases, preds)

print_cases(exact_matches, preds, "Exact Matches (excluding 'CANNOTANSWER')")
print_cases(mismatches, preds, "Mismatches")



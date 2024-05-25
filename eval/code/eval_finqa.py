# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_finqa.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/llms/mistral-hf")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from tqdm import tqdm
from collections import Counter


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_finqa(model_path, test_data_path, args):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    
    y_true = []
    y_pred = []
    count = 0
    # Generate predictions
    for sample in tqdm(test_data):
        if count > 10:
            break
        count += 1
        prompt = "Please strictly format your answer:" + sample['instruction'] + '\n\n' + sample['input']
        final_prompt = instr_prompt(content=prompt)
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
        # Split the expected output and the generated answer by comma and newline
        pattern = re.compile(r'[.,;:!?]\s*|\n')
        #true_entities = [entity.strip() for entity in pattern.split(sample['output']) if entity.strip()]
        true_entities = re.sub(r"[^\w\s']", ' ', sample['output']).lower().split()
        predicted_entities = re.sub(r"[^\w\s]", ' ', answer).lower().split()
        print('='*50)
        print(true_entities)
        print(predicted_entities)
        true_idx = 0
        is_correct = True
        
        for pred_entity in predicted_entities:
            if true_idx < len(true_entities) and true_entities[true_idx] == pred_entity:
                true_idx += 1
            if true_idx == len(true_entities):
                break
        if true_idx != len(true_entities):
            is_correct = False
        print(is_correct)
        y_true.append(1)
        y_pred.append(1 if is_correct else 0)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return results

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/FINQA/fingpt-fineval_test.json"
    path_li = ["../benchmark/FINQA/flare-cfa-test.json", "../benchmark/FINQA/flare-finqa_test.json", "../benchmark/FINQA/flare-mlesg.json"]
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }
    for path in path_li:
        test_data_path = path
        results = evaluate_finqa(model_path, test_data_path, args)
        print(results)

if __name__ == "__main__":
    main()

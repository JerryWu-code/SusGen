# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_ner2.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/")
from sklearn.metrics import precision_score, recall_score, f1_score
from seqeval.metrics import f1_score as entity_f1_score, precision_score as entity_precision_score, recall_score as entity_recall_score
from template import load_model, generate_text, instr_prompt
from tqdm import tqdm
from collections import Counter


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_text(pred, text):
    entities = []
    lines = pred.split('\n')
    for line in lines:
        match = re.search(r'<(.*?)>', line)
        if match:
            entity = match.group(1)
            if entity in ['PER', 'LOC', 'ORG']:
                entities.append(entity)
    return entities

def evaluate_ner(model_path, test_data_path, args):
    model, tokenizer, device, _ = load_model(model_path)
    
    test_data = load_json(test_data_path)
    
    true_entities_list = []
    pred_entities_list = []
    count = 0
    for sample in tqdm(test_data):
        if count > 10:
            break
        count += 1
        prompt = "Please strictly format your answer:" + sample['instruction'] + '\n\n' + sample['input']
        final_prompt = instr_prompt(content=prompt)
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
        true_entities = sample['output'].split()
        pred_entities = process_text(answer, sample['input'])
        
        true_entities_list.append(true_entities)
        pred_entities_list.append(pred_entities)
        
        if len(true_entities) == len(pred_entities):
            true_entities_list.append(true_entities)
            pred_entities_list.append(pred_entities)
        else:
            max_len = max(len(true_entities), len(pred_entities))
            true_entities.extend(['O'] * (max_len - len(true_entities)))
            pred_entities.extend(['O'] * (max_len - len(pred_entities)))
            true_entities_list.append(true_entities)
            pred_entities_list.append(pred_entities)
    
    f1 = entity_f1_score(true_entities_list, pred_entities_list)
    precision = entity_precision_score(true_entities_list, pred_entities_list)
    recall = entity_recall_score(true_entities_list, pred_entities_list)
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return results

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/NER/flare-ner-test.json"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results = evaluate_ner(model_path, test_data_path, args)
    print(results)

if __name__ == "__main__":
    main()


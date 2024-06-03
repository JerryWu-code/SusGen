# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_ner.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
from tqdm import tqdm
import pandas as pd
from collections import Counter
import random

ent_dict = {
    'PER': 'person',
    'ORG': 'organization',
    'LOC': 'location',
}
ent_dict_rev = {v: k for k, v in ent_dict.items()}

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_ner(model_path, test_data_path, args, prompt_type='mistral'):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    random.seed(42)  # For reproducibility
    test_data = random.sample(test_data, 50)
    y_true = []
    y_pred = []
    eval_results = []
    # count = 0
    # Generate predictions
    for sample in tqdm(test_data):
        # if count > 50:
        #     break
        # count += 1
        # prompt = "Please strictly format your answer with instructions:" + sample['instruction'] + '\n\n' + sample['input']
        # final_prompt = instr_prompt(content=prompt)
        if prompt_type == 'mistral':
            final_prompt = mistral_formal_infer(sample)
        elif prompt_type == 'llama3':
            final_prompt = llama3_formal_infer(sample)
        else:
            raise ValueError("Invalid prompt type")
        
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
        # check if per, org, loc are in true entities, if so, we need to check if the count of these entities are matched in the predicted entities
        entity_types = ['per', 'org', 'loc']

        true_entity_counts = Counter(true_entities)
        predicted_entity_counts = Counter(predicted_entities)

        for entity_type in entity_types:
            true_count = true_entity_counts.get(entity_type, 0)
            predicted_count = predicted_entity_counts.get(entity_type, 0)
            if true_count != predicted_count:
                is_correct = False
                break
        
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
        
        eval_results.append({
            'prompt': final_prompt,
            'generated': answer,
            'target': sample['output'],
            'is_correct': 'Yes' if is_correct else 'No'
        })
        
    df = pd.DataFrame(eval_results)
    # df.to_csv(output_csv_path, index=False)
    

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
    
    # with open(output_txt_path, 'w') as f:
    #     for key, value in results.items():
    #         f.write(f"{key}: {value}\n")
            
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/NER/NER.json"
    output_csv_path = "../results/Mistral-v0.2/NER/ner_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/NER/ner_eval_results.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results, df = evaluate_ner(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)
    with open(output_txt_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(results)

if __name__ == "__main__":
    main()

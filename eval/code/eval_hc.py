import json, sys, os
sys.path.append("/home/whatx/SusGen/src/")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
from tqdm import tqdm
import re
import pandas as pd
import random


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_headline_classification(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=100):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path, lora_path, quantization)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    if random_count < len(test_data):
        random.seed(42)  # For reproducibility
        test_data = random.sample(test_data, random_count)
    y_true = []
    y_pred = []
    eval_results = []
    # Generate predictions
    for sample in tqdm(test_data):
        # prompt = sample['instruction']+'\n\n'+sample['input']
        # final_prompt = instr_prompt(content=prompt)
        if prompt_type == 'mistral':
            final_prompt = mistral_formal_infer(sample)
        elif prompt_type == 'llama3':
            final_prompt = llama3_formal_infer(sample)
        else:
            raise ValueError("Invalid prompt type")
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
 
        # y_pred.append(answer.strip())
        if 'yes' or 'no' in sample['output'].lower():
            predicted_anwser = 'yes' if 'yes' in answer.lower() else 'no'
            y_true.append(1)
            y_pred.append(1 if predicted_anwser == sample['output'].lower() else 0)
        else:
            true_entities = re.sub(r"[^\w\s']", ' ', sample['output']).split()
            predicted_entities = re.sub(r"[^\w\s]", ' ', answer).split()
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
            
        eval_results.append({
            'prompt': final_prompt,
            'generated': answer,
            'target': sample['output'],
            'is_correct': 'Yes' if y_pred[-1] else 'No'
        })
        
    df = pd.DataFrame(eval_results)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'micro_f1': micro_f1
    }
    
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/HC/fingpt-headline-cls_test.json"
    output_csv_path = "../results/Mistral-v0.2/HC/hc_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/HC/hc_eval_results.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results, df = evaluate_headline_classification(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)
    with open(output_txt_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(results)
if __name__ == "__main__":
    main()

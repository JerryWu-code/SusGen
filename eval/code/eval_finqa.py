# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_finqa.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
from tqdm import tqdm
from collections import Counter
import pandas as pd
import random


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_answer(text):
    # Match the answer pattern
    pattern = re.compile(r'answer is\s*([ABCD])(?=\s|[.,:])')
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return None
    
def extract_numbers(true_value, predicted_text, threshold=0.01):
    
    def convert_to_decimal(value):
        if value.endswith('%'):
            return round(float(value[:-1]) / 100, 5)
        return round(float(value), 5)
    # remove commas in between numbers
    predicted_text = re.sub(r'(\d),(\d)', r'\1\2', predicted_text)
    
    true_value = convert_to_decimal(true_value)
    
    pattern = re.compile(r'-?\d+\.?\d*%?')
    predicted_numbers = pattern.findall(predicted_text)
    predicted_numbers = [convert_to_decimal(num) for num in predicted_numbers]
    
    if true_value == 0:
        return any(num == 0 for num in predicted_numbers)
    
    for num in predicted_numbers:
        if abs((num - true_value) / true_value) < threshold:
            return True
    return False

def is_number(s):
    pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(pattern.match(s))

def evaluate_finqa(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=100):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path, lora_path, quantization)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    random.seed(42)  # For reproducibility
    test_data = random.sample(test_data, random_count)
    y_true = []
    y_pred = []
    eval_results = []
    # count = 0
    # Generate predictions
    for sample in tqdm(test_data):
        # if count > 50:
        #     break
        # count += 1
        # prompt = sample['instruction'] + '\n\n' + sample['input']
        # final_prompt = instr_prompt(content=prompt)
        if prompt_type == 'mistral':
            final_prompt = mistral_formal_infer(sample)
        elif prompt_type == 'llama3':
            final_prompt = llama3_formal_infer(sample)
        else:
            raise ValueError("Invalid prompt type")
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
        if sample['output'] in 'ABCD':
            predicted_answer = extract_answer(answer)
            print('='*50)
            print(predicted_answer)
            y_true.append(1)
            y_pred.append(1 if predicted_answer == sample['output'] else 0)
            print(predicted_answer == sample['output'])
        elif is_number(sample['output']):
            print('='*50)
            print(sample['output'])
            y_true.append(1)
            y_pred.append(1 if extract_numbers(sample['output'], answer) else 0)
            print('True' if y_pred[-1] else 'False')
        else:
            # Split the expected output and the generated answer by comma and newline
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
            'model_output': answer,
            'true_output': sample['output'],
            'is_correct': 'Yes' if y_pred[-1] else 'No'
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
    #     f.write(f"Accuracy: {accuracy}\n")
    #     f.write(f"Precision: {precision}\n")
    #     f.write(f"Recall: {recall}\n")
    #     f.write(f"F1 Score: {f1}\n")    
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    # path_li = ["../benchmark/FINQA/flare-cfa-test.json", "../benchmark/FINQA/flare-finqa_test.json", "../benchmark/FINQA/flare-mlesg.json"]
    # test_data_path = path_li[1]
    test_data_path = "../benchmark/FINQA/FinQA.json"
    output_csv_path = "../results/Mistral-v0.2/SA/finqa_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/SA/finqa_eval_results.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }
    results, df = evaluate_finqa(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)    
    with open(output_txt_path, 'w') as f:
        f.write(f"EmAccuracy: {results['accuracy']}\n")
        f.write(f"Precision: {results['precision']}\n")
        f.write(f"Recall: {results['recall']}\n")
        f.write(f"F1 Score: {results['f1_score']}\n")
    print(results)
    
    # final_result = {}
    # for path in path_li:
    #     test_data_path = path
    #     results = evaluate_finqa(model_path, test_data_path, args)
    #     print(results)
    #     final_result[path] = results
    # print(final_result)
    
if __name__ == "__main__":
    main()

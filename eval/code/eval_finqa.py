# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_finqa.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
from tqdm import tqdm
from collections import Counter
import pandas as pd
import random
from batch_inference import load_model, inference

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

def evaluate_finqa(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=100, batch_size=100, model=None):
    # define a function to check correctness
    def check_correctness(batch_samples, answers, prompts):
        y_true = []
        y_pred = []
        eval_results = []
        for sample, answer, final_prompt in zip(batch_samples, answers, prompts):
            if test_data_path == "../benchmark/FINQA/FSRL.json":
                true_entities = [line.split(':') for line in sample['output'].split('\n')]
                predicted_entities = [line.split(':') for line in answer.split('\n') if ':' in line]
                # select the one that occured in labels
                true_labels = [entity[1] for entity in true_entities]
                predicted_labels = [entity[1] for entity in predicted_entities[:len(true_labels)]]
                
                # Ensure the lengths are the same
                min_len = min(len(true_labels), len(predicted_labels))
                true_labels = true_labels[:min_len]
                predicted_labels = predicted_labels[:min_len]
                y_true.extend(true_labels)
                y_pred.extend(predicted_labels)
                eval_results.append({
                    'prompt': final_prompt,
                    'generated': answer,
                    'target': sample['output'],
                    'true_labels': true_labels,
                    'predicted_labels': predicted_labels
                })
            elif 'yes' in sample['output'].lower() or 'no' in sample['output'].lower():
                predicted_anwser = 'yes' if 'yes' in answer.lower() else 'no'
                y_true.append(1)
                y_pred.append(1 if predicted_anwser == sample['output'].lower() else 0)
                
            elif sample['output'] in 'ABCD':
                predicted_answer = extract_answer(answer)
                y_true.append(1)
                y_pred.append(1 if predicted_answer == sample['output'] else 0)
            elif is_number(sample['output']):
                y_true.append(1)
                y_pred.append(1 if extract_numbers(sample['output'], answer) else 0)
            else:
                # Split the expected output and the generated answer by comma and newline
                true_entities = re.sub(r"[^\w\s']", ' ', sample['output']).split()
                predicted_entities = re.sub(r"[^\w\s]", ' ', answer).split()

                true_idx = 0
                is_correct = True
                for pred_entity in predicted_entities:
                    if true_idx < len(true_entities) and true_entities[true_idx] == pred_entity:
                        true_idx += 1
                    if true_idx == len(true_entities):
                        break
                if true_idx != len(true_entities):
                    is_correct = False
                y_true.append(1)
                y_pred.append(1 if is_correct else 0)
            
            eval_results.append({
                'prompt': final_prompt,
                'model_output': answer,
                'true_output': sample['output'],
                'is_correct': 'Yes' if y_pred[-1] else 'No'
            })
        return y_true, y_pred, eval_results
    
    # Load the model and tokenizer
    # model, tokenizer, device, _ = load_model(model_path, lora_path, quantization)
    if model is None:
        model = load_model(model_path)
        
    # modify args
    args['max_tokens'] = 512
    batch_size = batch_size // 2
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    if random_count < len(test_data):
        random.seed(42)  # For reproducibility
        test_data = random.sample(test_data, random_count)
    
    # Set up cache folder
    cache_folder = os.path.join('../results', model_path.split('/')[-1], 'cache', os.path.basename(test_data_path).replace('.json', ''))
    os.makedirs(cache_folder, exist_ok=True)
    
    # Check cache and determine where to start
    cached_files = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(cache_folder) if f.startswith('cache_')])
    start_index = 0
    if cached_files:
        start_index = max(cached_files) + 1
        with open(os.path.join(cache_folder, f'cache_{start_index-1}.json'), 'r') as f:
            cache_data = json.load(f)
        y_true = cache_data['y_true']
        y_pred = cache_data['y_pred']
        eval_results = cache_data['eval_results']
    else:
        y_true = []
        y_pred = []
        eval_results = []
        
    # Generate predictions
    # for idx, sample in enumerate(tqdm(test_data[start_index:], initial=start_index, total=len(test_data))):
    for batch_start in tqdm(range(start_index, len(test_data), batch_size), initial=start_index, total= len(test_data) // batch_size + 1):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_samples = test_data[batch_start:batch_end]
        prompts = []
        for sample in batch_samples:
            if prompt_type == 'mistral':
                prompts.append('Please show the result first.' + mistral_formal_infer(sample))
            elif prompt_type == 'llama3':
                prompts.append('Please show the result first.' + llama3_formal_infer(sample))
            else:
                raise ValueError("Invalid prompt type")
        answers = inference(model, prompts, args, verbose=False)
        # if prompt_type == 'mistral':
        #     final_prompt = mistral_formal_infer(sample)
        # elif prompt_type == 'llama3':
        #     final_prompt = llama3_formal_infer(sample)
        # else:
        #     raise ValueError("Invalid prompt type")
        
        # _, answers = generate_text(model, tokenizer, device, final_prompt, args)
        
        true_labels, predicted_labels, eval_result = check_correctness(batch_samples, answers, prompts)
        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)
        eval_results.extend(eval_result)
        
        # Save to cache
        cache_data = {
            'y_true': y_true,
            'y_pred': y_pred,
            'eval_results': eval_results
        }
        with open(os.path.join(cache_folder, f'cache_{batch_end}.json'), 'w') as f:
            json.dump(cache_data, f)
    
    df = pd.DataFrame(eval_results)
    # df.to_csv(output_csv_path, index=False)

    # Calculate evaluation metrics
    if test_data_path == "../benchmark/FINQA/FSRL.json":
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', labels=['O', 'I-QUANT', 'B-QUANT', 'I-TIME', 'B-TIME', 'I-MANNER', 'B-MANNER', 'I-THEME', 'B-THEME', 'I-VALUE', 'B-VALUE', 'I-WHOLE', 'B-WHOLE', 'I-LOCATION', 'B-LOCATION', 'I-AGENT', 'B-AGENT', 'I-CAUSE', 'B-CAUSE', 'I-SOURCE', 'B-SOURCE', 'I-REF_TIME', 'B-REF_TIME', 'I-CONDITION', 'B-CONDITION'])
        recall = recall_score(y_true, y_pred, average='macro', labels=['O', 'I-QUANT', 'B-QUANT', 'I-TIME', 'B-TIME', 'I-MANNER', 'B-MANNER', 'I-THEME', 'B-THEME', 'I-VALUE', 'B-VALUE', 'I-WHOLE', 'B-WHOLE', 'I-LOCATION', 'B-LOCATION', 'I-AGENT', 'B-AGENT', 'I-CAUSE', 'B-CAUSE', 'I-SOURCE', 'B-SOURCE', 'I-REF_TIME', 'B-REF_TIME', 'I-CONDITION', 'B-CONDITION'])
        f1 = f1_score(y_true, y_pred, average='macro', labels=['O', 'I-QUANT', 'B-QUANT', 'I-TIME', 'B-TIME', 'I-MANNER', 'B-MANNER', 'I-THEME', 'B-THEME', 'I-VALUE', 'B-VALUE', 'I-WHOLE', 'B-WHOLE', 'I-LOCATION', 'B-LOCATION', 'I-AGENT', 'B-AGENT', 'I-CAUSE', 'B-CAUSE', 'I-SOURCE', 'B-SOURCE', 'I-REF_TIME', 'B-REF_TIME', 'I-CONDITION', 'B-CONDITION'])
    else:
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
    # # Clear cache after successful completion
    # for f in os.listdir(cache_folder):
    #     os.remove(os.path.join(cache_folder, f))
    # os.rmdir(cache_folder)
    
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    # path_li = ["../benchmark/FINQA/flare-cfa-test.json", "../benchmark/FINQA/flare-finqa_test.json", "../benchmark/FINQA/flare-mlesg.json"]
    # test_data_path = path_li[1]
    test_data_path = "../benchmark/FINQA/FSRL.json"
    # output_csv_path = "../results/Mistral-v0.2/SA/finqa_eval_results.csv"
    # output_txt_path = "../results/Mistral-v0.2/SA/finqa_eval_results.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1,
        'max_new_tokens': 256,
    }
    results, df = evaluate_finqa(model_path, test_data_path, args)
    # df.to_csv(output_csv_path, index=False)    
    # with open(output_txt_path, 'w') as f:
    #     f.write(f"EmAccuracy: {results['accuracy']}\n")
    #     f.write(f"Precision: {results['precision']}\n")
    #     f.write(f"Recall: {results['recall']}\n")
    #     f.write(f"F1 Score: {results['f1_score']}\n")
    # print(results)
    
    # final_result = {}
    # for path in path_li:
    #     test_data_path = path
    #     results = evaluate_finqa(model_path, test_data_path, args)
    #     print(results)
    #     final_result[path] = results
    # print(final_result)
    
if __name__ == "__main__":
    main()

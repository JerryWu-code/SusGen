# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_ner.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
from tqdm import tqdm
import pandas as pd
from collections import Counter
import random
from batch_inference import load_model, inference


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_ner(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=100, batch_size=100, model=None):
    # define a function to check correctness
    def check_correctness(batch_samples, answers, prompts):
        y_true = []
        y_pred = []
        eval_results = []
        for sample, answer, final_prompt in zip(batch_samples, answers, prompts):
            # if for new test set
            if test_data_path == "../benchmark/NER/FINER-ORD.json":
                true_entities = [line.split(':') for line in sample['output'].split('\n')]
                predicted_entities = [line.split(':') for line in answer.split('\n') if ':' in line]
                
                true_labels = [entity[1] for entity in true_entities]
                predicted_labels = [entity[1] for entity in predicted_entities[:len(true_labels)]]
                # Ensure the lengths are the same
                min_len = min(len(true_labels), len(predicted_labels))
                true_labels = true_labels[:min_len]
                predicted_labels = predicted_labels[:min_len]
                # print('='*50)
                # print(true_labels)
                # print(predicted_labels)
                
                y_true.extend(true_labels)
                y_pred.extend(predicted_labels)
            else:
                # Split the expected output and the generated answer by comma and newline
                pattern = re.compile(r'[.,;:!?]\s*|\n')
                #true_entities = [entity.strip() for entity in pattern.split(sample['output']) if entity.strip()]
                true_entities = re.sub(r"[^\w\s']", ' ', sample['output']).lower().split()
                predicted_entities = re.sub(r"[^\w\s]", ' ', answer).lower().split()
                # print('='*50)
                # print(true_entities)
                # print(predicted_entities)
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
                # print(is_correct)
                y_true.append(1)
                y_pred.append(1 if is_correct else 0)
            
            if test_data_path == "../benchmark/NER/FINER-ORD.json":
                eval_results.append({
                'prompt': final_prompt,
                'generated': answer,
                'target': sample['output'],
                'true_labels': true_labels,
                'predicted_labels': predicted_labels
                })
            else:
                eval_results.append({
                    'prompt': final_prompt,
                    'generated': answer,
                    'target': sample['output'],
                    'is_correct': 'Yes' if y_pred[-1] else 'No'
                })
        return y_true, y_pred, eval_results

    # Load the model and tokenizer
    # model, tokenizer, device, _ = load_model(model_path, lora_path, quantization)
    # model = load_model(model_path)
    
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
        
    # y_true = []
    # y_pred = []   
    # eval_results = []
    
    # count = 0
    # Generate predictions
    # for idx, sample in enumerate(tqdm(test_data[start_index:], initial=start_index, total=len(test_data))):
    #     current_index = start_index + idx
    #     # if count > 50:
    #     #     break
    #     # count += 1
    #     # prompt = "Please strictly format your answer with instructions:" + sample['instruction'] + '\n\n' + sample['input']
    #     # final_prompt = instr_prompt(content=prompt)
    #     if prompt_type == 'mistral':
    #         final_prompt = mistral_formal_infer(sample)
    #     elif prompt_type == 'llama3':
    #         final_prompt = llama3_formal_infer(sample)
    #     else:
    #         raise ValueError("Invalid prompt type")
    for batch_start in tqdm(range(start_index, len(test_data), batch_size), initial=start_index, total= len(test_data) // batch_size + 1):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_samples = test_data[batch_start:batch_end]
        prompts = []
        for sample in batch_samples:
            if prompt_type == 'mistral':
                prompts.append(mistral_formal_infer(sample))
            elif prompt_type == 'llama3':
                prompts.append(llama3_formal_infer(sample))
            else:
                raise ValueError("Invalid prompt type")
        answers = inference(model, prompts, args, verbose=False)
        # _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
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
        # # Save to cache every 100 rows
        # if (current_index + 1) % 100 == 0 or (current_index + 1) == len(test_data):
        #     cache_data = {
        #         'y_true': y_true,
        #         'y_pred': y_pred,
        #         'eval_results': eval_results
        #     }
        #     with open(os.path.join(cache_folder, f'cache_{current_index}.json'), 'w') as f:
        #         json.dump(cache_data, f)
                
    df = pd.DataFrame(eval_results)
    # df.to_csv(output_csv_path, index=False)
    

    # Calculate evaluation metrics
    if test_data_path == "../benchmark/NER/FINER-ORD.json":
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', labels=['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'])
        recall = recall_score(y_true, y_pred, average='macro', labels=['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'])
        f1 = f1_score(y_true, y_pred, average='macro', labels=['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG'])
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'EntityF1_score': f1
    }
    
    # # Clear cache after successful completion
    # for f in os.listdir(cache_folder):
    #     os.remove(os.path.join(cache_folder, f))
    # os.rmdir(cache_folder)
            
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    # test_data_path = "../benchmark/NER/NER.json"
    test_data_path = "../benchmark/NER/FINER-ORD.json"

    # output_csv_path = "../results/Mistral-v0.2/NER/ner_eval_results.csv"
    # output_txt_path = "../results/Mistral-v0.2/NER/ner_eval_results.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1,
        "max_new_tokens": 256
    }

    results, df = evaluate_ner(model_path, test_data_path, args)
    # df.to_csv(output_csv_path, index=False)
    # with open(output_txt_path, 'w') as f:
    #     for key, value in results.items():
    #         f.write(f"{key}: {value}\n")
    # print(results)

if __name__ == "__main__":
    main()

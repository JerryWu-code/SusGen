# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_sa.py
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

def evaluate_sa(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=100, batch_size=100, model=None):
    # define a function to check correctness
    def check_correctness(batch_samples, answers, prompts):
        y_true = []
        y_pred = []
        eval_results = []
        for sample, answer, final_prompt in zip(batch_samples, answers, prompts):
            # Normalize the outputs for case insensitivity
            true_sentiment = sample['output'].strip().lower()
            predicted_sentiment = re.sub(r"[^\w\s]", ' ', answer).lower().split()
            # predicted_sentiment
            sentiment_counts = Counter(predicted_sentiment)
            
            # print('='*50)
            # print(true_sentiment)
            # print(predicted_sentiment)            
            most_common_sentiment = None

            # new condition for hawkish, dovish, neutral
            if true_sentiment in ['hawkish', 'dovish', 'neutral']:
                # print(f"Hawkish: {sentiment_counts['hawkish']}, Dovish: {sentiment_counts['dovish']}, Neutral: {sentiment_counts['neutral']}")
                if sentiment_counts['hawkish'] == sentiment_counts['dovish'] == sentiment_counts['neutral']:
                    for word in predicted_sentiment:
                        if word in ['hawkish', 'dovish', 'neutral']:
                            most_common_sentiment = word
                            break
                else:
                    most_common_sentiment = max(['hawkish', 'dovish', 'neutral'], key=lambda x: sentiment_counts[x])
            else:
                # print(f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Neutral: {sentiment_counts['neutral']}")
                if sentiment_counts['positive'] == sentiment_counts['negative'] == sentiment_counts['neutral']:
                    for word in predicted_sentiment:
                        if word in ['positive', 'negative', 'neutral']:
                            most_common_sentiment = word
                            break
                else:
                    most_common_sentiment = max(['positive', 'negative', 'neutral'], key=lambda x: sentiment_counts[x])
            print(f"Most common sentiment: {most_common_sentiment}")
            y_true.append(1)
            y_pred.append(1 if true_sentiment == most_common_sentiment else 0)
            
            eval_results.append({
                'prompt': final_prompt,
                'generated': answer,
                'target': sample['output'],
                'is_correct': 'Yes' if true_sentiment == most_common_sentiment else 'No'
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
    #     # if count > 10:
    #     #     break
    #     # count += 1
    #     # prompt = sample['instruction'] + '\n\n' + sample['input']
    #     # final_prompt = instr_prompt(content=prompt)
    #     if prompt_type == 'mistral':
    #         final_prompt = mistral_formal_infer(sample)
    #     elif prompt_type == 'llama3':
    #         final_prompt = llama3_formal_infer(sample)
    #     else:
    #         raise ValueError("Invalid prompt type")
        
    #     _, answer = generate_text(model, tokenizer, device, final_prompt, args)
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
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
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
    # test_data_path = "../benchmark/SA/FiQA-SA.json"
    test_data_path = "../benchmark/SA/FOMC.json"
    
    output_csv_path = "../results/Mistral-v0.2/SA/sa_eval_results_FOMC.csv"
    output_txt_path = "../results/Mistral-v0.2/SA/sa_eval_results_FOMC.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results, df = evaluate_sa(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)
    with open(output_txt_path, 'w') as f:
        f.write(f"Accuracy: {results['accuracy']}\n")
        f.write(f"Precision: {results['precision']}\n")
        f.write(f"Recall: {results['recall']}\n")
        f.write(f"F1 Score: {results['f1_score']}\n")
    # print(results)

if __name__ == "__main__":
    main()

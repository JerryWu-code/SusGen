# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_sum.py
import json, sys, os
sys.path.append("/home/whatx/SusGen/src/")
# from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
import pandas as pd
from tqdm import tqdm
import evaluate
from bert_score import score as bert_score
import random
from batch_inference import load_model, inference
import torch

# from bart_score import BARTScorer

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_bertscore_cpu(y_pred, y_true, batch_size=16):
    device = torch.device('cpu')
    bertscore_f1 = 0.0
    for i in range(0, len(y_pred), batch_size):
        batch_preds = y_pred[i:i + batch_size]
        batch_refs = y_true[i:i + batch_size]
        bertscore_results = bert_score(batch_preds, batch_refs, lang='en', model_type="microsoft/deberta-v3-base", device=device, rescale_with_baseline=True)
        bertscore_f1 += bertscore_results[2].mean().item() * len(batch_preds) / len(y_pred)
    return bertscore_f1

def evaluate_summarization(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=20, batch_size=32, model=None):
    # random_count = 256
    batch_size = 16
    # define a function to check correctness
    def check_correctness(batch_samples, answers, prompts):
        y_true = []
        y_pred = []
        eval_results = []
        for sample, answer, final_prompt in zip(batch_samples, answers, prompts):
            y_pred.append(answer)
            y_true.append(sample['output'])
            
            eval_results.append({
                'prompt': final_prompt,
                'generated': answer,
                'target': sample['output']
            })
        return y_true, y_pred, eval_results
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

    # for idx, sample in enumerate(tqdm(test_data[start_index:], initial=start_index, total=len(test_data))):
    #     current_index = start_index + idx
    #     # if count > 50:
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
        torch.cuda.empty_cache()
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
    
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=y_pred, references=y_true)

    # bertscore_results = bert_score(y_pred, y_true, lang='en', model_type="microsoft/deberta-v3-base", rescale_with_baseline=True)
    # bertscore_f1 = bertscore_results[2].mean().item()
    bertscore_f1 = calculate_bertscore_cpu(y_pred, y_true)

    # bart_scorer = BARTScorer(device=device, checkpoint="facebook/bart-large-cnn")
    # bart_scorer.load(path="path_to_bart_score_weights/bart_score.pth")
    # bartscore_results = bart_scorer.score(predictions, references, batch_size=8)
    # bartscore_mean = sum(bartscore_results) / len(bartscore_results)

    results = {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bertscore": bertscore_f1,
        # "bartscore": bartscore_mean,
    }
    
    # # Clear cache after successful completion
    # for f in os.listdir(cache_folder):
    #     os.remove(os.path.join(cache_folder, f))
    # os.rmdir(cache_folder)
    
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/SUM/EDTSum.json"
    output_csv_path = "../results/Mistral-v0.2/SUM/sum_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/SUM/sum_eval_results.txt"
    args = {
        "max_length": 8192,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1,
        "max_new_tokens": 512
    }

    results, df = evaluate_summarization(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)
    with open(output_txt_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    # print(results)

if __name__ == "__main__":
    main()


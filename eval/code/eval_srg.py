import json
import sys, os
sys.path.append("/home/whatx/SusGen/src/")
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import evaluate
from bert_score import score as bert_score
import pandas as pd
# from template import load_model, generate_text, instr_prompt
from batch_inference import load_model, inference
import torch
from torch.cuda.amp import autocast

rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
bleu = evaluate.load('bleu')

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def flatten_json_data(json_data):
    flat_samples = []
    for document in json_data:
        file_name = document.get("file", "unknown_file")
        for section, entries in document['content'].items():
            for entry in entries:
                flat_sample = {
                    "file": file_name,
                    "section": section,
                    "instruction": entry.get("instruction", ""),
                    "input": entry.get("input", ""),
                    "context": entry.get("context", ""),
                    "output": entry.get("output", "")
                }
                # if flat_sample['context']:
                #     flat_sample['prompt'] = f"{flat_sample['instruction']}\n\nContext:\n{flat_sample['context']}\n\nQuestion\n{flat_sample['input']}"
                # else:
                #     flat_sample['prompt'] = f"{flat_sample['instruction']}\n\nQuestion\n{flat_sample['input']}"

                flat_samples.append(flat_sample)
    return flat_samples

def mistral_construct_prompt(record):
    if not record["input"]:
        text = (
            "[INST]"
            "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
            "### Instruction:\n{0}\n\nContext:\n{1}\n\n[/INST]### Response:\n"
        ).format(record["instruction"], record["context"])
    else:
        text = (
            "[INST]"
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\nContext:\n{1}\n\n### Input:\n{2}\n\n[/INST]### Response:\n"
        ).format(record["instruction"], record["context"], record["input"])
    return text    

def llama3_construct_prompt(record):
    if not record["input"]:
        text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
            "### Instruction:\n{0}\n\nContext:\n{1}\n\n### Response:\n"
        ).format(record["instruction"], record["context"])
    else:
        text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\nContext:\n{1}\n\n### Input:\n{2}\n\n### Response:\n"
        ).format(record["instruction"], record["context"], record["input"])
    return text

def calculate_bertscore_cpu(y_pred, y_true, batch_size=16):
    device = torch.device('cpu')
    bertscore_f1 = 0.0
    for i in range(0, len(y_pred), batch_size):
        batch_preds = y_pred[i:i + batch_size]
        batch_refs = y_true[i:i + batch_size]
        bertscore_results = bert_score(batch_preds, batch_refs, lang='en', model_type="distilbert-base-uncased", device=device, rescale_with_baseline=True)
        bertscore_f1 += bertscore_results[2].mean().item() * len(batch_preds) / len(y_pred)
    return bertscore_f1

def evaluate_srg(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=1, batch_size=32, model=None):
    batch_size = 16
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
    
    # model = load_model(model_path)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    test_data = flatten_json_data(test_data)
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
    
    # predictions = []
    # references = []
    # eval_results = []
    
    # for document in test_data:
    #     for category, samples in document['content'].items():
    #         for sample in tqdm(samples):
    #             if prompt_type == 'mistral':
    #                 prompt = mistral_construct_prompt(sample)
    #             elif prompt_type == 'llama3':
    #                 prompt = llama3_construct_prompt(sample)
    #             _, generated_text = generate_text(model, tokenizer, device, prompt, args)
    #             generated_text = generated_text.strip()

    #             predictions.append(generated_text)
    #             references.append(sample['output'])

    #             eval_results.append({
    #                 'prompt': prompt,
    #                 'generated': generated_text,
    #                 'target': sample['output']
    #             })
    for batch_start in tqdm(range(start_index, len(test_data), batch_size), initial=start_index, total=len(test_data) // batch_size + 1):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_samples = test_data[batch_start:batch_end]
        prompts = []
        # for document in test_data[batch_start:batch_end]:
        #     for category, samples in document['content'].items():
        #         for sample in samples:
        #             if prompt_type == 'mistral':
        #                 prompts.append(mistral_construct_prompt(sample))
        #             elif prompt_type == 'llama3':
        #                 prompts.append(llama3_construct_prompt(sample))
        #             else:
        #                 raise ValueError("Invalid prompt type")
        #             batch_samples.append(sample)
        for sample in batch_samples:
            if prompt_type == 'mistral':
                prompts.append(mistral_construct_prompt(sample))
            elif prompt_type == 'llama3':
                prompts.append(llama3_construct_prompt(sample))
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
            
    df = pd.DataFrame(eval_results)
    
    rouge_results = rouge.compute(predictions=y_pred, references=y_true)
    # bertscore_results = bert_score(y_pred, y_true, lang='en', model_type="microsoft/deberta-v3-base", rescale_with_baseline=True)
    # bertscore_results = bert_score(y_pred, y_true, lang='en', model_type="distilbert-base-uncased", rescale_with_baseline=True)
    # bertscore_f1 = bertscore_results[2].mean().item()
    bertscore_f1 = calculate_bertscore_cpu(y_pred, y_true)
    meteor_results = meteor.compute(predictions=y_pred, references=y_true)
    bleu_results = bleu.compute(predictions=y_pred, references=y_true)
    bleu_1 = bleu_results['precisions'][0]
    bleu_2 = bleu_results['precisions'][1]
    bleu_3 = bleu_results['precisions'][2]
    bleu_4 = bleu_results['precisions'][3]
    
    # batch_size_evaluation = 16
    # rouge_results = {"rougeL": {"fmeasure": 0.0, "precision": 0.0, "recall": 0.0}}
    # bertscore_f1 = 0.0
    # meteor_score = 0.0
    # bleu_1, bleu_2, bleu_3, bleu_4 = 0.0, 0.0, 0.0, 0.0

    # for i in range(0, len(y_pred), batch_size_evaluation):
    #     batch_preds = y_pred[i:i + batch_size_evaluation]
    #     batch_refs = y_true[i:i + batch_size_evaluation]
        
    #     rouge_result_batch = rouge.compute(predictions=batch_preds, references=batch_refs)
    #     bertscore_result_batch = bert_score(batch_preds, batch_refs, lang='en', model_type="distilbert-base-uncased", rescale_with_baseline=True)
    #     meteor_result_batch = meteor.compute(predictions=batch_preds, references=batch_refs)
    #     bleu_result_batch = bleu.compute(predictions=batch_preds, references=[[ref] for ref in batch_refs])
        
    #     rouge_results["rougeL"]["fmeasure"] += rouge_result_batch["rougeL"]["fmeasure"] * len(batch_preds) / len(y_pred)
    #     rouge_results["rougeL"]["precision"] += rouge_result_batch["rougeL"]["precision"] * len(batch_preds) / len(y_pred)
    #     rouge_results["rougeL"]["recall"] += rouge_result_batch["rougeL"]["recall"] * len(batch_preds) / len(y_pred)
        
    #     bertscore_f1 += bertscore_result_batch[2].mean().item() * len(batch_preds) / len(y_pred)
    #     meteor_score += meteor_result_batch["meteor"] * len(batch_preds) / len(y_pred)
        
    #     bleu_1 += bleu_result_batch['precisions'][0] * len(batch_preds) / len(y_pred)
    #     bleu_2 += bleu_result_batch['precisions'][1] * len(batch_preds) / len(y_pred)
    #     bleu_3 += bleu_result_batch['precisions'][2] * len(batch_preds) / len(y_pred)
    #     bleu_4 += bleu_result_batch['precisions'][3] * len(batch_preds) / len(y_pred)
    
    results = {
        "rougeL": rouge_results["rougeL"],
        "bertscore": bertscore_f1,
        "meteor": meteor_results["meteor"],
        "bleu-1": bleu_1,
        "bleu-2": bleu_2,
        "bleu-3": bleu_3,
        "bleu-4": bleu_4
    }

    # results = {
    #     "rougeL": rouge_results["rougeL"],
    #     "bertscore": bertscore_f1,
    #     "meteor": meteor_score,
    #     "bleu-1": bleu_1,
    #     "bleu-2": bleu_2,
    #     "bleu-3": bleu_3,
    #     "bleu-4": bleu_4
    # }
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/SRG/Annual.json"
    output_csv_path = "../results/Mistral-v0.2/SRG/srg_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/SRG/srg_eval_results.txt"
    args = {
        "max_length": 8192,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1,
        "max_new_tokens": 512
    }

    results, df = evaluate_srg(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)
    with open(output_txt_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    # print(results)

if __name__ == "__main__":
    main()

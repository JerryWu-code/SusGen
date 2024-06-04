import json, sys, os
sys.path.append("/home/whatx/SusGen/src/")
from template import load_model, generate_text, instr_prompt
from utils.prompt_template import mistral_formal_infer, llama3_formal_infer
import pandas as pd
from tqdm import tqdm
import evaluate
from bert_score import score as bert_score
import random
# from bart_score import BARTScorer

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_summarization(model_path, test_data_path, args, prompt_type='mistral', lora_path=False, quantization='int4', random_count=10):
    model, tokenizer, device, _ = load_model(model_path, lora_path, quantization)
    
    test_data = load_json(test_data_path)
    random.seed(42)  # For reproducibility
    test_data = random.sample(test_data, random_count)
    predictions = []
    references = []
    eval_results = []
    # count = 0

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
        
        _, summary = generate_text(model, tokenizer, device, final_prompt, args)
        
        predictions.append(summary)
        references.append(sample['output'])
        
        eval_results.append({
            'prompt': final_prompt,
            'generated': summary,
            'target': sample['output']
        })
    
    df = pd.DataFrame(eval_results)
    # df.to_csv(output_csv_path, index=False)
    
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)

    bertscore_results = bert_score(predictions, references, lang='en', model_type="allenai/longformer-large-4096-finetuned-triviaqa", rescale_with_baseline=True)
    bertscore_f1 = bertscore_results[2].mean().item()

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
    
    # with open(output_txt_path, 'w') as f:
    #     for key, value in results.items():
    #         f.write(f"{key}: {value}\n")
    return results, df

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/SUM/EDTSum.json"
    output_csv_path = "../results/Mistral-v0.2/SUM/sum_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/SUM/sum_eval_results.txt"
    args = {
        "max_length": 4096,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results, df = evaluate_summarization(model_path, test_data_path, args)
    df.to_csv(output_csv_path, index=False)
    with open(output_txt_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print(results)

if __name__ == "__main__":
    main()


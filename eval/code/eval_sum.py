import json, sys, os
sys.path.append("/home/whatx/SusGen/src/llms/mistral-hf")
from template import load_model, generate_text, instr_prompt
from tqdm import tqdm
import evaluate
from bert_score import score as bert_score
# from bart_score import BARTScorer

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_summarization(model_path, test_data_path, args):
    model, tokenizer, device, _ = load_model(model_path)
    
    test_data = load_json(test_data_path)
    
    predictions = []
    references = []
    count = 0

    for sample in tqdm(test_data):
        if count > 10:
            break
        count += 1
        prompt = sample['instruction'] + '\n\n' + sample['input']
        final_prompt = instr_prompt(content=prompt)
        
        _, summary = generate_text(model, tokenizer, device, final_prompt, args)
        
        predictions.append(summary)
        references.append(sample['output'])
    
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)

    bertscore_results = bert_score(predictions, references, lang='en', rescale_with_baseline=True)
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
    
    return results

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/SUM/flare-edtsum_test.json"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results = evaluate_summarization(model_path, test_data_path, args)
    print(results)

if __name__ == "__main__":
    main()

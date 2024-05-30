# cd /home/whatx/SusGen/eval/code && CUDA_VISIBLE_DEVICES=1 python eval_sa.py
import json, sys, os, re
sys.path.append("/home/whatx/SusGen/src/llms/mistral-hf")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from tqdm import tqdm
import pandas as pd
from collections import Counter

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_sa(model_path, test_data_path, args, output_csv_path, output_txt_path):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    
    y_true = []
    y_pred = []
    eval_results = []
    count = 0
    # Generate predictions
    for sample in tqdm(test_data):
        if count > 100:
            break
        count += 1
        prompt = sample['instruction'] + '\n\n' + sample['input']
        final_prompt = instr_prompt(content=prompt)
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
        # Normalize the outputs for case insensitivity
        true_sentiment = sample['output'].strip().lower()
        predicted_sentiment = re.sub(r"[^\w\s]", ' ', answer).lower().split()
        # count number of positive, negative, neutral in predicted_sentiment
        sentiment_counts = Counter(predicted_sentiment)
        
        print('='*50)
        print(true_sentiment)
        print(predicted_sentiment)
        print(f"Positive: {sentiment_counts['positive']}, Negative: {sentiment_counts['negative']}, Neutral: {sentiment_counts['neutral']}")
        most_common_sentiment = None
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
            'prompt': prompt,
            'generated': answer,
            'target': sample['output'],
            'is_correct': 'Yes' if true_sentiment == most_common_sentiment else 'No'
        })
        
    df = pd.DataFrame(eval_results)
    df.to_csv(output_csv_path, index=False)

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
    
    with open(output_txt_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    return results

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../benchmark/SA/flare-fiqasa_test.json"
    output_csv_path = "../results/Mistral-v0.2/SA/sa_eval_results.csv"
    output_txt_path = "../results/Mistral-v0.2/SA/sa_eval_results.txt"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results = evaluate_sa(model_path, test_data_path, args, output_csv_path, output_txt_path)
    print(results)

if __name__ == "__main__":
    main()

import json, sys, os
sys.path.append("/home/whatx/SusGen/src/llms/mistral-hf")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_sa(model_path, test_data_path, args):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    
    y_true = []
    y_pred = []

    # Generate predictions
    for sample in tqdm(test_data):
        prompt = sample['instruction'] + '\n\n' + sample['input']
        final_prompt = instr_prompt(content=prompt)
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
        # Normalize the outputs for case insensitivity
        true_sentiment = sample['output'].strip().lower()
        predicted_sentiment = answer.strip().lower()
        
        y_true.append(1)
        y_pred.append(1 if true_sentiment in predicted_sentiment else 0)

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
    
    return results

def main():
    model_path = "../../ckpts/Mistral-7B-Instruct-v0.3-hf"
    test_data_path = "../benchmark/SA/esg_sentiment_test_final.json"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results = evaluate_sa(model_path, test_data_path, args)
    print(results)

if __name__ == "__main__":
    main()

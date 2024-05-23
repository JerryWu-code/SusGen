import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from template import load_model, generate_text, instr_prompt
from tqdm import tqdm

ent_dict = {
    'PER': 'person',
    'ORG': 'organization',
    'LOC': 'location',
}
ent_dict_rev = {v: k for k, v in ent_dict.items()}

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def evaluate_re(model_path, test_data_path, args):
    # Load the model and tokenizer
    model, tokenizer, device, _ = load_model(model_path)
    
    # Load the test dataset
    test_data = load_json(test_data_path)
    
    y_true = []
    y_pred = []
    count = 0
    # Generate predictions
    for sample in tqdm(test_data):
        if count > 5:
            break
        count += 1
        prompt = sample['instruction'] + '\n\n' + sample['input']
        final_prompt = instr_prompt(content=prompt)
        
        _, answer = generate_text(model, tokenizer, device, final_prompt, args)
        
        # Split the expected output and the generated answer
        pattern = re.compile(r'[.,;:!?]\s*|\n')
        true_entities = [entity.strip() for entity in pattern.split(sample['output']) if entity.strip()]
        predicted_entities = answer.split()
        
        true_iter = iter(true_entities)
        is_correct = all(entity in true_iter for entity in predicted_entities)

        y_true.append(1)
        y_pred.append(1 if is_correct else 0)

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
    
    return results

def main():
    model_path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    test_data_path = "../../../eval/benchmark/RE/fingpt-finred-cls_test.json"
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    results = evaluate_re(model_path, test_data_path, args)
    print(results)

if __name__ == "__main__":
    main()

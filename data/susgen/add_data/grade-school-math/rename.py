import json

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            l = json.loads(line)
            question = l['question'].split(". ")[-1]
            context = l['question'].split(question)[0].strip()
            data.append({
                "instruction": "Read the following texts carefully. Answer the last question with reasoning and mathematical calculation"data/susgen/FINAL/PER_3500/train.json".",
                "input": l['question'],
                "output": l['answer']
            })
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

data = load_jsonl("train.jsonl")
save_json(data, "train.json")
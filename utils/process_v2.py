# cd /home/whatx/SusGen/utils/ && CUDA_VISIBLE_DEVICES=0 python processv2.py

import json, os, sys, time
from tqdm import tqdm
sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from template import load_model, generate_text, instr_prompt
import random

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)
    
def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def append_to_jsonl(data, file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            pass
    if data:
        with open(file_path, 'a', encoding='utf-8') as file:
            json.dump(data, file)
            file.write('\n')

def concat_jsonl(target_path, output_file="../data/susgen/tcfd_qa/v2_concat.json"):
    file_list = [os.path.join(target_path, file) for file in os.listdir(
        target_path) if file.endswith(".jsonl")]
    final = []
    num = 0
    for file in file_list:
        data_ = load_jsonl(file)
        for data in data_:
            temp = {
                "instruction": data['instruction'],
                "input": data['input'],
                "output": data['answer']
            }
            final.append(temp)
            num += 1
    print(f"Total records: {num}")
    save_json(final, output_file)

def denoise_text():
    data = load_json("../data/susgen/tcfd_qa/tcfd_qa_v4.json")
    final = []
    noise = 0
    for _, v in tqdm(enumerate(data)):
        out = v['answer']
        # if out includes "\n\n", replace it with "\n"
        if "\n\n" in out:
            out = out.replace("\n\n", "\n")
            noise += 1
        temp = {
            "instruction": v['instruction'],
            "input": v['input'],
            "output": out
        }
        final.append(temp)
    print(f"Total noise removed: {noise}")
    save_json(final, "../data/susgen/tcfd_qa/tcfd_qa_v4_clean.json")

def task1(model, tokenizer, device, args):
    data = load_json("../data/susgen/tcfd_qa/tcfd_qa_v3.json")
    print(f"Total samples: {len(data)}")
    final = []
    for num, v in tqdm(enumerate(data)):
        instr = v['instruction']
        out = v['output']
        prompt = instr_prompt(
            "Process the text following the instructions below:\n"
            "1.Replace all the specific company entity name with \"we\" or \"our company\"\n"
            "2.Replace other private information with generic terms\n"
            f"Text:\n{out}")
        _, answer = generate_text(model, tokenizer, device, prompt=prompt, args=args)
        temp = {
            "instruction": instr,
            "input": "",
            "answer": answer
        }
        append_to_jsonl(temp, "../data/susgen/tcfd_qa/cache/de_privacy.jsonl")
        final.append(temp)
        print("="*50)
        print(f"Processed {num} samples.")
        print("="*50)
    save_json(final, "../data/susgen/tcfd_qa/tcfd_qa_v1.json")
    print('Task 1 completed.')

def task2(model, tokenizer, device, args, times):
    data = load_json("../data/susgen/tcfd_qa/history/tcfd_qa_v4_clean.json")
    print(f"Total new records will be created: {len(data) * times}")
    prompt1_versions = [
        "Rephrase this text carefully without altering its original meaning or elements, adjust to the original length, and ensure it is coherent and fluent:",
        "Redraft the text below, maintaining the initial meaning and structure, matching the original length, and ensuring clear and fluent language:",
        "Transform the text by rephrasing it to preserve the original intent and elements, match its length, and maintain coherence and fluency:",
        "Revise the text to keep the core meaning and elements intact, ensure it matches the original length, and enhance readability and flow:",
        "Modify the text to reflect the same meaning and elements without changing its length, and improve its coherence and readability:",
        "Reconstruct the text to retain its original meaning and details, ensure the length is unchanged, and guarantee coherence and fluency:",
        "Rework the text to maintain all original concepts and length, ensuring it is presented coherently and fluently:",
        "Edit the text to preserve its fundamental meaning and components, maintain the same length, and improve textual coherence and fluency:",
        "Adapt the text while keeping the original meaning and structural elements, ensure it's of similar length, and remains coherent and fluent:",
        "Refashion the text to hold the original intent and details, align the length with the original, and secure clarity and fluency:"
    ]   
    prompt2_versions = [
        "Revise the text below by substituting company names with 'we' or 'our company' and replace other private information with generic terms:",
        "Alter the text to replace specific company names with 'we' or 'our company' and anonymize other sensitive details using generic terms:",
        "Modify the text by replacing all specific entity names with 'we' or 'our company' and other private details with non-specific terms:",
        "Transform the text, switching company names to 'we' or 'our company' and substituting private information with general terms:",
        "Edit the text to replace specific business names with 'we' or 'our company' and other personal identifiers with generic labels:",
        "Update the text by changing any specific company names to 'we' or 'our company' and using generic terms for other private data:",
        "Reconfigure the text, replacing all mentions of company names with 'we' or 'our company' and other private information with generic terms:",
        "Revise the text to use 'we' or 'our company' instead of specific entity names and replace other private specifics with general terms:",
        "Refine the text by substituting any specific company names with 'we' or 'our company' and changing private data to generic terms:",
        "Rework the text to remove specific company names, using 'we' or 'our company' instead, and anonymize other private information with generic terms:"
    ]
    version = 0
    state = 0
    final = []
    for _ in range(times):
        for _, v in tqdm(enumerate(data)):
            instr = v['instruction']
            out = v['output']
            # random choice prompt
            prompt1_text = f"{random.choice(prompt1_versions)}\nText:\n{instr}"
            prompt2_text = f"{random.choice(prompt2_versions)}\nText:\n{out}"
            prompt1 = instr_prompt(prompt1_text)
            prompt2 = instr_prompt(prompt2_text)

            _, answer1 = generate_text(model, tokenizer, device, prompt=prompt1, args=args)
            _, answer2 = generate_text(model, tokenizer, device, prompt=prompt2, args=args)
            temp = {
                "instruction": answer1,
                "input": v['input'],
                "answer": answer2
            }
            append_to_jsonl(temp, "../data/susgen/tcfd_qa/cache/v4/diverse_v4_8.jsonl")
            final.append(temp)
            print("="*50)
            state += 1
            print(f"Processed {state} samples.")
            print("="*50)
            data.append(temp)
    print('Task 2 completed.')
    # save_json(final, "../data/susgen/tcfd_qa/cache/tcfd_qa_v2_+4.json")

def main():
    args = {
        "max_length": 8096,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }

    # model, tokenizer, device, _ = load_model()
    # no need for task 1 in v2
    # task1(model, tokenizer, device, args)
    # denoise_text()
    # task2(model, tokenizer, device, args, 1)
    target_path = "../data/susgen/tcfd_qa/cache/v4"
    concat_jsonl(target_path, output_file="../data/susgen/tcfd_qa/tcfd_qa_v4_final_.json")

if __name__ == "__main__":
    main()
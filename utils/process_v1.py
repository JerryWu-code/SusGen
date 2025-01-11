# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
#    1. Process the tcfd_qa dataset to remove private information.
#    2. Contruct additive diversity version of the dataset.

#############################################################################
# cd /home/whatx/SusGen/utils/ && CUDA_VISIBLE_DEVICES=0 python process.py
import json, os, sys, time
from tqdm import tqdm
sys.path.append((os.path.dirname(os.path.dirname(__file__))))
from template import load_model, generate_text, instr_prompt

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
    data = load_json("../data/susgen/tcfd_qa/tcfd_qa_v1.json")
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
            "input": "",
            "output": out
        }
        final.append(temp)
    print(f"Total noise removed: {noise}")
    save_json(final, "../data/susgen/tcfd_qa/tcfd_qa_v1_clean.json")

def task1(model, tokenizer, device, args):
    data = load_json("../data/susgen/tcfd_qa/history/qa_dict_formatted1.json")
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

def task2(model, tokenizer, device, args, times):
    data = load_json("../data/susgen/tcfd_qa/tcfd_qa_v1_clean.json")
    print(f"Total new records will be created: {len(data) * times}")
    state = 0
    final = []
    for _ in range(times):
        for _, v in tqdm(enumerate(data)):
            instr = v['instruction']
            out = v['output']
            prompt1 = instr_prompt(
                "Process the text follwing the instructions below:\n"
                "1.Rephrase the whole text without change original meaning and elements.\n"
                "2.Adjust the processed text to similar length as the original text.\n"
                "3.Ensure the text is coherent and fluent and output the final text.\n"
                f"Text:\n{instr}")
            prompt2 = instr_prompt(
                "Process the text following the instructions below:\n"
                "1.Replace all the specific company entity name with \"we\" or \"our company\"\n"
                "2.Replace other private information with generic terms\n"
                f"Text:\n{out}")
            _, answer1 = generate_text(model, tokenizer, device, prompt=prompt1, args=args)
            _, answer2 = generate_text(model, tokenizer, device, prompt=prompt2, args=args)
            temp = {
                "instruction": answer1,
                "input": "",
                "answer": answer2
            }
            append_to_jsonl(temp, "../data/susgen/tcfd_qa/cache/diverse_v2_+4.jsonl")
            final.append(temp)
            print("="*50)
            state += 1
            print(f"Processed {state} samples.")
            print("="*50)
            data.append(temp)
    
    # save_json(final, "../data/susgen/tcfd_qa/tcfd_qa_v2_+4.json")

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
    # task1(model, tokenizer, device, args)
    # denoise_text()
    # task2(model, tokenizer, device, args, 1)
    target_path = "../data/susgen/tcfd_qa/cache"
    concat_jsonl(target_path, output_file="../data/susgen/tcfd_qa/tcfd_qa_v2.json")

if __name__ == "__main__":
    main()
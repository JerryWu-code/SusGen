# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to balance the alpaca_gpt4 dataset for using in the training of mistral

#############################################################################
import json, random

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

# shuffle and randomly sample from the json file
def sample_json(file, target_num=2000):
    data = load_json(file)
    data = random.sample(data, target_num)
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    target_num = 2000
    file = "alpaca_data_gpt4.json"
    out_file = f"alpaca_gpt4_{target_num // 1000}k.json"
    data = sample_json(file, target_num=target_num)
    save_json(data, out_file)

if __name__ == "__main__":
    main()
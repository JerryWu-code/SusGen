# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to balance the fingpt dataset for using in the training of mistral

#############################################################################
import json, os
import pandas as pd

def merge_json(target_path, output_file):
    final = []
    file_list = [os.path.join(target_path, file) for file in os.listdir(
        target_path) if file.endswith(".json")]
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
            final.extend(data)
    with open(output_file, 'w') as f:
        json.dump(final, f, indent=4)
    print(f"Merge {len(file_list)} files to {output_file}")

def csv_2_json(csv_file, json_file, target_num=100):
    df = pd.read_csv(csv_file)
    df = df[['instruction', 'input', 'output']]
    df_new = df.sample(n=target_num).copy()
    df_new.to_json(json_file, orient='records', indent=4)

def sample_task(csv_list, target_num_list, target_path, cache_path):
    for num, file in enumerate(csv_list):
        json_file = os.path.join(cache_path, file.split(f"{target_path}/")[1] + f"_{target_num_list[num]}.json")
        csv_2_json(file, json_file, target_num=target_num_list[num])
        print(f"Convert {file} to {json_file}")

def main():
    target_num_list = [3000, 250, 250]
    target_path = "history"
    cache_path = "cache"
    merge_out = f"fingpt_v1_{sum(target_num_list)//1000}k.json"
    csv_list = [os.path.join(target_path, file) for file in os.listdir(target_path) if file.endswith(".csv")]

    sample_task(csv_list, target_num_list, target_path, cache_path)
    merge_json(cache_path, merge_out)

if __name__ == "__main__":
    main()

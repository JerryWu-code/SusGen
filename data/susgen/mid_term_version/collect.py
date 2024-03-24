# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to collect and shuffle data files to get a final dataset
#############################################################################

import os, json, random

def merge_json(target_path, output_file):
    final = []
    file_list = [os.path.join(target_path, file) for file in os.listdir(
        target_path) if file.endswith(".json")]
    for file in file_list:
        with open(file, 'r') as f:
            data = json.load(f)
            final.extend(data)
    random.shuffle(final)
    with open(output_file, 'w') as f:
        json.dump(final, f, indent=4)
    print(f"Merge {len(file_list)} files to {output_file}")

def main():
    output_file = 'susgen_6k.json'
    merge_json('history/', output_file)

if __name__ == "__main__":
    main()
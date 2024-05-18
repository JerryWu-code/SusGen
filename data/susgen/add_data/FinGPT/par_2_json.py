# Author: Wu Qilong
# Institute: National University of Singapore
# Description: Use this script to turn parquet files to json files.

#############################################################################
import json, os
import pandas as pd

def parquet_to_json(parquet_path, json_path):
    df = pd.read_parquet(parquet_path)
    # reorder columns
    df = df[['instruction', 'input', 'output']]

    json_data = df.to_dict(orient='records')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def process_a_folder(target_folder):
    name = target_folder.split("/")[0]
    parquet_files = [i for i in os.listdir(target_folder) if i.endswith(".parquet")]
    json_files = [name + "_" + i.split("-")[0] + ".json" for i in parquet_files]
    for i in range(len(parquet_files)):
        parquet_to_json(
            os.path.join(target_folder, parquet_files[i]), 
            os.path.join(target_folder, json_files[i])
        )
    print(f"Processed {len(parquet_files)} files in {target_folder} to: {json_files}.")
    # Delete parquet files
    for i in parquet_files:
        os.remove(os.path.join(target_folder, i))
    print(f"Deleted {len(parquet_files)} original parquet files in {target_folder}.")

def main():
    # target_folder = "fingpt-convfinqa/data"
    target_folders = [os.path.join(i, "data") for i in os.listdir(".")]
    for target_folder in target_folders:
        if os.path.isdir(target_folder):
            process_a_folder(target_folder)

if __name__ == "__main__":
    main()
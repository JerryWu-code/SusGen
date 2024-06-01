# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to load financial instruction tuning dataset from huggingface.

#############################################################################
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

# token = os.getenv("HUGGING_FACE_TOKEN")
repo_type = "dataset"

def download_datasets(repo_ids, repo_type, cache_dir):
    for repo_id in repo_ids:
        dataset = load_dataset(
            path=repo_id, 
            cache_dir=cache_dir,
        )
        print(f"Downloaded {repo_id} to {cache_dir}")

def load_datasets(repo_ids, cache_dir):
    datasets = {}
    for repo_id in repo_ids:
        dataset = load_dataset(repo_id, cache_dir=cache_dir)
        datasets[repo_id] = dataset
    return datasets

def main():
    # Define the cache directory & example path
    cache_dir = "..."
    example_path = "..."
    repo_ids = [
        "FinGPT/fingpt-finred", 
        "FinGPT/fingpt-finred-re", 
        "FinGPT/fingpt-fiqa_qa",
        "climatebert/tcfd_recommendations",
        "rexarski/TCFD_disclosure", # ...
    ]
    # 1. Download the datasets
    download_datasets(repo_ids, repo_type, cache_dir)

    # 2. Load the datasets
    datasets = load_datasets(repo_ids, cache_dir)
    for name, data in datasets.items():
        print(f"Dataset: {name}")
        df = data['train'].to_pandas()
        print(df.head(), "\n\n")
        # save to csv

        df.to_csv(f"{example_path}/fin_instruction/{name.split('/')[-1]}.csv", index=False)

if __name__ == "__main__":
    main()
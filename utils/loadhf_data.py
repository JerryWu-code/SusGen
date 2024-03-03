# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to load financial instruction tuning dataset from huggingface.

#############################################################################
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config import cache_dir
from huggingface_hub import snapshot_download

# token = os.getenv("HUGGING_FACE_TOKEN")

repo_ids = [
    "FinGPT/fingpt-finred", 
    "FinGPT/fingpt-finred-re", 
    "FinGPT/fingpt-fiqa_qa",
    "climatebert/tcfd_recommendations",
    "rexarski/TCFD_disclosure",
]

repo_type = "dataset"
use_auth_token = True

for repo_id in repo_ids:
    snapshot_path = snapshot_download(repo_id=repo_id, 
                                      repo_type=repo_type,
                                      cache_dir=cache_dir,
                                      use_auth_token=use_auth_token)
    print(f"This Repository {repo_id} has been downloaded to：{snapshot_path}")

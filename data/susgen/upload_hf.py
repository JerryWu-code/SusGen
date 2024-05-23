# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to upload dataset to huggingface.

#############################################################################
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="FINAL/PER_3500/FINAL_PER3500_30k.json",
    path_in_repo="susgen-30k.json",
    repo_id="WHATX/susgen-30k",
    repo_type="dataset",
    commit_message="Add susgen-30k.json",
)
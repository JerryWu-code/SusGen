# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to download pdfs from the excel file and put into the folder.

#############################################################################
import os
import sys
import subprocess
import shlex
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
from tqdm import tqdm
from src.config import *


# use wget to download the pdfs and rename them
def download_pdfs(target_list=tcfd_list, data_path=raw_data_path, updated_list=None):
    if "tcfd" in target_list:
        suffix = "_TCFD"
    elif "esg" in target_list:
        suffix = "_ESG"

    # read the download list file
    df = pd.read_csv(target_list)

    # create the folder if it does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # download the pdfs
    for i in tqdm(range(len(df))):
        file_name = str(df["Company"][i]) + "_" + str(df["Year Published"][i]) + suffix + ".pdf"
        
        # check if the file is downloaded, yes mark the df_status as 1, no mark as 0
        if os.path.exists(data_path + "/" + file_name):
            continue

        else:
            # if timeout, try again, else skip
            for _ in range(3):
                cmd = "wget -O " + shlex.quote(data_path + "/" + file_name) + " " + shlex.quote(df["Report URL"][i])
                result = subprocess.run(cmd, shell=True, capture_output=True)
                if result.returncode != 0:
                    print(f"wget command failed with error: {result.stderr.decode()}")
                    df.loc[i, "Status"] = 0
                    continue
                try:
                    # if the pdf could be opened and the file is not empty, mark the status as 1
                    if os.path.getsize(data_path + "/" + file_name) > 256000: # ignore the pdfs that less than 250KB
                        with open(data_path + "/" + file_name, "rb") as pdf:
                            df.loc[i, "Status"] = 1
                            break
                except:
                    if os.path.exists(data_path + "/" + file_name):
                        os.remove(data_path + "/" + file_name)
                    df.loc[i, "Status"] = 0
                    continue

    # save the status
    if updated_list:
        df.to_csv(updated_list, index=False)

# delete the pdfs that cant be opened and not ended with "pdf"
def check_pdfs(data_path=raw_data_path):
    for file in os.listdir(data_path):
        if not file.endswith(".pdf"):
            os.remove(data_path, file)

def main():
    download_pdfs(target_list=tcfd_list, data_path=raw_data_path, updated_list=updated_tcfd_list)
    download_pdfs(target_list=esg_list, data_path=raw_data_path, updated_list=updated_esg_list)

if __name__ == "__main__":
    main()
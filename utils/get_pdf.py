# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to download pdfs from the excel file and put into the folder.

#############################################################################
import os, sys, subprocess, shlex
import pandas as pd
from tqdm import tqdm

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
        if os.path.exists(data_path + "/" + file_name) and os.path.getsize(data_path + "/" + file_name) > 256000:
            df.loc[i, "Status"] = 1
            continue
        
        cmd = "wget -O " + shlex.quote(data_path + "/" + file_name) + " " + shlex.quote(
            df["Report URL"][i]) + " --no-check-certificate"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=35)

            if result.returncode == 0:
                if os.path.getsize(data_path + "/" + file_name) > 256000: # ignore the pdfs that less than 250KB
                    df.loc[i, "Status"] = 1

                else:
                    os.remove(os.path.join(data_path, file_name))
                    df.loc[i, "Status"] = 0

            else:
                print(f"wget command failed with error: {result.stderr}")
                df.loc[i, "Status"] = 0

        except subprocess.TimeoutExpired:
            print(f"Download timed out for: {df['Report URL'][i]}")
            df.loc[i, "Status"] = 0

    # save the status
    if updated_list:
        df["Status"] = df["Status"].astype("Int64")
        df.to_csv(updated_list, index=False)

def main():
    raw_data_path = "../data/raw_data"
    tcfd_list = os.path.join(raw_data_path, "target_list", "tcfd.csv")
    esg_list = os.path.join(raw_data_path, "target_list", "esg.csv")
    updated_tcfd_list = os.path.join(raw_data_path, "target_list", "tcfd_new.csv")
    updated_esg_list = os.path.join(raw_data_path, "target_list", "esg_new.csv")

    download_pdfs(target_list=tcfd_list, data_path=raw_data_path, updated_list=updated_tcfd_list)
    download_pdfs(target_list=esg_list, data_path=raw_data_path, updated_list=updated_esg_list)

if __name__ == "__main__":
    main()
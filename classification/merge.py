# SusGen Project for A* and NUS
# 2024-03-15: Xuan W.

# This is the SusGen project for NUS and A*
# 17-03-2024: Xaun.W

import os
import pandas as pd

def get_final_csv(file1, file2, output_file):
    """
    merge the two csv to be the final final_tcfd_recommendations.csv
    args:
        file1: the path of non_zero.csv
        file2: the path of zero_predict.csv
        output_file: path of the output the final csv
    """
    if not os.path.exists(file1):
        raise FileNotFoundError(f"File '{file1}' not found.")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"File '{file2}' not found.")

    # load the csv
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2, usecols=["text", "predictions"]).rename(columns={"predictions": "label"})

    # save final
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_csv(output_file, index=False) 
    print(f"Successully output file in {output_file}")

if __name__ == "__main__":
    file1, file2 = "./data/non_zero.csv", "./data/zero_predict.csv"
    output_file = "./data/final_tcfd_recommendations.csv"
    get_final_csv(file1, file2, output_file)

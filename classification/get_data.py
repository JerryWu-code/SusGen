# SusGen Project for A* and NUS
# 2024-03-14: Xuan W.

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def extract_labels(input_csv_file, non_zero_csv_file, zero_csv_file):
    """
    get zero.csv and none_zero.csv 
    
    Args:
        input_csv_file (str): input CSV
        non_zero_csv_file (str): output CSV labelled "1"-"4"
        zero_csv_file(str): output csv labelled "0"
    """
    try:
        df = pd.read_csv(input_csv_file)
        non_zero_df = df[df['label'] != 0]
        zero_df = df[df['label'] == 0]

        if not non_zero_df.empty:
            if not os.path.exists(os.path.dirname(non_zero_csv_file)):
                os.makedirs(os.path.dirname(non_zero_csv_file))
            non_zero_df.to_csv(non_zero_csv_file, index=False)
            print("Successfully save the data labelled '1-4' to ", non_zero_csv_file)
        if not zero_df.empty:
            if not os.path.exists(os.path.dirname(zero_csv_file)):
                os.makedirs(os.path.dirname(zero_csv_file))
            zero_df.to_csv(zero_csv_file, index=False)
            print("Successfully save the data labelled '0' to ", zero_csv_file)

    except FileNotFoundError:
        print("No input file found:", input_csv_file)
    except Exception as e:
        print("Error:", e)

def split_train_test(data, train, test, train_test_scale=0.7):
    """
    split the non_zero.csv into train set and test set in train.csv and test.csv

    Args:
        data: input all non_zero.csv
        train: output train set for text classification
        test: output test set for text classification 
    """
    
    try:
        # Read CSV file
        df = pd.read_csv(data)

        # Split into training and testing sets using train_test_split function
        train_df, test_df = train_test_split(df, test_size=1-train_test_scale, random_state=42)

        if not os.path.exists(os.path.dirname(train)):
            os.makedirs(os.path.dirname(train))
        if not os.path.exists(os.path.dirname(test)):
            os.makedirs(os.path.dirname(test))

        # Save training and testing sets
        train_df.to_csv(train, index=False)
        test_df.to_csv(test, index=False)

        print("Successfully saved the training set to", train)
        print("Successfully saved the testing set to", test)
    except FileNotFoundError:
        print("File not found:", data)
    except Exception as e:
        print("An error occurred:", e)

def label_vis(data, img_file):
    """
    visualise the distribution of tcfd recommendation data
    args:
        data:input csv
    """
    df = pd.read_csv(data)
    label_counts = df['label'].value_counts()

    # plot
    plt.bar(label_counts.index, label_counts.values)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution of TCFD Recommendations')
    for i, count in enumerate(label_counts.values):
            plt.text(label_counts.index[i], count, str(count), ha='center', va='bottom')
    
    # save file
    img = os.path.join("./visualisation/", img_file)
    if not os.path.exists(os.path.dirname(img)):
        os.makedirs(os.path.dirname(img))
    plt.savefig(img)
    # plt.show()

if __name__ == "__main__":
    # extract the label information
    input_file = "./data/tcfd_recommendations.csv"
    non_zero_file = "./data/non_zero.csv"
    zero_file = "./data/zero.csv"
    extract_labels(input_file, non_zero_file, zero_file)

    # split the data
    # train = "./data/train.csv"
    # test = "./data/test.csv"
    # split_train_test(non_zero_file, train, test)

    # # # visulize the tcfd_recommendations.csv 
    label_vis(input_file, img_file="distribution_tcfd.jpg")
    # label_vis(train, img_file = "distribution_train")


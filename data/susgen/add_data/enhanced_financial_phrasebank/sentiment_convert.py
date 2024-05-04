import pandas as pd
import json
parquet_train_path = 'train-00000-of-00001.parquet'
df = pd.read_parquet(parquet_train_path)
df = pd.json_normalize(df['train'])
data_list = []

instruction_text = "You are an expert in the field of ESG and help determine the overall sentiment category of a given text in the ESG domain."

label_to_output = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

for index, row in df.iterrows():
    data_dict = {
        "instruction": instruction_text,
        "input": row['sentence'],
        "output": label_to_output[row['label']]
    }
    data_list.append(data_dict)

with open('sentiment.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)

print("JSON file has been created successfully.")

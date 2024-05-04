import pandas as pd
import json
parquet_train_path = 'train-00000-of-00001-612d72c293451962.parquet'
parquet_test_path = 'test-00000-of-00001-1c54f1266ce9d89f.parquet'
df = pd.read_parquet(parquet_train_path)
dft = pd.read_parquet(parquet_test_path)
emotion_columns = [
    'Environmental Negative', 'Environmental Neutral', 'Environmental Positive',
    'Governance Negative', 'Governance Neutral', 'Governance Positive',
    'Social Negative', 'Social Neutral', 'Social Positive'
]

data_list = []
data_list1 = []
for index, row in df.iterrows():
    output_labels = [col for col in emotion_columns if row[col] == 1]

    data_dict = {
        "instruction": "You are an expert in the field of ESG and help to determine the sentiment category of a given text in the domains of Environment, Governance and Social.",
        "input": row['Text'],
        "output": ', '.join(output_labels) 
    }

    data_list.append(data_dict)

with open('esg_sentiment_train.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)

print("JSON training file has been created successfully.")

for index, row in dft.iterrows():
    output_labels = [col for col in emotion_columns if row[col] == 1]

    data_dict = {
        "instruction": "You are an expert in the field of ESG and help to determine the sentiment category of a given text in the domains of Environment, Governance and Social.",
        "input": row['Text'],
        "output": ', '.join(output_labels) 
    }

    data_list1.append(data_dict)

with open('esg_sentiment_test.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list1, json_file, indent=4, ensure_ascii=False)

print("JSON testing file has been created successfully.")
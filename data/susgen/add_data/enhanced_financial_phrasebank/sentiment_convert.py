import pandas as pd
import json
import random
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

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

instructions_list = [
    "Classify the sentiment of the following financial phrase as Positive, Negative, or Neutral: ",
    "Determine whether the sentiment of this statement is Positive, Negative, or Neutral: ",
    "Is the sentiment of the following financial phrase Positive, Negative, or Neutral? ",
    "Identify the sentiment (Positive, Negative, Neutral) of the given financial phrase: ",
    "Analyze the sentiment of this financial statement. Choose from Positive, Negative, or Neutral: ",
    "Evaluate the sentiment of this financial phrase as Positive, Negative, or Neutral: ",
    "Determine the emotional tone of this financial statement. Is it Positive, Negative, or Neutral? ",
    "Sentiment analysis needed: Is this phrase Positive, Negative, or Neutral? ",
    "Assess the sentiment conveyed in this financial text. Choose from Positive, Negative, or Neutral: ",
    "Classify the sentiment in this financial phrase. Options are Positive, Negative, or Neutral: ",
    "What is the emotional tone (Positive, Negative, Neutral) of the following phrase? ",
    "Identify the sentiment in this text. Options: Positive, Negative, or Neutral: ",
    "Determine the sentiment of this financial phrase. Choose from Positive, Negative, or Neutral: ",
    "Analyze whether this statement is Positive, Negative, or Neutral: ",
    "What sentiment is expressed in this statement? Choose from Positive, Negative, or Neutral: ",
    "Classify the emotional tone of this financial text as Positive, Negative, or Neutral: ",
    "Evaluate whether this phrase has a Positive, Negative, or Neutral sentiment: ",
    "Analyze the emotional sentiment of this phrase. Options: Positive, Negative, or Neutral: ",
    "Classify the sentiment expressed in this statement as Positive, Negative, or Neutral: ",
    "What is the tone of this financial phrase (Positive, Negative, Neutral)? ",
    "Is the sentiment in this text Positive, Negative, or Neutral? ",
    "Please determine the sentiment of the following phrase. Options: Positive, Negative, or Neutral: ",
    "Analyze the sentiment of the given text. Is it Positive, Negative, or Neutral? ",
    "What is the sentiment conveyed by this phrase? Choose from Positive, Negative, or Neutral: ",
    "Identify whether the sentiment is Positive, Negative, or Neutral: ",
    "Determine the sentiment expressed in this text as Positive, Negative, or Neutral: ",
    "Assess the tone of this financial phrase. Options: Positive, Negative, or Neutral: ",
    "Classify this statement as Positive, Negative, or Neutral: ",
    "What sentiment does this phrase express? Choose from Positive, Negative, or Neutral: ",
    "Identify the emotional tone of this statement. Is it Positive, Negative, or Neutral? ",
    "Determine if the sentiment is Positive, Negative, or Neutral: ",
    "Analyze the given phrase for sentiment. Options: Positive, Negative, or Neutral: ",
    "Classify the following text by sentiment (Positive, Negative, Neutral): ",
    "What is the emotional tone conveyed in this phrase? Choose from Positive, Negative, or Neutral: ",
    "Determine the sentiment classification of this financial text: Positive, Negative, or Neutral: ",
    "Evaluate the sentiment (Positive, Negative, Neutral) in this phrase: ",
    "What is the sentiment of the following phrase? Options: Positive, Negative, or Neutral: ",
    "Classify the sentiment in this statement as Positive, Negative, or Neutral: ",
    "What is the emotional sentiment of this financial phrase? Choose from Positive, Negative, or Neutral: ",
    "Identify the sentiment classification for this text: Positive, Negative, or Neutral: ",
    "Assess whether this phrase is Positive, Negative, or Neutral: ",
    "What emotional tone does this text convey? Choose from Positive, Negative, or Neutral: ",
    "Classify whether this statement is Positive, Negative, or Neutral: ",
    "Determine the emotional tone of this text as Positive, Negative, or Neutral: ",
    "Analyze the sentiment of the following statement. Options: Positive, Negative, or Neutral: ",
    "Evaluate the tone (Positive, Negative, Neutral) of this phrase: ",
    "What is the sentiment classification of this statement? Choose from Positive, Negative, or Neutral: "
]

data = load_json('sentiment.json')

for item in data:
    new_instruction = random.choice(instructions_list)
    item['instruction'] = new_instruction

save_json(data, 'sentiment_v2.json')

print("The dataset has been updated with random instructions.")

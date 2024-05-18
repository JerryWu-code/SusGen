import random
from datasets import load_dataset
import json
from sklearn.model_selection import train_test_split


def transform_fiqa_dataset():
    """Transform FiQA 2018 dataset into the required format and split it."""
    qa_instructions = [
        "Answer the following question based on the provided financial document: {input}",
        "Given the financial context, answer the question: {input}",
        "Based on the text, provide an answer to the question: {input}",
        "Read the following financial text and answer the question: {input}",
        "With reference to the provided document, answer the question: {input}",
        "Given the financial information, what is the answer to the question: {input}",
        "From the given text, answer the question: {input}",
        "Using the provided financial context, answer the following: {input}",
        "Refer to the text and answer the question: {input}",
        "Based on the given document, what is the answer to the question: {input}"
    ]

    sa_instructions = [
        "Classify the sentiment of the following text as Positive, Negative, or Neutral: \"{input}\"",
        "Determine the sentiment (Positive, Negative, Neutral) of this statement: \"{input}\"",
        "What is the sentiment of the following text? Choose from Positive, Negative, or Neutral: \"{input}\"",
        "Analyze the sentiment of this text and classify it as Positive, Negative, or Neutral: \"{input}\"",
        "Is the sentiment of the following statement Positive, Negative, or Neutral? \"{input}\"",
        "Evaluate the sentiment of this text: Positive, Negative, or Neutral: \"{input}\"",
        "Identify the sentiment (Positive, Negative, Neutral) in the following text: \"{input}\"",
        "Assess whether the sentiment of this text is Positive, Negative, or Neutral: \"{input}\"",
        "Classify the emotional tone of this text: Positive, Negative, or Neutral: \"{input}\"",
        "Determine if the sentiment expressed is Positive, Negative, or Neutral: \"{input}\""
    ]
    
    # Step 1: Download the FiQA 2018 dataset
    fiqa_dataset = load_dataset("salt/flue", "fiqa")
    print(fiqa_dataset)
    # # Step 2: Transform the dataset into the required format
    # transformed_data = []
    # for entry in fiqa_dataset:
    #     if 'question' in entry:
    #         # For QA pairs
    #         question = entry['question']
    #         answer = entry['answer']
    #         instruction = random.choice(qa_instructions).format(input=question)
            
    #         transformed_entry = {
    #             "instruction": instruction,
    #             "input": question,
    #             "answer": answer
    #         }
    #         transformed_data.append(transformed_entry)
    #     elif 'sentence' in entry:
    #         # For Sentiment Analysis
    #         sentence = entry['sentence']
    #         sentiment = entry['label']
    #         sentiment_str = "Positive" if sentiment == 2 else "Negative" if sentiment == 0 else "Neutral"
            
    #         instruction = random.choice(sa_instructions).format(input=sentence)
            
    #         transformed_entry = {
    #             "instruction": instruction,
    #             "input": sentence,
    #             "answer": sentiment_str
    #         }
    #         transformed_data.append(transformed_entry)

    # # Step 3: Split the dataset into training, testing, and validation sets
    # train_data, temp_data = train_test_split(transformed_data, test_size=0.3, random_state=42)
    # val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # # Step 4: Save the transformed data to JSON files
    # save_to_json(train_data, "/eval/benchmark/fiqa/fiqa_train.json")
    # save_to_json(val_data, "/eval/benchmark/fiqa/fiqa_val.json")
    # save_to_json(test_data, "/eval/benchmark/fiqa/fiqa_test.json")

    print("Data transformation and splitting complete. Files saved as fiqa_train.json, fiqa_val.json, and fiqa_test.json.")

def transform_phrasebank_dataset():
    """Transform Financial PhraseBank dataset into the required format."""
    # Constrained instructions for sentiment classification
    instructions = [
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

    # Step 1: Download the Financial PhraseBank dataset
    dataset = load_dataset("financial_phrasebank", config_name="sentences_75agree", split="train")

    # Step 2: Transform the dataset into the required format
    transformed_data = []
    for entry in dataset:
        phrase = entry['sentence']
        sentiment = entry['label']
        sentiment_str = "Positive" if sentiment == 2 else "Negative" if sentiment == 0 else "Neutral"
        
        # Randomly select an instruction
        instruction = random.choice(instructions).format(phrase=phrase)
        
        transformed_entry = {
            "instruction": instruction,
            "input": phrase,
            "answer": sentiment_str
        }
        transformed_data.append(transformed_entry)

    # Step 3: Save the transformed data to a JSON file
    output_file = "/eval/benchmark/financial_phrasebank/financial_phrasebank_transformed.json"
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=4)

    print(f"Transformed data saved to {output_file}")

def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # print("Transforming Financial PhraseBank dataset...")
    # transform_phrasebank_dataset()
    print("Transforming FiQA 2018 dataset...")
    transform_fiqa_dataset()

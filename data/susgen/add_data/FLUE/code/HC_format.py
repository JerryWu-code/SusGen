import pandas as pd
import json

# Step 1: Load the CSV file
csv_file = "./HC/finalDataset_0208.csv"
df = pd.read_csv(csv_file)

# Step 2: Generate Instructions
instructions = [
    "Based on the following news headline, does the news item talk about price or not? Answer with 0 or 1.",
    "Based on the following news headline, does the news talk about price going up? Answer with 0 or 1.",
    "Based on the following news headline, does the news talk about price staying constant? Answer with 0 or 1.",
    "Based on the following news headline, does the news talk about price going down? Answer with 0 or 1.",
    "Based on the following news headline, does it talk about an event in the past? Answer with 0 or 1.",
    "Based on the following news headline, does it talk about an event in the future? Answer with 0 or 1.",
    "Based on the following news headline, does it talk about a general event (apart from prices) in the past? Answer with 0 or 1."
]

# Step 3: Transform the data
transformed_data = []

for _, row in df.iterrows():
    news = row['News']
    
    entry1 = {
        "instruction": instructions[0],
        "input": news,
        "answer": int(row['Price or Not'])
    }
    entry2 = {
        "instruction": instructions[1],
        "input": news,
        "answer": int(row['Direction Up'])
    }
    entry3 = {
        "instruction": instructions[2],
        "input": news,
        "answer": int(row['Direction Constant'])
    }
    entry4 = {
        "instruction": instructions[3],
        "input": news,
        "answer": int(row['Direction Down'])
    }
    entry5 = {
        "instruction": instructions[4],
        "input": news,
        "answer": int(row['PastPrice'])
    }
    entry6 = {
        "instruction": instructions[5],
        "input": news,
        "answer": int(row['FuturePrice'])
    }
    entry7 = {
        "instruction": instructions[6],
        "input": news,
        "answer": int(row['PastNews'])
    }
    
    transformed_data.extend([entry1, entry2, entry3, entry4, entry5, entry6, entry7])

# Step 4: Save the transformed data to a JSON file
output_file = "./HC/FLUE_HC.json"
with open(output_file, 'w') as f:
    json.dump(transformed_data, f, indent=4)

print(f"Transformed data saved to {output_file}")

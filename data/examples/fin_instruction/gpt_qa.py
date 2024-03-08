from openai import OpenAI
from config import *
import json
import os
import shutil
from tqdm import tqdm

def get_qa_query(background, foreground):

    client = OpenAI(api_key = openai_key3)
    prompt = """
    As a senior equity analyst with expertise in climate science 
    evaluating a company 's sustainability report , 
    you are presented with the following information:
    {0}
    Given the {1} aspect, and then you will generate a question based on the information.
    """.format(basic_info, domain)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": """
                You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: 2021-09 Current date: 2024-01-19
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,  # 0 for deterministic
        max_tokens=256,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
    )

    query = response.choices[0].message.content
    print("Foreground: " + foreground)
    print("Background: " + background)
    print("Query: " + query.capitalize())
    # print(response)
    return query

def process_file(file_path, output_path, checkpoint_file_path):
    temp_output_path = checkpoint_file_path + ".temp"
    checkpoint_file = checkpoint_file_path + ".checkpoint"
    start_line = 0

    # check if exists checkpoint file
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as cp_file:
            start_line = int(cp_file.read().strip())

    total_lines = sum(1 for line in open(file_path, 'r'))
    with open(temp_output_path, "a" if start_line > 0 else "w") as temp_file:
        with open(file_path, "r") as jsonl_file:
            for current_line, line in enumerate(tqdm(jsonl_file, initial=start_line, total=total_lines, desc="Processing")):
                
                # skip line lower than checkpoint
                if current_line < start_line:
                    continue

                json_data = json.loads(line)
                background = json_data['background']
                foreground = json_data['foreground']
                query = get_merged_query(background, foreground)
                json_data['query'] = query

                temp_file.write(json.dumps(json_data) + '\n')
                temp_file.flush()
                os.fsync(temp_file.fileno())

                # update checkpoint
                with open(checkpoint_file, "w") as cp_file:
                    cp_file.write(str(current_line + 1))

    # Move temp file to actual output file
    shutil.move(temp_output_path, output_path)

def run_all():
    file_path = [
        "../data/annotation/train_v2.jsonl",
    ]
    output_path = [
        "../data/annotation/train_v3.jsonl",
    ]
    ckpt_path = [
        "../checkpoint/train_v3.jsonl.checkpoint",
    ]
    for i in range(len(file_path)):
        process_file(file_path[0], output_path[0], ckpt_path[0])
        print("Finish processing file: " + file_path[0])

def main():
    # For single query test
    foreground = "The adult A is next to the adult B, and the guitar is in front of the sofa."
    background = "A painting of a field with a man and a woman in it."
    get_qa_query(background, foreground)

    # Run all query
    # run_all()

if __name__ == "__main__":
    main()
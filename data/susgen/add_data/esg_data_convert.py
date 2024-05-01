import re
import csv
import json

data_path = 'esg_data.csv'

instructions = []
questions = []
answers = []

pattern = re.compile(r"<<SYS>>(.*?)<</SYS>>(.*?)\[/INST\](.*?) </s>", re.DOTALL)

with open(data_path, newline='', encoding='Windows-1252') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        line = row['perguntas'] 
        # print(line)
        match = pattern.search(line)
        if match:
            # instruction, question, answer = match.groups()
            instruction = match.group(1).strip()
            question = match.group(2).strip()
            answer = match.group(3).strip()
            instructions.append(instruction)
            questions.append(question)
            answers.append(answer)
        else:
            print("No matches found.")

print(instructions)
print(questions)
print(answers)

# with open('Dataset/instructions.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Instruction'])  # 写入列名
#     for instruction in instructions:
#         writer.writerow([instruction])

# with open('Dataset/questions.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Question'])
#     for question in questions:
#         writer.writerow([question])

# with open('Dataset/answers.csv', 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Answer'])
#     for answer in answers:
#         writer.writerow([answer])

data_to_save = []
for instruction, question, answer in zip(instructions, questions, answers):
    data_to_save.append({
        "instruction": instruction,
        "input": question,
        "output": answer
    })

with open('esg_data.json', 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, indent=4, ensure_ascii=False)

print("Data has been written to JSON file successfully.")

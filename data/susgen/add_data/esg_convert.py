import pandas as pd
import json
path = 'EN-train1.xlsx'
df = pd.read_excel(path)
data_list = []

for text in df['text']:
    instruction_index = text.find('## Instruction:')
    answer_index = text.find('## Answer :')
    
    if instruction_index != -1 and answer_index != -1:
        pre_instruction = text[:instruction_index].strip()
        instruction = text[instruction_index + len('## Instruction:'):answer_index].strip()
        answer = text[answer_index + len('## Answer :'):].strip()
        
        data_dict = {
            "instruction": pre_instruction,
            "input": instruction,
            "output": answer
        }
        data_list.append(data_dict)

with open('esg.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, indent=4, ensure_ascii=False)

print("JSON file has been created successfully.")

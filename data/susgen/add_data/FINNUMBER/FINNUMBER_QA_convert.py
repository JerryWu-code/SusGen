import pandas as pd
import json

def process_to_json(input_file_path, output_json_path):
    df = pd.read_excel(input_file_path)

    data_list = []

    for index, row in df.iterrows():
        instruction_text = row['instruction']
        output_text = row['output']

        command_start = instruction_text.find('###command###') + len('###command###')
        main_text_start = instruction_text.find('###main text###') + len('###main text###')
        question_start = instruction_text.find('###question###') + len('###question###')
        answer_start = instruction_text.find('###answer###')

        command_content = instruction_text[command_start:instruction_text.find('###main text###')].strip()
        main_text_content = instruction_text[main_text_start:instruction_text.find('###question###')].strip()
        question_content = instruction_text[question_start:answer_start].strip()

        # 构建字典
        data_dict = {
            "instruction": command_content,
            "input": main_text_content + '\nQuestion: \n' + question_content,
            "output": output_text
        }
        data_list.append(data_dict)

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=4, ensure_ascii=False)

    print("JSON file has been created successfully.")

BQA_path = 'EN-BQA-400.xlsx'
BQA_json = 'BQA-400.json'
EQA_path = 'EN-EQA-400.xlsx'
EQA_json = 'EQA-400.json'
MCQA_path = 'EN-MCQA-400.xlsx'
MCQA_json = 'MCQA-400.json'
NQA1_path = 'EN-NQA-ARI-400.xlsx'
NQA1_json = 'NQA-ARI-400.json'
NQA2_path = 'EN-NQA-COM-400.xlsx'
NQA2_json = 'NQA-COM-400.json'
NQA3_path = 'EN-NQA-EXT-400.xlsx'
NQA3_json = 'NQA-EXT-400.json'

process_to_json(BQA_path, BQA_json)
process_to_json(EQA_path, EQA_json)
process_to_json(MCQA_path, MCQA_json)
process_to_json(NQA1_path, NQA1_json)
process_to_json(NQA2_path, NQA2_json)
process_to_json(NQA3_path, NQA3_json)

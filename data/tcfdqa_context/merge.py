# merge the qa pair and the summary <context> info

import json

def merge_contexts(qa_dict_file_path, sum_file_path, output_file_path):
    # Load the JSON files
    with open(sum_file_path, 'r') as sum_file, open(qa_dict_file_path, 'r') as qa_dict_file:
        sum_data = json.load(sum_file)
        qa_dict_data = json.load(qa_dict_file)

    # Create a lookup dictionary for context based on file names
    context_lookup = {item['files']: item['context'] for item in sum_data}

    # Merge contexts into qa_dict_data
    for qa_item in qa_dict_data:
        file_name = qa_item['file']
        if file_name in context_lookup:
            print(f"exist files: {file_name}")
            context = context_lookup[file_name]
            for key in qa_item['content']:
                if key.lower() in context:
                    for qa_pair in qa_item['content'][key]:
                        qa_pair['context'] = context['summary'] + " " + context[key.lower()]


    # Save the merged result to the output file
    with open(output_file_path, 'w') as output_file:
        json.dump(qa_dict_data, output_file, indent=4)
        print("Successfully save the merge.json file!")


if __name__ == "__main__":
    # Usage
    qa_path = './jsons/qa_dict.json'
    sum_path = './jsons/sum.json'
    merge_path = './jsons//merge.json'

    merge_contexts(qa_path, sum_path, merge_path)

import json
import random

# Load the JSON data
with open('data/susgen/tcfd_qa/tcfd_qa_v1.json', 'r') as file:
    data = json.load(file)

# List of new prefixes to choose from
prefix_list = [
    "As an expert in the TCFD framework, provide detailed responses to enhance the organization's TCFD report in the areas of Governance, Strategy, Risk Management, and Metrics & Targets, focusing on climate-related risks and opportunities.",
    "Drawing on your expertise in the TCFD framework, offer in-depth insights on improving an organization's TCFD disclosures across Governance, Strategy, Risk Management, and Metrics & Targets, with an emphasis on managing climate-related risks and opportunities.",
    "As a specialist in the TCFD framework, elaborate on how to refine an organization’s TCFD report, specifically addressing Governance, Strategy, Risk Management, and Metrics & Targets to better capture climate-related risks and opportunities.",
    "Utilizing your knowledge of the TCFD framework, advise on optimizing the organization's TCFD report by detailing essential practices in Governance, Strategy, Risk Management, and Metrics & Targets for climate-related risks and opportunities.",
    "As a TCFD framework authority, guide the enhancement of an organization's report by focusing on improved disclosure of climate-related risks and opportunities in the areas of Governance, Strategy, Risk Management, and Metrics & Targets.",
    "Leverage your TCFD framework expertise to enrich an organization’s report, providing detailed guidance on Governance, Strategy, Risk Management, and Metrics & Targets, particularly in the context of climate-related risks and opportunities."
]


# Function to replace prefix before "Answer the following questions:"
def replace_prefix(instruction):
    split_point = "Answer the following questions:"
    if split_point in instruction:
        questions = instruction.split(split_point)[-1]
        new_prefix = random.choice(prefix_list)
        new_instruction = new_prefix + split_point + questions
        return new_instruction
    else:
        return instruction  # Return as is if the split point isn't found

# Modify each entry in the JSON data
for item in data:
    item['instruction'] = replace_prefix(item['instruction'])

# Write the modified data to a new JSON file
with open('data/susgen/tcfd_qa/tcfd_qa_v3.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Prefixes replaced and new JSON file created successfully.")



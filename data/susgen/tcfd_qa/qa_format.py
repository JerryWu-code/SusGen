import json
from os import read


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)



def alter_json(json_data):
    
    # Instructions for each section
    section_instructions = {   
        "Governance": "Considering the organization's framework for climate-related risks and opportunities governance, explain how the governance structure (board-level committees, etc.) oversees and evaluates these aspects. Be specific about the roles, responsibilities, and frequency of climate-related risk assessments and strategic planning discussions.",
        "Strategy": "Detail the organization's short-term and long-term strategies regarding climate-related risks and opportunities. Include an analysis of the financial implications, potential scenarios, and the strategic fit into the organization's overall business strategy. Focus on how these factors influence operational planning, investment strategies, and innovation initiatives.",
        "Risk Management": "Describe the processes involved in identifying and assessing climate-related risks. Outline how these risks are integrated into the organization’s overall risk management. Include specifics about the tools, methodologies, and data sources used in risk assessment and management. Highlight any particular practices for managing these risks effectively.",
        "Metrics and Targets": "Provide a comprehensive overview of the metrics and targets established by the organization to monitor climate-related risks and opportunities. Detail the benchmarks, time frames, and progress measurement approaches. Explain how these metrics align with the organization's overall sustainability and climate strategy."
    }

    # General instruction applicable to all items
    general_instruction = "Imagine you are a leading expert in climate-related financial disclosures, specializing in the TCFD framework. Your role encompasses deep insights into how organizations can effectively disclose information regarding Governance, Strategy, Risk Management, and Metrics & Targets in relation to climate-related risks and opportunities. Your task is to assist in generating detailed, accurate, and insightful answers for a QA session focused on enhancing an organization's TCFD report. For each of the following sections, provide expert-level responses based on the core requirements of TCFD disclosures: \n"

    # Transform the JSON data
    transformed_data = []

    for item in json_data:
        for section, questions in item["content"].items():
            for q in questions:
                instruction = general_instruction + section_instructions[section] + "\nAnswer the following questions: \n1. " + q["Q"] + "\n"
                transformed_data.append({
                    "instruction": instruction,
                    "input": "",
                    "output": q["A"]
                })
    return transformed_data


def main():
    path = "data/susgen/tcfd_qa/qa_dict.json"
    data = read_json(path)
    transformed_data = alter_json(data)
    with open("data/susgen/tcfd_qa/qa_dict_formatted.json", "w", encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)
    # print(transformed_data)

if __name__ == "__main__":
    main()
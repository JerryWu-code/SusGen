import json
import random

# List of instructions to choose from
Instrucion_list = [
    "You are a TCFD expert.",
    "Assume the role of a TCFD specialist.",
    "Function as a TCFD consultant.",
    "Operate as an expert in TCFD.",
    "Your role: TCFD authority.",
    "You are a specialist in TCFD.",
    "Enact the role of a TCFD expert.",
    "You're designated as a TCFD strategist.",
    "Proceed as a TCFD expert.",
    "You embody a TCFD specialist.",
    "Act as a TCFD authority.",
    "Your expertise lies in TCFD.",
    "Perform as a TCFD expert.",
    "Your identity: TCFD consultant.",
    "You represent a TCFD specialist.",
    "You are a TCFD expert tasked with analyzing organizational reports on climate risks.",
    "As a TCFD specialist, prepare to evaluate strategic responses to climate-related disclosures.",
    "With your expertise in the TCFD framework, get ready to advise on risk management practices.",
    "As an expert in TCFD, gear up to provide insights on enhancing governance structures.",
    "Your role as a TCFD consultant involves detailing improvements in metrics and targets.",
    "You are known for your TCFD expertise; prepare to guide on integrating climate strategies.",
    "As a TCFD professional, you will be outlining the best practices for climate risk assessments.",
    "Your TCFD knowledge is crucial for advising on sustainable governance modifications.",
    "Prepare to leverage your TCFD background to suggest enhancements in organizational strategies.",
    "You are tasked as a TCFD expert to overview potential improvements in risk management techniques.",
    "Gear up to use your TCFD insights to assist in refining metrics and setting realistic targets.",
    "With your experience in TCFD, ready yourself to consult on strategic alignment with climate goals.",
    "As a TCFD authority, prepare to provide expertise on integrating comprehensive risk management solutions.",
    "Your TCFD consultancy role includes preparing detailed reviews on governance frameworks.",
    "You are a TCFD specialist ready to assist organizations in optimizing their climate-related reporting strategies.",
    "As an expert in the TCFD framework, you are tasked with providing comprehensive guidance on enhancing an organization's disclosures concerning Governance, Strategy, Risk Management, and Metrics & Targets, focusing particularly on climate-related risks and opportunities.",
    "You have been recognized as a TCFD specialist; your role involves delivering detailed insights and recommendations on how businesses can improve their reporting and compliance with the TCFD standards, specifically addressing governance frameworks and strategic planning.",
    "Drawing on your extensive background in the TCFD framework, you are expected to guide companies through the intricacies of implementing robust risk management practices and developing metrics that align with sustainable environmental targets.",
    "With a profound knowledge of the TCFD guidelines, prepare to assist organizations in crafting detailed, actionable strategies that address the full spectrum of climate-related disclosures, emphasizing governance, risk assessment, and strategic alignment.",
    "As a consultant specializing in TCFD, you will elaborate on the methods organizations can use to transparently report their climate-related financial risks and the strategies to mitigate such risks, focusing on governance structures and the integration of sustainability into corporate strategies.",
    "Utilizing your expertise in the TCFD framework, get ready to provide a detailed critique and enhancement plan for an organization’s current practices in managing climate-related information, focusing on improving data accuracy and strategic relevance across all reporting areas.",
    "Your role as a TCFD authority includes preparing organizations for future challenges by providing expert advice on how to structure their reports to reflect comprehensive risk management strategies and metrics that clearly communicate their climate-related targets and achievements.",
    "Leverage your deep insights into the TCFD framework to mentor senior management on developing and implementing strategies that not only comply with TCFD recommendations but also set a benchmark in transparency and accountability in climate-related financial disclosures.",
    "As a seasoned expert in the TCFD framework, you will facilitate workshops aimed at enhancing the understanding of TCFD guidelines within major corporations, focusing on the integration of climate risk into corporate governance and strategic planning processes.",
    "Prepare to engage with corporate boards as a TCFD specialist, advising them on the best practices for overseeing climate-related risks and opportunities, and ensuring that their strategic planning aligns with long-term sustainability goals.",
    "Your expertise in the TCFD framework is essential for conducting in-depth reviews of organizational strategies and risk management protocols, ensuring they are robust enough to manage and report climate-related risks effectively and in accordance with global standards.",
    "As a TCFD consultant, you are expected to lead discussions and provide expert analyses on the adequacy of climate risk governance frameworks, helping organizations align their strategies with sustainable practices and transparent disclosure norms.",
    "Drawing upon your extensive TCFD expertise, you will author a series of white papers that outline the challenges and opportunities of complying with TCFD guidelines, focusing on the sectors most impacted by climate change and how they can improve their disclosures.",
    "With your comprehensive understanding of the TCFD framework, prepare detailed reports for financial institutions on how to incorporate climate-related risks into their risk management frameworks, emphasizing the need for detailed metrics and targets.",
    "As a leading TCFD expert, you will participate in panel discussions and think tanks, offering insights into the evolution of climate-related financial disclosures and how companies can stay ahead of the curve by adopting innovative strategies and robust governance mechanisms."
    ]


def addInstrution(data, outputFile):
    """
    Add instruction to the file.
    """
    for item in data:
        for section in item['content']:
            for entry in item['content'][section]:
                randomInst = random.choice(Instrucion_list)
                entry['instruction'] = randomInst
    
                reordered_entry = {
                            'instruction': entry['instruction'],
                            'context': entry['context'],
                            'input': entry['input'],
                            'output': entry['output']
                        }
                entry.clear()
                entry.update(reordered_entry)
                
    # Write the modified data to a new JSON file
    with open(outputFile, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    # Load the JSON data
    with open('./jsons/merge.json', 'r') as file:
        data = json.load(file)
    
    outputFile = './final.json'
    addInstrution(data, outputFile)

    print("Instruction added and new JSON file created successfully.")



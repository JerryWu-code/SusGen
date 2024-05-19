import json
import random

prefix_list = [
    "You are an ESG expert, categorize the sentiment in {}.",
    "Determine the sentiment as an ESG specialist in {}.",
    "As an ESG consultant, assess the sentiment in {}.",
    "Your role: ESG authority, sentiment in {}.",
    "As an ESG expert, analyze sentiment in {}.",
    "You are a specialist in ESG, categorize sentiment in {}.",
    "Enact the role of ESG expert, determine sentiment in {}.",
    "You're an ESG strategist, assess sentiment in {}.",
    "Proceed as an ESG expert, sentiment in {}.",
    "You embody an ESG specialist, categorize sentiment in {}.",
    "Act as an ESG authority, determine sentiment in {}.",
    "Your expertise is ESG, assess sentiment in {}.",
    "Perform as an ESG expert, categorize sentiment in {}.",
    "Your identity: ESG consultant, determine sentiment in {}.",
    "You represent ESG expertise, assess sentiment in {}.",
    "As an ESG expert, determine the sentiment category of a given text in {}.",
    "Assume the role of an ESG specialist, categorizing the sentiment of texts in {}.",
    "Function as an ESG consultant, assessing the sentiment category in {}.",
    "Your role as an ESG authority includes determining sentiment in {}.",
    "As an ESG expert, analyze and categorize sentiment in {}.",
    "You are a specialist in ESG, responsible for determining sentiment in {}.",
    "Enact the role of ESG expert, and assess the sentiment category in {}.",
    "You're designated as an ESG strategist, determining sentiment in {}.",
    "Proceed as an ESG expert, categorizing sentiment in {}.",
    "You embody an ESG specialist, tasked with assessing sentiment in {}.",
    "Act as an ESG authority, responsible for determining sentiment in {}.",
    "Your expertise in ESG involves categorizing sentiment in {}.",
    "Perform as an ESG expert, assessing the sentiment category in {}.",
    "Your identity as an ESG consultant includes determining sentiment in {}.",
    "You represent ESG expertise, responsible for assessing sentiment in {}.",
    "As an expert in the field of ESG, you are responsible for determining the sentiment category of a given text in {}. Your analysis should be thorough and accurate, reflecting your deep understanding of ESG factors.",
    "Assume the role of an ESG specialist, tasked with categorizing the sentiment of texts in {}. Utilize your expertise to provide precise and insightful sentiment analysis.",
    "Function as an ESG consultant, focusing on assessing the sentiment category of a given text in {}. Your role requires you to leverage your knowledge to deliver accurate sentiment categorizations.",
    "Your role as an authority in ESG involves determining the sentiment category of texts in {}. Apply your specialized skills to ensure that your sentiment assessments are both accurate and insightful.",
    "As an expert in ESG, you are responsible for analyzing and categorizing the sentiment in {}. Your work should highlight your deep understanding of the social and governance aspects of ESG.",
    "You are a specialist in ESG, tasked with determining the sentiment category of a given text in {}. Your analysis should be comprehensive, showcasing your expertise in ESG criteria.",
    "Enact the role of an ESG expert, focusing on the sentiment category in {}. Your expertise will be essential in providing detailed and accurate sentiment analysis.",
    "You're designated as an ESG strategist, responsible for determining the sentiment of texts in {}. Utilize your in-depth knowledge to offer precise sentiment categorizations.",
    "Proceed as an ESG expert, tasked with categorizing the sentiment in {}. Your role requires a thorough understanding of ESG factors to deliver accurate sentiment assessments.",
    "You embody an ESG specialist, focusing on assessing the sentiment category in {}. Your expertise will guide the accurate categorization of sentiment in various ESG-related texts.",
    "Act as an ESG authority, responsible for determining the sentiment category in {}. Apply your specialized knowledge to ensure precise and insightful sentiment analysis.",
    "Your expertise in ESG means you are tasked with categorizing the sentiment in {}. Your thorough understanding of ESG factors will drive accurate sentiment assessments.",
    "Perform as an ESG expert, assessing the sentiment category in {}. Your deep knowledge in ESG will ensure that your sentiment analysis is both precise and insightful.",
    "Your identity as an ESG consultant involves determining the sentiment category in {}. Leverage your expertise to provide accurate and thorough sentiment categorizations.",
    "You represent ESG expertise, responsible for assessing the sentiment in {}. Your specialized knowledge will guide you in delivering accurate and insightful sentiment analysis."
]

# Update datasets
with open('esg_sentiment_train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []

for item in data:
    outputs = item['output'].split(', ')
    
    for output in outputs:
        domain, sentiment = output.split() 
        base_instruction = random.choice(prefix_list).format(domain)
        new_item = {
            "instruction": base_instruction,
            "input": item['input'],
            "output": sentiment
        }
        new_data.append(new_item)

with open('esg_sentiment_train_final.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, indent=4)

print("Modified Training JSON file has been created successfully.")


with open('esg_sentiment_test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

new_data = []

for item in data:
    outputs = item['output'].split(', ')
    for output in outputs:
        domain, sentiment = output.split() 
        base_instruction = random.choice(prefix_list).format(domain)
        new_item = {
            "instruction": base_instruction,
            "input": item['input'],
            "output": sentiment
        }
        new_data.append(new_item)

with open('esg_sentiment_test_final.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, indent=4)

print("Modified Testing JSON file has been created successfully.")

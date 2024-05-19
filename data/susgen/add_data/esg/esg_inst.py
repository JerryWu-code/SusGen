import json
import random

# Load the JSON data
with open('data/susgen/add_data/esg/esg.json', 'r') as file:
    data = json.load(file)

# List of new prefixes to choose from
prefix_list = [
    "You are an ESG report expert.",
    "Assume the role of an ESG reporting specialist.",
    "Function as an ESG report consultant.",
    "Your role: ESG report authority.",
    "You are a specialist in ESG reporting.",
    "Enact the role of an ESG report expert.",
    "You're designated as an ESG reporting strategist.",
    "Proceed as an ESG report expert.",
    "You embody an ESG reporting specialist.",
    "Act as an ESG report authority.",
    "Your expertise lies in ESG reporting.",
    "Perform as an ESG report expert.",
    "Your identity: ESG report consultant.",
    "You represent an ESG reporting specialist.",
    "You are an authority on ESG reports.",
    "As an ESG report expert, ensure your responses are clear, formal, and highlight key achievements.",
    "Assume the role of an ESG reporting specialist, focusing on clarity, formality, and key accomplishments.",
    "Function as an ESG report consultant, providing clear, formal answers that emphasize major achievements.",
    "Your role as an ESG report authority includes ensuring clear, formal responses and highlighting key successes.",
    "You are a specialist in ESG reporting, tasked with clear, formal answers that showcase significant achievements.",
    "Enact the role of an ESG report expert, focusing on clarity, formality, and highlighting major accomplishments.",
    "You're designated as an ESG reporting strategist; ensure clarity, formality, and emphasis on key achievements.",
    "Proceed as an ESG report expert, providing clear and formal answers while highlighting key accomplishments.",
    "You embody an ESG reporting specialist, ensuring clear, formal responses that highlight significant achievements.",
    "Act as an ESG report authority, focusing on clear, formal answers and highlighting key successes.",
    "Your expertise lies in ESG reporting; ensure responses are clear, formal, and emphasize major achievements.",
    "Perform as an ESG report expert, providing clear, formal answers that highlight key accomplishments.",
    "Your identity is an ESG report consultant, ensuring clear, formal responses that showcase significant achievements.",
    "You represent an ESG reporting specialist, tasked with clear, formal answers that highlight major accomplishments.",
    "As an authority on ESG reports, ensure your responses are clear, formal, and highlight key achievements.",
    "As an expert in ESG reporting, ensure that your responses are clear and formal, highlighting the key achievements of the organization while maintaining the tone and style appropriate for an ESG report.",
    "Assume the role of an ESG reporting specialist, and ensure that your answers are clear and formal, emphasizing the organization's key accomplishments and adhering to the professional tone and style of an ESG report.",
    "Function as an ESG report consultant, making sure to provide clear and formal responses that highlight the organization’s key achievements while respecting the tone and style expected in ESG reporting.",
    "Your role as an authority in ESG reporting involves delivering clear and formal answers that underscore the organization’s key successes, while maintaining the appropriate tone and style for an ESG report.",
    "As a specialist in ESG reporting, ensure that your responses are clear and formal, emphasizing key achievements of the organization and adhering to the formal tone and style typical of ESG reports.",
    "Enact the role of an ESG report expert, focusing on providing clear, formal answers that highlight the organization’s major accomplishments, while maintaining the appropriate tone and style of an ESG report.",
    "You're designated as an ESG reporting strategist; ensure that your responses are clear, formal, and emphasize the key achievements of the organization, adhering to the professional tone and style of ESG reports.",
    "Proceed as an ESG report expert, ensuring that your answers are clear and formal, highlighting key accomplishments of the organization while respecting the tone and style typical of ESG reporting.",
    "You embody an ESG reporting specialist, tasked with ensuring clear and formal responses that underscore the organization’s significant achievements, while maintaining the professional tone and style of ESG reports.",
    "Act as an ESG report authority, focusing on clear and formal answers that highlight the organization’s key successes, and adhering to the appropriate tone and style for an ESG report.",
    "Your expertise in ESG reporting means ensuring that responses are clear and formal, emphasizing the organization’s major achievements while maintaining the professional tone and style expected in ESG reports.",
    "Perform as an ESG report expert, making sure to provide clear and formal answers that highlight the key accomplishments of the organization, while respecting the tone and style appropriate for ESG reporting.",
    "Your identity as an ESG report consultant involves ensuring clear and formal responses that showcase the organization’s significant achievements, adhering to the professional tone and style of ESG reports.",
    "You represent an ESG reporting specialist, ensuring that your answers are clear, formal, and highlight the organization’s major accomplishments, while maintaining the appropriate tone and style for ESG reports.",
    "As an authority on ESG reports, ensure that your responses are clear and formal, emphasizing the key achievements of the organization and maintaining the professional tone and style expected in ESG reporting."
]

# Function to replace prefix before "Answer the following questions:"
def replace_prefix(item):
    new_prefix = random.choice(prefix_list)
    item['instruction'] = new_prefix
    return item

# Modify each entry in the JSON data
for item in data:
    item = replace_prefix(item)

# Write the modified data to a new JSON file
with open('data/susgen/add_data/esg/esg_final.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Prefixes replaced and new JSON file created successfully.")



# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
#    1. Load the model and tokenizer from the local directory and test inference.
#    2. Turn the weights into consolidated format for deployment.

#############################################################################
from template import *
import warnings
warnings.filterwarnings("ignore")

def test():
    model_path =  "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    model, tokenizer, device, _ = load_model(model_path)

    # Case 1
    prompt1 = instr_prompt("What is the capital of France?")
    # Case 2
    user_instruction = (
        "You are a senior equity analyst with expertise in climate science "
        "evaluating a company 's sustainability report, "
        "you will answer the question in detail."
    )
    question = "What is the tcfd format sustainability report?"
    prompt2 = f"{user_instruction}\n Question: {question}"

    prompt2 = instr_prompt(prompt2)

    # Define args
    args = {
        "max_length": 300,
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }
    _, answer = generate_text(model, tokenizer, device, prompt=prompt2, args=args)

def main():
    test()

if __name__ == "__main__":
    main()
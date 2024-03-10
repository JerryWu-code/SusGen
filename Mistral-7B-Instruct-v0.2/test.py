# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
"""
    1. Load the model and tokenizer from the local directory and test inference.
    2. Turn the weights into consolidated format for deployment.
"""

#############################################################################
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_model(model_path="./"):
    # 1.Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # 2.Load the model and move to GPU
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    return model, tokenizer, device, config

# Function to generate text
def generate_text(model, tokenizer, device, prompt, max_length=1024, temperature=0.7, 
    top_p=0.9, top_k=40, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.module.generate(  # Use model.module to access the wrapped model
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences
        )
    
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    return generated_text

def turn_weights_to_consolidated_format(model, tokenizer, model_path="./"):
    if hasattr(model, 'module'):
        # The original model is stored in the `module` attribute
        model = model.module
    else:
        # The model is not wrapped with DataParallel, so use it directly
        model = model
    
    # 1.Save the model in consolidated format & name it "consolidated.00.pth"
    # torch.save(model.state_dict(), 'consolidated.00.pth')
    # 2.Save the tokenizer in consolidated format
    # tokenizer.save_pretrained(model_path, save_format="consolidated")

def main():
    # 1.Load the model and tokenizer
    model, tokenizer, device, config = load_model(model_path="./")
    # 2.Set the model to evaluation mode
    model.eval()

    # 3.Define the prompt & generate text
    prompt = "Write a short story about a robot that learns to love."
    prompt_ = "{user_instruction} \n Question:{question}".format(
        user_instruction=(
                "You are a senior equity analyst with expertise in climate science"
                "evaluating a company 's sustainability report, "
                "you will answer the question in detail."
        ),
        question="What is the company's carbon footprint?",
    )
    generated_text = generate_text(model, tokenizer, device, prompt)
    print(generated_text)
    
    # 4.Turn the weights into consolidated format for deployment
    # turn_weights_to_consolidated_format(model, tokenizer, model_path="./")

if __name__ == "__main__":
    main()
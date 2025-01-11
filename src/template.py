# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
#    1. Load the model and tokenizer from the local directory and test inference.
#    2. Turn the weights into consolidated format for deployment.

#############################################################################
import torch, os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, TextStreamer
)
import warnings
from peft import PeftModel
warnings.filterwarnings("ignore")

def load_model(model_path, lora_path=False, quantization='bf16'):
    # Load the tokenizer & set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if quantization == 'int4':
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
    elif quantization == 'int8':
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", 
        )

    if quantization == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to(device)

    config = AutoConfig.from_pretrained(model_path)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.bfloat16)

    return model, tokenizer, device, config

def generate_text(model, tokenizer, device, prompt, args):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Streaming generation
    # streamer = TextStreamer(tokenizer)
    #####
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            repetition_penalty=1.2,
            **args,
            # streamer=streamer # Streaming generation
        )
    
    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    question = result.split("### Response:")[0]
    answer = result.split("### Response:\n")[1]

    return question, answer

def turn_weights_to_consolidated_format(model, tokenizer, model_path):
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

def instr_prompt(content):
    final_prompt = "[INST] {} [/INST]".format(content)
    return final_prompt

def main():
    # 1.Load the model and tokenizer
    ckpt_folder = "../ckpts"
    base_model = "Mistral-7B-Instruct-v0.3-hf"

    # base_model = "Meta-Llama-3-8B-Instruct-hf"

    # ckpt_folder = "../../../results"
    # base_model = "SusGen_GPT_Mistral_Instruct_v0.3_30k_10epoch_merged"
    model, tokenizer, device, config = load_model(
        model_path=os.path.join(ckpt_folder, base_model),
        # lora_path="../results/SusGen30k-int4-adamw32_Mistral-7B-v0.3/checkpoint-1406",
        quantization='bf16')
    # 2.Set the model to evaluation mode
    model.eval()

    # 3.Define the prompt & generate text
    #  1) Prompt
    user_instruction = (
        "Instruction:\nYou are a senior equity analyst with expertise in climate science "
        "evaluating a company 's sustainability report, "
        "you will answer the question in detail."
    )
    question = "What is the tcfd format sustainability report? Only output one sentence."
    prompt = f"{user_instruction}\n Question:\n{question}"

    final_prompt = instr_prompt(content = prompt) + "### Response:\n"
    #  2) Set configuration
    args = {
        # "max_length": 512,
        "temperature": 0.2,
        "do_sample": True,  # "If set to False greedy decoding is used. Otherwise sampling is used.
        "top_p": 0.9,
        "top_k": 40,
        "max_new_tokens": 1024, 
        "num_return_sequences": 1
    }
    #  3) Generate text
    question, answer = generate_text(model, tokenizer, device, final_prompt, args)
    print(f"Question:\n{'-' * 10}\n{question.strip()}\n{'=' * 100}")
    print(f"Answer:\n{'-' * 10}\n{answer.strip()}\n{'=' * 100}")
    
    # 4.Turn the weights into consolidated format for deployment
    # turn_weights_to_consolidated_format(model, tokenizer, model_path="./")

if __name__ == "__main__":
    main()
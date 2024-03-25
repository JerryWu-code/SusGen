# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to do the inference mistral-7B with lora

#############################################################################
import torch, warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from template import instr_prompt
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

device = torch.device("cuda")
model_path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
lora_alpaca = "results/Mistral-7B_alpaca-lora/ckpts/final"
lora_susgenv1 = "results/Mistral-7B_susgenv1-lora/ckpts/checkpoint-900"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double=True,
    bnb_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_bos_token=True,
    add_eos_token=True,
    padding_side="left",
    padding="max_length",
    use_fast=True,
    # trust_remote_code=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

def load_alpaca():
    # lora_config = PeftConfig.from_pretrained(lora_alpaca)
    model = PeftModel.from_pretrained(base_model, lora_alpaca, torch_dtype=torch.bfloat16)
    return model

def load_susgenv1():
    # lora_config = PeftConfig.from_pretrained(lora_susgenv1)
    model_with_adapter = PeftModel.from_pretrained(base_model, lora_alpaca, torch_dtype=torch.bfloat16)
    model = model_with_adapter.merge_and_unload()
    model = PeftModel.from_pretrained(model, lora_susgenv1, torch_dtype=torch.bfloat16)

    return model

def inference(model, model_input, mode="alpaca"):
    model.eval()
    with torch.no_grad():
        result = tokenizer.decode(
            model.generate(
                **model_input, 
                max_new_tokens=512, 
                repetition_penalty=1.15)[0], 
                skip_special_tokens=True
            )
        if mode == "alpaca":
            question = result.split(" [/INST]")[0].split("[INST] ")[1]
            answer = result.split("[/INST] [/INST] ")[1]
            return question, answer
        elif mode == "susgenv1":
            question = result.split(" [/INST]")[0].split("[INST] ")[1]
            answer = result.split("[/INST] ")[1]
            return question, answer
        else:
            print(result)
            return result
    
def main():
    mode = "susgenv1" # or "alpaca"
    test_prompt = "What is tcfd format in the context of climate change?"

    if mode == "alpaca":
        model = load_alpaca()
    elif mode == "susgenv1":
        model = load_susgenv1()
    
    model_input = tokenizer(instr_prompt(test_prompt), return_tensors="pt").to(device)
    question, answer = inference(model, model_input, mode=None)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()

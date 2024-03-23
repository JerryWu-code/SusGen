# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to do the inference mistral-7B with lora

#############################################################################
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda")
model_path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
lora_path = "results/Mistral-7B_alpaca-lora/ckpts/checkpoint-1000"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double=True,
    bnb_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    add_bos_token=True,
    add_eos_token=True,
    padding_side="left",
    padding="max_length",
    # trust_remote_code=True
)

lora_config = PeftConfig.from_pretrained(lora_path)
print(lora_config)
model = PeftModel.from_pretrained(base_model, lora_path)

test_prompt = "What is tcfd format in the context of climate change?"
model_input = tokenizer(test_prompt, return_tensors="pt").to(device)

model.eval()
with torch.no_grad():
    print(tokenizer.decode(
        model.generate(
            **model_input, 
            max_new_tokens=512, 
            repetition_penalty=1.15)[0], 
            skip_special_tokens=True
        )
    )

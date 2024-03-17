# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to do inference of the Mistral.

#############################################################################
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path, device="cuda:7"):
    # 1.Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # 2.Load the model and move to GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        pad_token_id=tokenizer.eos_token_id
    )
    device = torch.device(device)
    model = model.to(device)

    return model, tokenizer


# Function to generate text
def generate_text(model, tokenizer, device, prompt, max_length=300, temperature=0.5,
                top_p=0.9, top_k=40, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            # attention_mask=input_ids.ne(tokenizer.pad_token_id).long()
        )
    
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    return generated_text

def prompt(text):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You only output one sentence following the instructions:\n"
            "1.Remove all the Chinese characters of the text and bracket annotation.\n"
            "2.Rephrase the remain sentence.\nText: {text}")).format(text=text)
    )
    return final_prompt

def parse_result(generated_text):
    question = generated_text[0].split("[/INST]")[0].split("[INST] ")[1]
    answer = generated_text[0].split("[/INST] ")[1]
    return question, answer

def process(model, tokenizer, device, text):
    final_prompt = prompt(text)
    generated_text = generate_text(model, tokenizer, device, final_prompt)
    question, answer = parse_result(generated_text)
    return question, answer

def main():
    # 1.Load the model and tokenizer
    path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    device = "cuda:1"
    model, tokenizer = load_model(model_path=path, device=device)
    # 2.Set the model to evaluation mode
    model.eval()
    # 3.Define the prompt & generate text
    text = "Slow and romantic music, such as classical music or jazz, is suitable for a romantic dinner. [slow]\n（中文翻译：Q4. 什么样的音乐适合浪漫的晚餐？\nA4. 缓慢而浪漫的音乐，如古典音乐或爵士乐，适合浪漫的晚餐。【慢】）"
    question, answer = process(model, tokenizer, device, text)
    print(question)
    print(answer)

if __name__ == "__main__":
    main()
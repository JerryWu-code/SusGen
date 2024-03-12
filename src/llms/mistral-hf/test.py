# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
#    1. Load the model and tokenizer from the local directory and test inference.
#    2. Turn the weights into consolidated format for deployment.

#############################################################################
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_model(model_path="../../../ckpts/Mistral-7B-Instruct-v0.2-hf"):
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

def generate_text(model, tokenizer, device, prompt, max_length=256, temperature=0.5, 
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

def prompt(text):
    final_prompt = "[INST] {content} [/INST]".format(
        content = ((
            "You only output one sentence following the instructions:\n"
            "1.Remove all the Chinese characters of the text and bracket annotation.\n"
            "2.Rephrase the remain sentence.\nText: {text}")).format(text=text)
    )
    return final_prompt

def test():
    test = "Write a short story about a robot that learns to love."
    final_prompt = prompt(test)
    print(final_prompt)

def main():
    # 1.Load the model and tokenizer
    path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
    model, tokenizer, device, config = load_model(model_path=path)
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
    prompt__ = ("<s> [INST] Compose a welcome email within 200 words for new customers who have just made their first purchase with your product."
                "Start by expressing your gratitude for their business, and then convey your excitement for having them as a customer."
                "Include relevant details about their recent order. Sign the email with \"The Fun Shop Team\".\n"

                "Order details:\n"
                "- Customer name: Anna\n"
                "- Product: hat\n"
                "- Estimate date of delivery: Feb. 25, 2024\n"
                "- Return policy: 30 days [/INST]\n")
    prompt___ = "[INST] {content} [/INST]".format(
        content = "You only output one sentence following the instructions:\n1.Remove all the Chinese characters of the text and bracket annotation.\n2.Rephrase the remain sentence.\nText: {text}".format(
            text="Slow and romantic music, such as classical music or jazz, is suitable for a romantic dinner. [slow]\n（中文翻译：Q4. 什么样的音乐适合浪漫的晚餐？\nA4. 缓慢而浪漫的音乐，如古典音乐或爵士乐，适合浪漫的晚餐。【慢】）"
        ),
    )
    generated_text = generate_text(model, tokenizer, device, prompt___)
    print("Prompt:\n", generated_text[0].split("[/INST]")[0].split("[INST] ")[1])
    print("Answer:\n", generated_text[0].split("[/INST] ")[1])
    
    # 4.Turn the weights into consolidated format for deployment
    # turn_weights_to_consolidated_format(model, tokenizer, model_path="./")

if __name__ == "__main__":
    main()
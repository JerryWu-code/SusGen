# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to:
#    1. Load the model and tokenizer from the local directory and test inference.
#    2. Turn the weights into consolidated format for deployment.

#############################################################################
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path="../../../ckpts/Mistral-7B-Instruct-v0.2-hf"):
    # 1.Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # 2.Load the model and move to GPU
    config = AutoConfig.from_pretrained(model_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    return model, tokenizer, device, config

def generate_text(model, tokenizer, device, prompt, args):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Streaming generation
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer)
    #####
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=args["max_length"],
            do_sample=args["do_sample"],
            temperature=args["temperature"],
            top_p=args["top_p"],
            top_k=args["top_k"],
            num_return_sequences=args["num_return_sequences"],
            repetition_penalty=1.2,
            streamer=streamer # Streaming generation
        )
    
    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)

    prompt = generated_text[0].split("[/INST]")[0].split("[INST] ")[1]
    answer = generated_text[0].split("[/INST] ")[1]

    # print("------------------------------------------------------------")
    # print("Prompt:\n", prompt)
    # print("------------------------------------------------------------")
    # print("Answer:\n", answer)

    return prompt, answer

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
    ckpt_folder = "../../../ckpts"
    base_model = "Mistral-7B-Instruct-v0.3-hf"
    # base_model = "Meta-Llama-3-8B-Instruct-hf"

    ckpt_folder = "../../../results"
    base_model = "SusGen_GPT_Mistral_Instruct_v0.3_30k_10epoch_merged"
    model, tokenizer, device, config = load_model(model_path=os.path.join(ckpt_folder, base_model))
    # 2.Set the model to evaluation mode
    model.eval()

    # 3.Define the prompt & generate text
    #  1) Prompt
    user_instruction = (
        "Instruction:\nYou are a senior equity analyst with expertise in climate science "
        "evaluating a company 's sustainability report, "
        "you will answer the question in detail."
    )
    question = "What is the tcfd format sustainability report?"
    prompt = f"{user_instruction}\n Question:\n{question}"

    final_prompt = instr_prompt(content = prompt)
    #  2) Set configuration
    args = {
        "max_length": 512,
        "temperature": 0.2,
        "do_sample": True,  # "If set to False greedy decoding is used. Otherwise sampling is used.
        "top_p": 0.9,
        "top_k": 40,
        "num_return_sequences": 1
    }
    #  3) Generate text
    _, answer = generate_text(model, tokenizer, device, final_prompt, args)
    # print(answer)
    
    # 4.Turn the weights into consolidated format for deployment
    # turn_weights_to_consolidated_format(model, tokenizer, model_path="./")

if __name__ == "__main__":
    main()
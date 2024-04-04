# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to do the inference mistral-7B with lora

#############################################################################
import torch, warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from template import instr_prompt
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

device = torch.device("cuda")
model_path = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
lora_alpaca = "results/Mistral-7B_alpaca-lora/ckpts/final"
lora_susgenv1 = "results/Mistral-7B_susgenv1-lora/ckpts/checkpoint-900" # 200, 550, 900

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
                max_new_tokens=1024, 
                repetition_penalty=1.15)[0], 
                skip_special_tokens=True
            )
        if mode == "susgen_v1_round1" or mode == "susgen_v1_round2":
            question = result.split(" [INST] [/INST]### Response: ")[0].split("[INST] ### Instruction: ")[1]
            answer = result # .split("Response: ")[1]
            # .split("Response: ")[1]
            # .split("[/INST] ")[1]
            return question, answer
        else:
            print(result)
            return result

def prompt_adjust(prompt):
    # [INST] ### Instruction: {0}\n\n [/INST]### Response: {1}
    return f"### Instruction: {prompt}\n\n"

def main():
    # mode = "susgen_v1_round1"
    mode = "susgen_v1_round2"

    # 1. TCFD general
    # test_prompt = "What is tcfd format in the context of climate change?"
    # 2. TCFD detailed
    # test_prompt = "Please explain the concept of 'tcfd' in the context of climate change in detail."
    # 3. Origin last one
    # test_prompt = "Imagine you are a leading expert in climate-related financial disclosures, specializing in the TCFD framework. Your role encompasses deep insights into how organizations can effectively disclose information regarding Governance, Strategy, Risk Management, and Metrics & Targets in relation to climate-related risks and opportunities. Your task is to assist in generating detailed, accurate, and insightful answers for a QA session focused on enhancing an organization's TCFD report. For each of the following sections, provide expert-level responses based on the core requirements of TCFD disclosures: \nProvide a comprehensive overview of the metrics and targets established by the organization to monitor climate-related risks and opportunities. Detail the benchmarks, time frames, and progress measurement approaches. Explain how these metrics align with the organization's overall sustainability and climate strategy.\nAnswer the following questions: \n1. Describe the targets used by the organization to manage climate-related risks and opportunities and performance against targets."
    # 4. TCFD format question
    test_prompt = (
        # "我将给你一段任务要求，最后请你使用中文进行回答"
        "Now you are a expert in esg and climate change, and you are asked to write sustainability report by answering the question following the below instruction: \n"
        "Instruction: \n"
        "1. Answer the question in the context of TCFD sustainability report format. \n"
        "2. You need to write this for a car company anonymously in detail. \n"
        "3. What you write should follow the text below: \n"
        "3. The final answer should be formatted in one to three paragraphs, within 500 words. \n"
        "Question: \n"
        "Our company has >48,000 tonnes of Greenhouse Gas (GHG) emissions reduced through zero-emission "
        "transportation modes (walkers/cyclists), low emission rental vehicles (EVs/hybrids)11 and "
        "efficiency optimisation. Company has >200,000 trees planted and ~30,000 carbon credits "
        "directed to protect and conserve forests across."
        "Describe the targets used by organizations to manage climate-related risks and opportunities and performance against targets."
        "Text: {text}"
        ) 
    # test_prompt = (
    #     "Now you are a expert in esg and climate change, and you are asked to write sustainability report by answering the question following the below instruction: \n"
    #     "Instruction: \n"
    #     "1. Answer the question in the context of TCFD sustainability report format. \n"
    #     "3. You need to write this for a car company anonymously in detail. \n"
    #     "3. The final answer should be formatted within three paragraphs, within 500 words in total. \n"
    #     "Question: \n"
    #     "Describe the targets used by organizations to manage climate-related risks and opportunities and performance against targets."
    #     )

    if mode == "susgen_v1_round1":
        model = load_alpaca()
    elif mode == "susgen_v1_round2":
        model = load_susgenv1()
    
    model_input = tokenizer(
        instr_prompt(prompt_adjust(test_prompt)), return_tensors="pt").to(device)
    question = test_prompt
    _, answer = inference(model, model_input, mode=mode)
    print(f"\n{'=' * 100}\nModel: {mode}\n{'-' * 10}")
    print(f"Question:\n{'-' * 10}\n{question}\n{'=' * 100}")
    print(f"Answer:\n{'-' * 10}\n{answer}\n{'=' * 100}")

if __name__ == "__main__":
    main()

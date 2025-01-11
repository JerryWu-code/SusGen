# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to do the inference mistral-7B with lora

#############################################################################
import torch, warnings, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import PeftModel
from utils.prompt_template import *
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

device = torch.device("cuda")
# lora_susgenv2_1 = "../src/llms/mistral-hf/results/Mistral-7B-Instruct_susgen30k-mistral-int4-adamw32/ckpts/checkpoint-186"
# lora_susgenv2_2_1epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct_susgen30k-int4-adamw32_new/ckpts/checkpoint-233"
# lora_susgenv2_2_2epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct_susgen30k-int4-adamw32_new/ckpts/checkpoint-466"
# lora_susgenv2_2_3epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct_susgen30k-int4-adamw32_new/ckpts/checkpoint-699"
# lora_susgenv2_2_5epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct_susgen30k-int4-adamw32_new/ckpts/checkpoint-1165"
# lora_susgenv2_3_1epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-233"
# lora_susgenv2_3_2epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-466"
# lora_susgenv2_3_3epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-699"
# lora_susgenv2_3_4epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-933"
# lora_susgenv2_3_5epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-1165"
# lora_susgenv2_3_10epoch = "../src/llms/mistral-hf/results/Mistral-7B-Instruct-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-2330"
# lora_susgenv2_b1_1epoch = "../src/llms/mistral-hf/results/Mistral-7B-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-233"
# lora_susgenv2_b1_2epoch = "../src/llms/mistral-hf/results/Mistral-7B-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-466"
# lora_susgenv2_b1_3epoch = "../src/llms/mistral-hf/results/Mistral-7B-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-699"
# lora_susgenv2_b1_4epoch = "../src/llms/mistral-hf/results/Mistral-7B-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-933"
# lora_susgenv2_b1_5epoch = "../src/llms/mistral-hf/results/Mistral-7B-v0.3-hf_susgen30k-int4-adamw32_new/ckpts/checkpoint-1165"

def load_lora(base_model, lora_path):
    # lora_config = PeftConfig.from_pretrained(lora_alpaca)
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.bfloat16)
    return model

def load_susgenv1():
    # lora_config = PeftConfig.from_pretrained(lora_susgenv1)
    model_with_adapter = PeftModel.from_pretrained(base_model, lora_alpaca, torch_dtype=torch.bfloat16)
    model = model_with_adapter.merge_and_unload()
    model = PeftModel.from_pretrained(model, lora_susgenv1, torch_dtype=torch.bfloat16)

    return model

def inference(model, tokenizer, prompt, prompt_template, mode):
    model.eval()
    input_ids = tokenizer.encode(prompt_template(prompt), return_tensors="pt").to(device)
    # Streaming generation
    streamer = TextStreamer(tokenizer)
    with torch.no_grad():
        result = tokenizer.batch_decode(
            model.generate(
                input_ids=input_ids, 
                max_new_tokens=512, 
                repetition_penalty=1.5,
                streamer=streamer, # Streaming generation
                num_return_sequences=1
                ), 
                skip_special_tokens=True,  
            )[0]
        if mode == "susgen_v1":
            question = result.split(" [INST] [/INST]### Response: ")[0].split("[INST] ### Instruction: ")[1]
            answer = result.split("Response: ")[1]
            return question, answer
        # if mode == "SusGen_GPT_Mistral_v0.2":
        #     question = result.split("[INST] ")[1].split("\n\n [/INST]")[0]
        #     answer = result.split("### Response:\n")[1]
        #     return question, answer
        if mode == "SusGen_GPT_Mistral_v0.3" or mode == "SusGen_GPT_Mistral_v0.2":
            question = result.split("### Response:")[0]
            answer = result.split("### Response:\n")[1]
            return question, answer
        else:
            print("="*50)
            print(result)
            return result

def main():
    # 1. TCFD general
    # test_prompt = "What is tcfd format in the context of climate change?"
    # 2. TCFD detailed
    # test_prompt = "Please explain the concept of 'tcfd' in the context of climate change in detail."
    # 3. Origin last one
    # test_prompt = "Imagine you are a leading expert in climate-related financial disclosures, specializing in the TCFD framework. Your role encompasses deep insights into how organizations can effectively disclose information regarding Governance, Strategy, Risk Management, and Metrics & Targets in relation to climate-related risks and opportunities. Your task is to assist in generating detailed, accurate, and insightful answers for a QA session focused on enhancing an organization's TCFD report. For each of the following sections, provide expert-level responses based on the core requirements of TCFD disclosures: \nProvide a comprehensive overview of the metrics and targets established by the organization to monitor climate-related risks and opportunities. Detail the benchmarks, time frames, and progress measurement approaches. Explain how these metrics align with the organization's overall sustainability and climate strategy.\nAnswer the following questions: \n1. Describe the targets used by the organization to manage climate-related risks and opportunities and performance against targets."
    # 4. TCFD format question
    # test_prompt = (
    #     "Now you are a expert in esg and climate change, and you are asked to write sustainability report by answering the question following the below instruction: \n"
    #     "Instruction: \n"
    #     "1. Answer the question in the context of TCFD sustainability report format. \n"
    #     "3. You need to write this for a car company anonymously in detail. \n"
    #     "3. The final answer should be formatted within three paragraphs, within 500 words in total. \n"
    #     "Question: \n"
    #     "Describe the targets used by organizations to manage climate-related risks and opportunities and performance against targets."
    #     )
    test_prompt = {
        "instruction":(
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
            "directed to protect and conserve forests across."),
        "input": "Describe the targets used by organizations to manage climate-related risks and opportunities and performance against targets."
        # "Text: {text}"
    }
    test_qa = {
        "instruction": "Please provide your answer to the financial question based on the context.",
        "input": "entergy corporation and subsidiaries management 2019s financial discussion and analysis a result of the entergy louisiana and entergy gulf states louisiana business combination , results of operations for 2015 also include two items that occurred in october 2015 : 1 ) a deferred tax asset and resulting net increase in tax basis of approximately $ 334 million and 2 ) a regulatory liability of $ 107 million ( $ 66 million net-of-tax ) as a result of customer credits to be realized by electric customers of entergy louisiana , consistent with the terms of the stipulated settlement in the business combination proceeding . see note 2 to the financial statements for further discussion of the business combination and customer credits . results of operations for 2015 also include the sale in december 2015 of the 583 mw rhode island state energy center for a realized gain of $ 154 million ( $ 100 million net-of-tax ) on the sale and the $ 77 million ( $ 47 million net-of-tax ) write-off and regulatory charges to recognize that a portion of the assets associated with the waterford 3 replacement steam generator project is no longer probable of recovery . see note 14 to the financial statements for further discussion of the rhode island state energy center sale . see note 2 to the financial statements for further discussion of the waterford 3 write-off . results of operations for 2014 include $ 154 million ( $ 100 million net-of-tax ) of charges related to vermont yankee primarily resulting from the effects of an updated decommissioning cost study completed in the third quarter 2014 along with reassessment of the assumptions regarding the timing of decommissioning cash flows and severance and employee retention costs . see note 14 to the financial statements for further discussion of the charges . results of operations for 2014 also include the $ 56.2 million ( $ 36.7 million net-of-tax ) write-off in 2014 of entergy mississippi 2019s regulatory asset associated with new nuclear generation development costs as a result of a joint stipulation entered into with the mississippi public utilities staff , subsequently approved by the mpsc , in which entergy mississippi agreed not to pursue recovery of the costs deferred by an mpsc order in the new nuclear generation docket . see note 2 to the financial statements for further discussion of the new nuclear generation development costs and the joint stipulation . net revenue utility following is an analysis of the change in net revenue comparing 2015 to 2014 . amount ( in millions ) .\n||amount ( in millions )|\n|2014 net revenue|$ 5735|\n|retail electric price|187|\n|volume/weather|95|\n|waterford 3 replacement steam generator provision|-32 ( 32 )|\n|miso deferral|-35 ( 35 )|\n|louisiana business combination customer credits|-107 ( 107 )|\n|other|-14 ( 14 )|\n|2015 net revenue|$ 5829|\nthe retail electric price variance is primarily due to : 2022 formula rate plan increases at entergy louisiana , as approved by the lpsc , effective december 2014 and january 2015 ; 2022 an increase in energy efficiency rider revenue primarily due to increases in the energy efficiency rider at entergy arkansas , as approved by the apsc , effective july 2015 and july 2014 , and new energy efficiency riders at entergy louisiana and entergy mississippi that began in the fourth quarter 2014 ; and 2022 an annual net rate increase at entergy mississippi of $ 16 million , effective february 2015 , as a result of the mpsc order in the june 2014 rate case . see note 2 to the financial statements for a discussion of rate and regulatory proceedings. .\nQuestion: what is the net change in net revenue during 2015 for entergy corporation?\nAnswer:",
        "output": "94.0"
    }
    test_sa = {
        "instruction": "What is the sentiment of the following fiscal analysis: Positive, Negative, or Neutral?",
        "input": "$BBRY Actually lost .03c per share if U incl VZ as no debt and 3.1 in Cash.",
        "output": "negative"
    }
    test_ner = {
        "instruction": "From the sentences in financial agreements in U.S. SEC filings, recognize named entities representing either a person ('PER'), an organization ('ORG'), or a location ('LOC'). Format your answer as: 'entity name, entity type'.",
        "input": "SUBORDINATED LOAN AGREEMENT HERBERT SMITH LLP Page 1 of 12 7 - December 2007 TABLE OF CONTENTS Clause Headings Page 1 .",
        "output": "HERBERT SMITH, PER"
    }
    test_ner_cls = {
        "instruction": "With the input text as context, identify the entity type of 'EVERGREEN SOLAR'.\nOptions: person, organization, location",
        "input": "Subordinated Loan Agreement - Silicium de Provence SAS and Evergreen Solar Inc . 7 - December 2007 [ HERBERT SMITH LOGO ] ................................ 2007 SILICIUM DE PROVENCE SAS and EVERGREEN SOLAR , INC .",
        "output": "organization"
    }
    test_sa_cls = {
        "instruction": "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.",
        "input": "$PYPL great vol flow trade here.. Love this company long time.",
        "output": "positive"
    }

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    )

    # model_path = "../ckpts/Mistral-7B-v0.2-hf"
    # model_path = "../ckpts/Mistral-7B-v0.3-hf"
    # model_path = "../ckpts/Mistral-7B-Instruct-v0.2-hf"
    model_path = "../ckpts/Mistral-7B-Instruct-v0.3-hf"
    if 'v0.2' in model_path:
        mode = "SusGen_GPT_Mistral_v0.2"
    elif 'v0.3' in model_path:
        mode = "SusGen_GPT_Mistral_v0.3"
    result_path = "../results"
    save_name = mode + "_30k_3epoch_base_lora"

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=True, padding_side="left", padding="max_length", use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
        quantization_config=bnb_config, attn_implementation="flash_attention_2")

    if mode == "SusGen_GPT_Mistral_v0.2":
        model = load_lora(base_model, "../results/SusGen30k-int4-adamw32_Mistral-7B-v0.2/checkpoint-699").to(device)
        # model = load_lora(base_model, lora_susgenv2_2_5epoch).to(device)
    elif mode == "SusGen_GPT_Mistral_v0.3":
        model = load_lora(base_model, "../results/SusGen30k-int4-adamw32_Mistral-7B-v0.3/checkpoint-1406").to(device)
        # model = load_lora(base_model, lora_susgenv2_3_3epoch).to(device)

    question, answer = inference(
        model, tokenizer, prompt=test_qa, prompt_template=mistral_formal_infer, mode=mode)
    print(f"\n{'=' * 100}\nModel: {save_name}\n{'-' * 10}")
    print(f"Question:\n{'-' * 10}\n{question.strip()}\n{'=' * 100}")
    print(f"Answer:\n{'-' * 10}\n{answer.strip()}\n{'=' * 100}")

    commit_message = "Initial commit of the model"
    # model.save_pretrained(os.path.join(result_path, "adapters", save_name))
    model.push_to_hub(f"WHATX/{save_name}", private=True, commit_message=commit_message)
    # model = model.merge_and_unload()
    # model.save_pretrained(os.path.join(result_path, "merged_model", save_name + "_merged"))
    # tokenizer.save_pretrained(os.path.join(result_path, "merged_model", save_name + "_merged"))

if __name__ == "__main__":
    main()

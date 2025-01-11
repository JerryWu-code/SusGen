import os, json, time, gc, shutil
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from utils.prompt_template import *

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
test_finqa = {
        "instruction": "Please answer the given financial question based on the context.",
        "input": "entergy corporation and subsidiaries management 2019s financial discussion and analysis a result of the entergy louisiana and entergy gulf states louisiana business combination , results of operations for 2015 also include two items that occurred in october 2015 : 1 ) a deferred tax asset and resulting net increase in tax basis of approximately $ 334 million and 2 ) a regulatory liability of $ 107 million ( $ 66 million net-of-tax ) as a result of customer credits to be realized by electric customers of entergy louisiana , consistent with the terms of the stipulated settlement in the business combination proceeding . see note 2 to the financial statements for further discussion of the business combination and customer credits . results of operations for 2015 also include the sale in december 2015 of the 583 mw rhode island state energy center for a realized gain of $ 154 million ( $ 100 million net-of-tax ) on the sale and the $ 77 million ( $ 47 million net-of-tax ) write-off and regulatory charges to recognize that a portion of the assets associated with the waterford 3 replacement steam generator project is no longer probable of recovery . see note 14 to the financial statements for further discussion of the rhode island state energy center sale . see note 2 to the financial statements for further discussion of the waterford 3 write-off . results of operations for 2014 include $ 154 million ( $ 100 million net-of-tax ) of charges related to vermont yankee primarily resulting from the effects of an updated decommissioning cost study completed in the third quarter 2014 along with reassessment of the assumptions regarding the timing of decommissioning cash flows and severance and employee retention costs . see note 14 to the financial statements for further discussion of the charges . results of operations for 2014 also include the $ 56.2 million ( $ 36.7 million net-of-tax ) write-off in 2014 of entergy mississippi 2019s regulatory asset associated with new nuclear generation development costs as a result of a joint stipulation entered into with the mississippi public utilities staff , subsequently approved by the mpsc , in which entergy mississippi agreed not to pursue recovery of the costs deferred by an mpsc order in the new nuclear generation docket . see note 2 to the financial statements for further discussion of the new nuclear generation development costs and the joint stipulation . net revenue utility following is an analysis of the change in net revenue comparing 2015 to 2014 . amount ( in millions ) .\n||amount ( in millions )|\n|2014 net revenue|$ 5735|\n|retail electric price|187|\n|volume/weather|95|\n|waterford 3 replacement steam generator provision|-32 ( 32 )|\n|miso deferral|-35 ( 35 )|\n|louisiana business combination customer credits|-107 ( 107 )|\n|other|-14 ( 14 )|\n|2015 net revenue|$ 5829|\nthe retail electric price variance is primarily due to : 2022 formula rate plan increases at entergy louisiana , as approved by the lpsc , effective december 2014 and january 2015 ; 2022 an increase in energy efficiency rider revenue primarily due to increases in the energy efficiency rider at entergy arkansas , as approved by the apsc , effective july 2015 and july 2014 , and new energy efficiency riders at entergy louisiana and entergy mississippi that began in the fourth quarter 2014 ; and 2022 an annual net rate increase at entergy mississippi of $ 16 million , effective february 2015 , as a result of the mpsc order in the june 2014 rate case . see note 2 to the financial statements for a discussion of rate and regulatory proceedings. .\nQuestion: what is the net change in net revenue during 2015 for entergy corporation?\nAnswer:",
        "output": "94.0"
    }

def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.2f} seconds to execute.")
        return result
    return wrapper

def convert_vllm_lora(lora_path, new_lora_path):
    import safetensors.torch
    tensors = safetensors.torch.load_file(os.path.join(lora_path, "adapter_model.safetensors"))

    nonlora_keys = []
    for k in list(tensors.keys()):
        if "lora" not in k:
            nonlora_keys.append(k)

    print(nonlora_keys) # just take a look what they are

    for k in nonlora_keys:
        del tensors[k]

    # if new path parent dir does not exist, create it
    os.makedirs(os.path.dirname(new_lora_path), exist_ok=True)
    safetensors.torch.save_file(tensors, os.path.join(new_lora_path, "adapter_model.safetensors"))

    # copy files in lora_path to new_lora_path
    for f in os.listdir(lora_path):
        if os.path.isfile(os.path.join(lora_path, f)) and f not in ["adapter_model.safetensors", "README.md"]:
            shutil.copy(os.path.join(lora_path, f), new_lora_path)

def load_model(model_path, lora_path=False, gpu_memory_utilization=0.9, max_model_len=4096, max_num_seqs=128):
    if lora_path:
        llm = LLM(
            model=model_path,
            enable_lora=True,
            # quantization="bitsandbytes",
            # load_format="bitsandbytes",
            # qlora_adapter_name_or_path=lora_path,
            gpu_memory_utilization=gpu_memory_utilization,
            # max_seq_len_to_capture=2048, # promt + response tokens
            max_model_len=max_model_len, # 2048 & 128 works for 24GB GPU
            # when prompt & res tokens exceed 2048, would directly return
            max_num_seqs=max_num_seqs, # set to 512 * #cards
            # tensor_parallel_size=2, # set to #cards, default is 1
        )

    else:
        llm = LLM(
            model=model_path, 
            gpu_memory_utilization=0.9,
            # max_seq_len_to_capture=2048, # promt + response tokens
            max_model_len=4096,
            max_num_seqs=128, # set to 512 * #cards
            # tensor_parallel_size=2, # set to #cards, default is 1
        )
    return llm

@time_logger
def inference(llm, prompts, args, verbose=False, lora_path=False):
    sampling_params = SamplingParams(stop=["### Response:\n"], **args)

    if lora_path:
        outputs = llm.generate(
            prompts, sampling_params,
            lora_request=LoRARequest("susgen_adapter", 1, lora_path)
        )
    else:
        outputs = llm.generate(prompts, sampling_params)
    # out_json = []
    out_list = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # out_json.append({"prompt": prompt, "generated_text": generated_text})
        out_list.append(generated_text)
        if verbose:
            print(f"Prompt: {prompt!r}\n\n{'*'*20}\n\nGenerated text: {generated_text!r}\n{'='*80}\n")

    # with open ("output.json", "w") as f:
    #     json.dump(out_json, f, indent=4)
    return out_list

def main():
    prompts1 = [llama3_formal_infer(test_ner_cls) if i % 2 == 0 else llama3_formal_infer(test_sa_cls) for i in range(128)]
    prompts2 = [mistral_formal_infer(test_finqa) for _ in range(128)]
    
    ckpt_folder = "../ckpts"
    base_model = "Mistral-7B-Instruct-v0.3"
    lora_path = "../results/SusGen_GPT_Mistral_v0.3_30k_3epoch_base_lora/"
    vllm_lora_path = False
    vllm_lora_path = "../results/SusGen_GPT_Mistral_v0.3_30k_3epoch_base_lora-vllm/"
    convert_vllm_lora(lora_path, vllm_lora_path)

    args = {
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.2,
    }
    model_path = os.path.join(ckpt_folder, base_model)
    
    llm = load_model(model_path, vllm_lora_path)
    
    out_list1 = inference(llm, prompts1, args, verbose=False, lora_path=vllm_lora_path)
    print(out_list1)

    out_list2 = inference(llm, prompts2, args, verbose=False, lora_path=vllm_lora_path)
    print(out_list2)

if __name__ == "__main__":
    main()
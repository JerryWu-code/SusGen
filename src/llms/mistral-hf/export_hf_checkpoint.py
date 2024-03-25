import torch
from finetune import print_tranable_params
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = "../../../ckpts/Mistral-7B-Instruct-v0.2-hf"
tokenizer = AutoTokenizer.from_pretrained(base)

base_model = AutoModelForCausalLM.from_pretrained(
    base,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print_tranable_params(base_model)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "results/Mistral-7B_alpaca-lora/ckpts/final",
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()
lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

# save the model
base_model.save_pretrained(
    "../../../ckpts/Mistral-7B-alpaca", state_dict=deloreanized_sd
)
print_tranable_params(base_model)

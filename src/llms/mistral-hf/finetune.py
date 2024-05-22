# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to finetune the mistral-7B with lora

#############################################################################
# Package for fine-tuning the Mistral-7B model with Lora
import torch, json, wandb, warnings, transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, TrainingArguments, 
    get_linear_schedule_with_warmup)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig)
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
# Package for accelerating the fine-tuning process
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, 
    FullStateDictConfig)
# Filter out the warnings
warnings.filterwarnings("ignore")

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Set up the wandb login
def login_wandb(project_name=""):
    wandb.login()
    wandb_project = project_name
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

# Set up accelerator
def accelerator_setup():
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    return accelerator

def data_2_prompt_short(record):
    if not record["input"]:
        text = "[INST] ### Instruction: {0}\n\n [/INST]### Response: {1}".format(
            record["instruction"], record["output"])
    else:
        text = "[INST] ### Instruction: {0}\n\n### Input: {1}\n\n [/INST]### Response: {2}".format(
            record["instruction"], record["input"], record["output"])
    return text

def data_2_prompt_formal(record):
    if not record["input"]:
        text = (
            "[INST] "
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n [/INST]### Response:\n{1}").format(
            record["instruction"], record["output"])
    else:
        text = (
            "[INST] "
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n### Input:\n{1}\n\n [/INST]### Response:\n{2}").format(
            record["instruction"], record["input"], record["output"])
    return text

# Set up model
def setup_model(hparams):
    if not hparams["quantization"]:
        model = AutoModelForCausalLM.from_pretrained(hparams["model_path"])
    elif hparams["quantization"] == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(hparams["model_path"], torch_dtype=torch.bfloat16)
    elif hparams["quantization"] == 'int4':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
                hparams["model_path"],
                quantization_config=bnb_config, # int4 quantization
        )

    elif hparams["quantization"] == 'int8':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hparams["model_path"],
            quantization_config=bnb_config, # int8 quantization
        )

    if hparams["lora_path"]:
        peft_config = PeftConfig.from_pretrained(hparams["lora_path"])
        base_with_adapters_model = PeftModel.from_pretrained(model, hparams["lora_path"])
        model = base_with_adapters_model.merge_and_unload()

    if hparams["lora"]:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16, # low rank, goes as the dataset
            lora_alpha=32, # delta_W scaled by alpha/r, set the rate to 2 or 4
            lora_dropout=0.1,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
        )
        # Used for gradient_check and preparing the model for kbit training (save memory)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, peft_config)
        # Get the PEFT model
        model = get_peft_model(model, peft_config)

    return model

# set up the tokenizer
def setup_tokenizer(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams["tokenizer_path"], **hparams)
    return tokenizer

def tokenize(tokenizer, hparams, prompt="This is a test sentence."):
    result = tokenizer(
        text=prompt,
        padding=hparams["padding"],
        truncation=hparams["truncation"],
        max_length=hparams["max_length"],
    )
    return result["input_ids"]

def print_tranable_params(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(name)
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {all_params}")
    print("Trainable parameters ratio: {:.2f}%".format(100 * trainable_params/all_params))

def get_tokenized_prompt(data_2_prompt, tokenizer, hparams):
    def wrapper(record):
        input_text = data_2_prompt(record)
        output_text = input_text.split("### Response:\n")[1]
        input_prompt = input_text.split("### Response:\n")[0] + "### Response:\n"
        if not hparams["max_length"]:
            input_ids = tokenizer(input_text, return_tensors="pt", **hparams)["input_ids"]
            prompt_ids = tokenizer(input_prompt, return_tensors="pt", **hparams)["input_ids"]
        else:
            input_ids = tokenizer(input_text, padding="max_length", return_tensors="pt",
                truncation=hparams["truncation"], max_length=hparams["max_length"])["input_ids"]
            prompt_ids = tokenizer(input_prompt, padding="max_length", return_tensors="pt",
                truncation=hparams["truncation"], max_length=hparams["max_length"])["input_ids"]
        
        # input_ids & labels are same when self-supervised learning, here use supervised learning trick
        input_ids = torch.cat([input_ids, torch.tensor([[2]])], dim=1)
        labels = input_ids.clone()
        labels[:, :len(prompt_ids[0])] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    return wrapper

def plot_data_lengths(tokenized_train_dataset=None, tokenized_val_dataset=None, save_name="alpaca.png"):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    if tokenized_val_dataset:
        lengths += [len(x['input_ids']) for x in tokenized_val_dataset]
    print(len(lengths))

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue', edgecolor='black', linewidth=1.2)
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.savefig(save_name)

def split_data(data, split_ratio=0.1):
    data_split = data.train_test_split(test_size=split_ratio)
    return data_split["train"], data_split["test"]

def main():
    # Set up the accelerator
    accelerator = accelerator_setup()
    # Set up the configuration
    torch.manual_seed(2024)

    hparams = {
        # start config the model
        "lora": True,
        "quantization": 'int4', # 'int4' or 'int8' or 'bf16' or None
        "accelerator": True,
        "model_path": "../../../ckpts/Mistral-7B-v0.2-hf",
        # "lora_path": "results/Mistral-7B_alpaca-lora/ckpts/final", # set to None if not using pre-adapter
        "lora_path": None,
        # start config the tokenizer
        "tokenizer_path": "../../../ckpts/Mistral-7B-v0.2-hf",
        "use_fast": True,
        "padding_side": "left", # set to left use less memory
        "truncation_side": "right", 
        # close to check distribution of sequence length to set the max_length  
        "add_eos_token": True,
        "add_bos_token": True,
        "pad_token": "</s>",
        "padding": True,
        "truncation": True,
        "max_length": 512, # set to None to use the max length of the dataset
    }

    # Set up the wandb login
    base_model = "Mistral-7B-Instruct"
    # base_model = "LLaMA3-Instruct"
    # project = "susgenv1-lora"
    project = "susgen30k-int4-adamw32"
    project_name = f"{base_model}_{project}"
    # login_wandb(project_name=project_name)

    # Set up the model and the tokenizer
    model = setup_model(hparams)
    model.config.window = 256 # set the window size for the PEFT model
    model = accelerator.prepare_model(model)   # require more memory to accelerate
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
        print("Model is parallelizable")
    tokenizer = setup_tokenizer(hparams)
    # print(model.config, '\n')
    # print(model)
    model.print_trainable_parameters()

    # load the dataset
    # data = load_dataset("json", data_files="../../../data/susgen/alpaca/alpaca_data_gpt4.json", split="train")
    data = load_dataset("json", data_files="/home/whatx/SusGen/data/susgen/FINAL/PER_3500/FINAL_PER3500_30k.json", split="train")
    # data = load_dataset("json", data_files="../../../data/susgen/mid_term_version/susgen_6k.json", split="train")
    train_data, val_data = split_data(data, split_ratio=0.005)
    # print(tokenize(tokenizer, hparams, alpaca[0]["instruction"])) # test the tokenizer

    data_2_prompt = data_2_prompt_formal
    tokenized_train_data = train_data.map(get_tokenized_prompt(data_2_prompt, tokenizer, hparams))
    tokenized_val_data = val_data.map(get_tokenized_prompt(data_2_prompt, tokenizer, hparams))
    print(tokenized_train_data)
    # plot_data_lengths(tokenized_train_alpaca, tokenized_val_dataset=tokenized_val_alpaca,
    #                   save_name="figs/alpaca_gpt4.png")

    # Set up the training arguments
    output_dir = os.path.join("results", project_name, "ckpts")
    logging_dir = os.path.join("results", project_name, "logs")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data,
        args=TrainingArguments(
            output_dir=output_dir,
            deepspeed="./configs/ds_configs/ds_config_stage_2.json", 
            # Use deepspeed for training acceleration(if not could comment out)
            warmup_steps=500,               # Number of steps for the warmup phase
            # max_steps=7000,               # Total number of training steps
            num_train_epochs=5,             # Number of epochs to train the model
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,  # Accumulate gradients before backpropagation
            learning_rate=5e-5,             # Want a small lr for finetuning
            lr_scheduler_type="cosine",     # Scheduler with warmup, or use "linear"
            bf16=True,                      # Use bfloat16 for training
            optim="paged_adamw_32bit",      # adamw_apex_fused, adamw, paged_adamw_8/32bit
            logging_steps=10,               # Log every ... step
            logging_dir=logging_dir,        # Directory for storing logs
            save_strategy="epoch",          # Save the model checkpoint "steps", or "epoch"
            # save_steps=500,                  # Save checkpoints every ... steps
            evaluation_strategy="steps",    # Evaluate the model every logging step
            eval_steps=250,                  # Evaluate and save checkpoints every ... steps
            do_eval=True,                   # Perform evaluation at the end of training
            report_to="wandb",              # Comment this out if you don't want to use weights & baises
            run_name=f"{project_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"   
                                            # Name of the W&B run (optional)
        ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False          # silence the warnings. Re-enable for inference!

    trainer.train()
    # trainer.train(resume_from_checkpoint=path_to_checkpoint)
    trainer.save_model(output_dir)

def test():
    # data_ = load_dataset("json", data_files="../../../data/susgen/mid_term_version/susgen_6k.json", split="train")
    # data_.train_test_split(test_size=0.1)
    hparams = {
        # start config the model
        "lora": True,
        "quantization": 'int4', # 'int4' or 'int8' or 'bf16' or None
        "accelerator": True,
        "model_path": "../../../ckpts/Mistral-7B-v0.2-hf",
        # "lora_path": "results/Mistral-7B_alpaca-lora/ckpts/final", # set to None if not using pre-adapter
        "lora_path": None,
        # start config the tokenizer
        "tokenizer_path": "../../../ckpts/Mistral-7B-v0.2-hf",
        "use_fast": False,
        "padding_side": "left", # set to left use less memory
        "truncation_side": "right", 
        # close to check distribution of sequence length to set the max_length  
        # "add_eos_token": True,
        "add_bos_token": True,
        "pad_token": "</s>",
        "padding": True,
        "truncation": True,
        "max_length": 512, # set to None to use the max length of the dataset
    }
    tokenizer = setup_tokenizer(hparams)
    print(tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, 
        tokenizer.unk_token, tokenizer.sep_token, tokenizer.mask_token)
    print(tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id,
        tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id)
    # id to token
    print(tokenizer.decode([tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id,
        tokenizer.unk_token_id]))
    # print the whole tokenizer dict
    print([{i:v} for i,v in tokenizer.get_vocab().items() if v<10])
    # test the tokenizer
    prompt = "This is a test sentence."
    print(tokenizer(prompt, return_tensors="pt", max_length=22, truncation=True, 
        padding="max_length")["input_ids"])
    print(torch.cat([tokenizer(prompt, prompt, return_tensors="pt")["input_ids"], torch.tensor([[2]])], dim=1))
    print(len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]))

if __name__ == "__main__":
    # test()
    main()
    # CUDA_VISIBLE_DEVICES=0,1 python finetune.py
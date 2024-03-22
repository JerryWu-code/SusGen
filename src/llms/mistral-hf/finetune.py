# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to finetune the mistral-7B with lora

#############################################################################
# Package for fine-tuning the Mistral-7B model with Lora
import torch, os, json, wandb, warnings, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
def login_wandb(project_name="mistral-7B_alpaca-lora"):
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

def data_2_prompt(record):
    if not record["input"]:
        text = "[INST] ### Instruction: {0}\n\n [/INST]### Response: {1}".format(
            record["instruction"], record["output"])
    else:
        text = "[INST] ### Instruction: {0}\n\n### Input: {1}\n\n [/INST]### Response: {2}".format(
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

    if hparams["lora"]:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8, # low rank, goes as the dataset
            lora_alpha=16, # delta_W scaled by alpha/r, set the rate to 2 or 4
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
    tokenizer = AutoTokenizer.from_pretrained(
        hparams["tokenizer_path"], 
        use_fast=hparams["use_fast"],
        padding_side=hparams["padding_side"], # set to left use less memory
        truncation_side=hparams["truncation_side"], # close to check distribution of sequence length to set the max_length
        add_eos_token=hparams["add_eos_token"],
        add_bos_token=hparams["add_bos_token"],
        pad_token=hparams["pad_token"], # or use the comment below
    )
    # tokenizer.pad_token = tokenizer.eos_token # "</s>"
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

def get_tokenized_prompt(tokenizer, hparams):
    def wrapper(record):
        if not hparams["max_length"]:
            result = tokenizer(
                text=data_2_prompt(record), 
                padding=hparams["padding"],
                truncation=hparams["truncation"])
        else:
            result = tokenizer(
                text=data_2_prompt(record),
                padding=hparams["padding"],
                truncation=hparams["truncation"],
                max_length=hparams["max_length"],
            )
        result["labels"] = result["input_ids"].copy()
        return result
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
        "model_path": "../../../ckpts/Mistral-7B-Instruct-v0.2-hf",

        # start config the tokenizer
        "tokenizer_path": "../../../ckpts/Mistral-7B-Instruct-v0.2-hf",
        "use_fast": False,
        "padding_side": "left", # set to left use less memory
        "truncation_side": "right", 
        # close to check distribution of sequence length to set the max_length  
        "add_eos_token": True,
        "add_bos_token": True,
        "pad_token": "</s>",
        "padding": True,
        "truncation": True,
        "max_length": 1024, # set to None to use the max length of the dataset
    }

    # Set up the wandb login
    base_model = "Mistral-7B"
    project = "alpaca-lora"
    project_name = f"{base_model}_{project}"
    # login_wandb(project_name=project_name)

    # Set up the model and the tokenizer
    model = setup_model(hparams)
    model = accelerator.prepare_model(model)   # require more memory to accelerate
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
    tokenizer = setup_tokenizer(hparams)
    # print(model.config, '\n')
    # print(model)
    print_tranable_params(model) # model.print_trainable_parameters()

    # load the dataset
    # alpaca = load_json("../../../data/susgen/alpaca/alpaca_data.json")
    alpaca = load_dataset("json", data_files="../../../data/susgen/alpaca/alpaca_data_gpt4.json", split="train")
    train_alpaca, val_alpaca = split_data(alpaca, split_ratio=0.1)
    # print(alpaca[0])
    # print(tokenize(tokenizer, hparams, alpaca[0]["instruction"])) # test the tokenizer

    tokenized_train_alpaca = train_alpaca.map(get_tokenized_prompt(tokenizer, hparams))
    tokenized_val_alpaca = val_alpaca.map(get_tokenized_prompt(tokenizer, hparams))
    print(tokenized_train_alpaca)
    # plot_data_lengths(tokenized_train_alpaca, tokenized_val_dataset=tokenized_val_alpaca,
    #                   save_name="figs/alpaca_gpt4.png")

    # Set up the training arguments
    output_dir = os.path.join("results", project_name, "ckpts")
    logging_dir = os.path.join("results", project_name, "logs")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_alpaca,
        eval_dataset=tokenized_val_alpaca,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=50,                # Number of steps for the warmup phase
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=200,
            learning_rate=2.5e-5,           # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,               # When to start reporting loss
            logging_dir=logging_dir,        # Directory for storing logs
            save_strategy="steps",          # Save the model checkpoint every logging step
            save_steps=25,                  # Save checkpoints every 25 steps
            evaluation_strategy="steps",    # Evaluate the model every logging step
            eval_steps=25,                  # Evaluate and save checkpoints every 25 steps
            do_eval=True,                   # Perform evaluation at the end of training
            report_to="wandb",              # Comment this out if you don't want to use weights & baises
            run_name=f"{project_name}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"   
                                            # Name of the W&B run (optional)
        ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False          # silence the warnings. Re-enable for inference!
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
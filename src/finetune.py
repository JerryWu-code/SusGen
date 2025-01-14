# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this to finetune the 7B Mistral & LLaMA3 with lora

#############################################################################
import torch, json, wandb, warnings, transformers, os, yaml
from torch import multiprocessing as mp
from torch import bfloat16, float16, float32
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    BitsAndBytesConfig, TrainingArguments, 
    get_linear_schedule_with_warmup,
    Trainer, DataCollatorForLanguageModeling)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig)
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator, DistributedType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig, FullStateDictConfig)
warnings.filterwarnings("ignore")
from prompt_template import mistral_formal,llama3_formal
from argparse import ArgumentParser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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

# Set up model
def setup_model(hparams):
    if hparams["acceleration"]:
        accelerator = accelerator_setup()

    if not hparams["quantization"]:
        model = AutoModelForCausalLM.from_pretrained(hparams["model_path"])
    elif hparams["quantization"] == 'bf16':
        model = AutoModelForCausalLM.from_pretrained(hparams["model_path"], torch_dtype=torch.bfloat16)
    else: 
        if hparams["quantization"] == 'int4':
            bnb_config = BitsAndBytesConfig(**hparams["int4_config"])
        if hparams["quantization"] == 'int8':
            bnb_config = BitsAndBytesConfig(**hparams["int8_config"])
        model = AutoModelForCausalLM.from_pretrained(hparams["model_path"], quantization_config=bnb_config)

    if hparams["lora_path"]:
        peft_config = PeftConfig.from_pretrained(hparams["lora_path"])
        base_with_adapters_model = PeftModel.from_pretrained(model, ["lora_path"])
        model = base_with_adapters_model.merge_and_unload()

    if hparams["lora"]:
        peft_config = LoraConfig(**hparams["lora"])
        # Used for gradient_check and preparing the model for kbit training (save memory)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, peft_config)
        # Get the PEFT model
        model = get_peft_model(model, peft_config)

    model.config.window = hparams["window"]
    if hparams["acceleration"]:
        # Set up the accelerator
        model = accelerator.prepare_model(model)
    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True
        print("Model is parallelizable")
    if hparams["show_config"]:
        print(model.config, '\n')
    model.print_trainable_parameters()

    return model

def tokenize(tokenizer, hparams, prompt="This is a test sentence."):
    result = tokenizer(text=prompt, **hparams)
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
        # output_text = input_text.split("### Response:\n")[1]
        # input_prompt = input_text.split("### Response:\n")[0] + "### Response:\n"

        input_ids = tokenizer(input_text, **hparams)["input_ids"]
        # prompt_ids = tokenizer(input_prompt, **hparams)["input_ids"]
        
        # input_ids & labels are same when self-supervised learning, here use supervised learning trick
        # input_ids = torch.cat([input_ids, torch.tensor([[tokenizer.eos_token_id]])], dim=1)
        labels = input_ids.clone()
        # print(prompt_ids.shape, input_ids.shape, labels.shape)
        # print(len(prompt_ids[0]), len(input_ids[0]), len(labels[0]))
        # labels[:, :len(prompt_ids[0])] = -100
        return {
            "input_ids": input_ids.flatten().tolist(),
            "labels": labels.flatten().tolist(),
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

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        temp = {
            "instruction": example['instruction'][i],
            "input": example['input'][i],
            "output": example['output'][i]
        }
        output_texts.append(prompt(temp))
    return output_texts

def main(config):
    # Set up the configuration
    torch.manual_seed(config["model"]["seed"])
    login_wandb(config["name"])

    # # Set up the model and the tokenizer
    model = setup_model(config["model"]) #.to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(**config["tokenizer"])
    tokenizer.pad_token = tokenizer.eos_token

    ### load the dataset & get the prompt template
    data = load_dataset("json", data_files=config["data"]["train"], split="train")
    if config["data"]["val"]:
        train_data = data
        val_data = load_dataset("json", data_files=config["data"]["val"], split="train")
    else:
        train_data, val_data = split_data(data, split_ratio=config["data"]["val_split_ratio"])
    prompt = eval(config["data"]["prompt"])

    # Set up the training arguments and save the configuration
    output_dir = os.path.join(config["output_dir"], config["name"])
    logging_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        local_rank=config["local_rank"],
        run_name=f"{config['name']}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        **config["training"]
    )
    training_args.distributed_type = DistributedType.DEEPSPEED

    if config["trainer"] == "Trainer":
        tokenized_train_data = train_data.map(get_tokenized_prompt(prompt, tokenizer, config["tokenizer"]["encode"]))
        #, batched=True, batch_size=512, num_proc=20).shuffle()
        tokenized_val_data = val_data.map(get_tokenized_prompt(prompt, tokenizer, config["tokenizer"]["encode"]))
        trainer = Trainer(
            model=model,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_val_data,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
    elif config["trainer"] == "SFTTrainer":
        trainer = SFTTrainer(
            model=model,
            train_dataset=data,
            formatting_func=formatting_prompts_func,
            args=training_args,
            data_collator=DataCollatorForCompletionOnlyLM(
                response_template="### Response:", tokenizer=tokenizer),
        )

    model.config.use_cache = False # silence the warnings. Re-enable for inference!

    if config["training"]["resume_from_checkpoint"]:
        trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    else:
        trainer.train()
    trainer.save_model(output_dir)

def test_tokenizer(config):
    ### set up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(**config["tokenizer"])
    tokenizer.pad_token = tokenizer.eos_token

    ### find out the special tokens
    print(tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, 
        tokenizer.unk_token, tokenizer.sep_token, tokenizer.mask_token)
    print(tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id,
        tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id)
    
    ### check id to token mapping
    print(tokenizer.decode([29473, tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]))
    print([{i:v} for i,v in tokenizer.get_vocab().items() if v<10])

    ### check the tokenizer encoding & decoding for easy prompt
    prompt_template = eval(config["data"]["prompt"])
    easy_prompt = {
        "instruction": "This is a test sentence.",
        "input": "",
        "output": "Sure."
    }
    easy_prompt = prompt_template(easy_prompt)
    print(tokenizer.decode(tokenizer(easy_prompt)["input_ids"]))
    print(torch.cat([tokenizer(easy_prompt, return_tensors="pt")["input_ids"], torch.tensor([[tokenizer.eos_token_id]])], dim=1).flatten().tolist())
    print(len(tokenizer(easy_prompt, return_tensors="pt", max_length=50, truncation=True, padding=True)["input_ids"][0]))

    ### check the tokenizer encoding & decoding for dataset, and plot the data lengths
    train_data = load_dataset("json", data_files=config["data"]["train"], split="train")
    train_data_head = train_data.select([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    tok_train_data_head = train_data_head.map(get_tokenized_prompt(prompt_template, tokenizer, config["tokenizer"]["encode"]))
    print(train_data_head[1])
    print(tokenize(tokenizer, config["tokenizer"]["encode"], train_data_head[1]["instruction"]))
    input_ids = tokenizer(train_data_head[1]["instruction"], **config["tokenizer"]["encode"])["input_ids"]
    print("="*20, "\n", input_ids)
    # plot_data_lengths(tok_train_data, tokenized_val_data, "figs/alpaca_gpt4.png")

if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = ArgumentParser(description="Fine-tune LLM with Lora using a config file")
    parser.add_argument("--config", type=str, help="Path to the config file", default="configs/finetune_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    local_rank = config["local_rank"]
    torch.cuda.set_device(local_rank)

    test_tokenizer(config)
    # main(config)
    # CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 finetune.py
    # CUDA_VISIBLE_DEVICES=3 /home/.../anaconda3/envs/llm/bin/torchrun --nproc_per_node=1 --master_port=29501 finetune.py

    # python -m torch.distributed.launch --nproc_per_node 1 finetune.py --config=configs/finetune_config.yaml
    # master_port=$(shuf -n 1 -i 10000-65535) deepspeed --include localhost:0,1 --master_port "${master_port}" finetune.py --config=configs/finetune_config.yaml
### Training Prompts
def mistral_short(record):
    if not record["input"]:
        text = "[INST]### Instruction: {0}\n\n[/INST]### Response: {1}</s>".format(
            record["instruction"], record["output"])
    else:
        text = "[INST]### Instruction: {0}\n\n### Input: {1}\n\n[/INST]### Response: {2}</s>".format(
            record["instruction"], record["input"], record["output"])
    return text

def mistral_formal(record):
    if not record["input"]:
        text = (
            "[INST]Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n[/INST]### Response:\n{1}</s>").format(
            record["instruction"], record["output"])
    else:
        text = (
            "[INST]Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n### Input:\n{1}\n\n[/INST]### Response:\n{2}</s>").format(
            record["instruction"], record["input"], record["output"])
    return text

def llama3_short(record):
    if not record["input"]:
        text = "### Instruction: {0}\n\n### Response: {1}</s>".format(
            record["instruction"], record["output"])
    else:
        text = "### Instruction: {0}\n\n### Input: {1}\n\n### Response: {2}</s>".format(
            record["instruction"], record["input"], record["output"])
    return text

def llama3_formal(record):
    if not record["input"]:
        text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n### Response:\n{1}<|end_of_text|>").format(
            record["instruction"], record["output"])
    else:
        text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n### Input:\n{1}\n\n### Response:\n{2}<|end_of_text|>").format(
            record["instruction"], record["input"], record["output"])
    return text

### Inference Prompts
def mistral_short_infer(record):
    if not record["input"]:
        text = (
            "[INST]### Instruction:\n{0}\n\n[/INST]").format(record["instruction"])
    else:
        text = ("[INST]### Instruction:\n{0}\n\n### Input:\n{1}\n\n[/INST]### Response:\n").format(
            record["instruction"], record["input"])
    return text

def mistral_formal_infer(record):
    if not record["input"]:
        text = (
            "[INST]"
            "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
            "### Instruction:\n{0}\n\n[/INST]### Response:\n").format(record["instruction"])
    else:
        text = (
            "[INST]"
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n### Input:\n{1}\n\n[/INST]### Response:\n").format(
            record["instruction"], record["input"])
    return text

def llama3_short_infer(record):
    if not record["input"]:
        text = (
            "### Instruction:\n{0}\n\n").format(record["instruction"])
    else:
        text = ("### Instruction:\n{0}\n\n### Input:\n{1}\n\n### Response:\n").format(
            record["instruction"], record["input"])
    return text

def llama3_formal_infer(record):
    if not record["input"]:
        text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
            "### Instruction:\n{0}\n\n### Response:\n").format(record["instruction"])
    else:
        text = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{0}\n\n### Input:\n{1}\n\n### Response:\n").format(
            record["instruction"], record["input"])
    return text
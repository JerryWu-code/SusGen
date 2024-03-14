# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to load financial instruction tuning dataset from huggingface.

#############################################################################
import warnings
warnings.filterwarnings("ignore")

import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 4096,
    max_batch_size: int = 4,
):
    """
    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Max to 4096.
        max_gen_len (int, optional): The maximum length of generated sequences. Max to 4096.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 6.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts_ = [
        "The theory of relativity states that",
        "Translate English to French: sea otter =>",
    ]

    prompts: List[str] = [
        ##### Type 1
        # For these prompts, the expected answer is the natural continuation of the prompt
        "Simply put, the theory of relativity states that ",

        ##### Type 2
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",

        ##### Type 3 Instruction based prompt ~ current bad performance need to be tuned
        "{user_instruction} \n Question:{question}".format(
            user_instruction=(
                "You are a senior equity analyst with expertise in climate science"
                "evaluating a company 's sustainability report, "
                "you will answer the question in detail."
            ),
            question="What is the company's carbon footprint?",
        ),
        (
            "<s> [INST] {text} [/INST]\n".format(
                text = "Compose a welcome email within 200 words for new customers who have just made their first purchase with your product."
            "Start by expressing your gratitude for their business, and then convey your excitement for having them as a customer."
            "Include relevant details about their recent order. Sign the email with \"The Fun Shop Team\".\n"

            "Order details:\n"
            "- Customer name: Anna\n"
            "- Product: hat\n"
            "- Estimate date of delivery: Feb. 25, 2024\n"
            "- Return policy: 30 days"
            )
        )
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        # print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)

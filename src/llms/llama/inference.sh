# !/bin/bash
# Author: "WHATX" -- Wu Qilong
# Institute: National University of Singapore, A Star IHPC
# Description: Use this script to run inference on a single node with 1 GPU.

#############################################################################

torchrun --nproc_per_node 1 test.py \
    --ckpt_dir ../../../ckpts/llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 256 \
    --max_batch_size 8
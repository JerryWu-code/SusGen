torchrun --nproc_per_node 1 test.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 2048 --max_batch_size 8
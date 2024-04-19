torchrun example_chat_completion.py \
    --ckpt_dir ../../../ckpts/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path ../../../ckpts/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 256 \
    --max_batch_size 6
import gradio as gr
import os, torch
from utils.prompt_template import *
from template import *
from transformers import TextStreamer

ckpt_folder = "../ckpts"
base_model = "Mistral-7B-Instruct-v0.3-hf"
model, tokenizer, device, config = load_model(
    model_path=os.path.join(ckpt_folder, base_model),
    lora_path="../results/SusGen30k-int4-adamw32_Mistral-7B-v0.3/checkpoint-1406")
    # quantization='int4')

def ask(instr, input):
    temp = {
        "instruction": instr,
        "input": input,
    }
    prompt = mistral_formal_infer(temp)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    input_length = inputs.input_ids.shape[1]
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, 
                            return_dict_in_generate=True, streamer=TextStreamer(tokenizer))
    
    tokens = outputs.sequences[0, input_length:]
    return tokenizer.decode(tokens, skip_special_tokens=True) 
 
with gr.Blocks() as server:
    with gr.Tab("LLM Inferencing"):
 
        model_instr = gr.Textbox(label="Your Instruction:", 
                                value="What’s your Instruction?", interactive=True)
        model_input = gr.Textbox(label="Your Input:", 
                                value="What’s your Input?", interactive=True)
        ask_button = gr.Button("Ask")
        model_output = gr.Textbox(label="The Answer:", interactive=False, 
                                value="Answer goes here...")
 
    ask_button.click(ask, inputs=[model_instr, model_input], outputs=[model_output])

server.launch()
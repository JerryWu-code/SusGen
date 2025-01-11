# Author: Wu Qilong
# Institute: National University of Singapore
# Description: Use this script to do inference combining with RAG.

#############################################################################
import logging
import sys
import torch
from typing import List, Union, Tuple
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import Settings

torch.set_default_device('cuda')
# ERROR to show only errors, INFO to show all logs
logging.basicConfig(stream=sys.stdout, level=logging.INFO) 
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# Set the log level for the sentence_transformers package
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

def load_llm_embed(llm_args: dict, embed_path: str) -> Tuple[HuggingFaceLLM, HuggingFaceEmbeddings]:
    llm = HuggingFaceLLM(**llm_args)
    embed_model = HuggingFaceEmbeddings(model_name=embed_path)
    return llm, embed_model

def load_documents(file_paths: Union[str, List[str]]) -> List:
    if isinstance(file_paths, str):
        documents = SimpleDirectoryReader(input_dir=file_paths).load_data()
    elif isinstance(file_paths, list):
        documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    else:
        raise ValueError("Invalid input. Please provide a string or list of strings.")
    
    return documents

def rag_qa(index: VectorStoreIndex, query_str: str) -> str:
    query_engine = index.as_query_engine()
    response = query_engine.query(query_str)
    return response

def main():
    system_prompt = (
    "You are a Q&A assistant in financial domain. "
    "Your goal is to answer questions as accurately as possible "
    "based on the instructions and context provided."
    )
    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = "<|USER|>{query_str}<|ASSISTANT|>"
    mistral_v2 = "../ckpts/Mistral-7B-Instruct-v0.2-hf"
    llama3 = "../ckpts/Meta-Llama-3-8B-Instruct-hf"
    llm_path = llama3
    embed_model_path = "../ckpts/all-mpnet-base-v2"
    # candidates documents ~ Folder path or file list: 
    docu_files = ["../data/raw_data/raw_pdf/3M_2020_ESG.pdf"]
    docu_folder = "RAG/data"
    documents_path = docu_files
    text_chunking = 1024

    llm_args = {
        "system_prompt": system_prompt,
        "query_wrapper_prompt": query_wrapper_prompt,
        "device_map": "auto",
        "context_window": 5120,
        "max_new_tokens": 4096,
        "generate_kwargs": {"temperature": 0.1, "do_sample": True},
        "tokenizer_kwargs": {"max_length": 4096},
        "model_kwargs": {"torch_dtype": torch.float16},
        "model_name": llm_path,
        "tokenizer_name": llm_path,
    }

    # Load the LLM and Embedding model
    llm, embed_model = load_llm_embed(llm_args, embed_model_path)
    # Load the documents
    documents = load_documents(documents_path)

    # Setting
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = text_chunking
    index = VectorStoreIndex.from_documents(documents, settings=Settings)

    # RAG_QA
    # query_str = "What is the revenue of the company in 2020?"
    query_str = "Please summarize the company's information in detail"# and describe information related to Governance, Strategy, Risk Management and Metrics and Targets."
    response = rag_qa(index, query_str)
    print(response)

if __name__ == "__main__":
    main()
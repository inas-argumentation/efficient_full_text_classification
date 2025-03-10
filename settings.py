import os

model_type = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
LLM_checkpoint = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda"

data_dir = lambda x: os.path.join(os.path.dirname(__file__), f"data/{x}")
from huggingface_hub import login
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
import os

os.environ["HF_TOKEN"] = "XXX"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
login(os.environ["HF_TOKEN"])

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
# hf = HuggingFaceBgeEmbeddings(
#     model_name = model_name,
#     model_kwargs = model_kwargs,
#     encode_kwargs = encode_kwargs
# )

hf = HuggingFaceEmbeddings(
    model_name = "meta-llama/Llama-3.2-1B",
    model_kwargs = model_kwargs,
    encode_kwargs = {"normalize_embeddings": True}
)

#Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` 
#or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})

hf.client.tokenizer.pad_token = hf.client.tokenizer.eos_token

embeddings = hf.embed_query("hi this is akash")
print(len(embeddings))
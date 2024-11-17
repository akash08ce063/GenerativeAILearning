import os
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["HF_TOKEN"] = "XXX"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
login(os.environ["HF_TOKEN"])

template = """
You are expert in medical field.

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

transformer_pipeline = pipeline(
        "text-generation",
        model=model,
        return_full_text=False,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.1)

llm = HuggingFacePipeline(pipeline=transformer_pipeline)

chain = prompt_template | llm
while(1):
    query = input("\n> Please, type your next query: ")
    answer = chain.invoke({"query" : query})
    print(f">>> Answer: {answer}")
    print("******************************************")            
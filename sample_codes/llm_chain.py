import os
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = "XXX"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=64, model_kwargs={"temperature":0})

hf = HuggingFacePipeline(pipeline=pipe)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's request only based on the given context."),
    ("user", "Question: {question} \n Context: {context}\n")
])

output_parser = StrOutputParser()

chain = prompt | hf.bind(skipPrompt= True) | output_parser

question = "Can you summarize this morning's meetings for me?"
context = "During this morning's meeting, we solved all world conflict."
results = chain.invoke({"question": question, "context": context})
results
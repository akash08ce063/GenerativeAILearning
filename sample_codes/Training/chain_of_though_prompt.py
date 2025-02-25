from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

input_question =  """Q: a juggler can juggle 16 balls. half of the balls are golf balls,
    and half of the golf balls are blue. How many blue golf balls are there?\n
    A: let's think step by step"""

llm = ChatOllama(
    model = "llama3.1",
    temperature = 0,
    num_predict = 256,
)
      

response = llm.invoke(input_question)
print(response)

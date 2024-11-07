import transformers
import torch
import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import StoppingCriteria, StoppingCriteriaList

# Note: `userdata.get` is a Colab API. If you're not using Colab,-Instruct set the env
# vars as appropriate for your system.
os.environ["HF_TOKEN"] = "XXX"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
login(os.environ["HF_TOKEN"])

# Specify template below.

template = """Give very short answer. Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])
print(prompt)


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# define stopping criteria

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]

print(stop_token_ids    )

stop_token_ids = [torch.LongTensor(x) for x in stop_token_ids]
print(stop_token_ids)

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
    
stopping_criteria = StoppingCriteriaList([StopOnTokens()])
pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        repetition_penalty=1.1,
        stopping_criteria= stopping_criteria,
        model_kwargs={"temperature":0})

# example repetitions and stopping criteria.
# generate_text = transformers.pipeline(
#     model=model, 
#     tokenizer=tokenizer,
#     return_full_text=True,  # langchain expects the full text
#     task='text-generation',
#     # we pass model parameters here too
#     stopping_criteria=stopping_criteria,  # without this model rambles during chat
#     temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
#     max_new_tokens=512,  # max number of tokens to generate in the output
#     repetition_penalty=1.1  # without this output begins repeating
# )

llm = HuggingFacePipeline(pipeline=pipe)
output = llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")

# llm_chain = LLMChain(prompt=prompt, llm=hf)
# question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
# #print(llm_chain.run(question))

# rag_chain = ( 
#             llm_chain
#             | StrOutputParser()
#         )

# print(rag_chain.invoke({"question": question}))

# model_id = "meta-llama/Llama-3.2-1B"

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# tokenizer.chat_template = messages

# pipeline_v2 = pipeline(
#     "text-generation",
#     model=model_id,
#     tokenizer=tokenizer,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )


# outputs = pipeline_v2(
#     messages,
#     max_new_tokens=256,
# )

print(output)
# print(outputs[0]["generated_text"][-1])



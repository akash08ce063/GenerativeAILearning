### NOT GOOD EXAMPLE as hugging facepipeline anyway converts AIMESSAGE and
### HUMANMESSAGE to AI: and Human: so better to use frew shot prompt in 
### simple prompting technique.

import os
from huggingface_hub import login
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains import ConversationChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList

# os.environ["HF_TOKEN"] = "XXX"
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
# login(os.environ["HF_TOKEN"])

# create our examples
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "2 + 3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# create a prompt example from above template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

print(few_shot_prompt.invoke({}).to_messages())

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

class OneLineStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_token_ids, max_new_tokens = 50):
        self.stop_token_ids = stop_token_ids
        self.max_new_tokens = max_new_tokens
        self.token_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if any(token_id in input_ids[0][-len(self.stop_token_ids):] for token_id in self.stop_token_ids):
            return True
        self.token_count +=1
        if self.token_count >= self.max_new_tokens:
            return True
        return False       
        
stop_tokens = ["###", "AI:"]
stop_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in stop_tokens]

# Initialize custom stopping criteria
stopping_criteria = StoppingCriteriaList([OneLineStoppingCriteria(stop_token_ids)])


transformer_pipeline = pipeline(
        "text-generation",
        model=model,
        return_full_text=False,
        tokenizer=tokenizer,
        stopping_criteria=stopping_criteria,
        max_new_tokens=200,
        temperature=0.1)

llm = HuggingFacePipeline(pipeline=transformer_pipeline)

chain = final_prompt | llm

while(1):
    query = input("\n> Please, type your next query: ")
    answer = chain.invoke({"input" : query})
    print(f">>> Answer: {answer}")
    print("******************************************")            
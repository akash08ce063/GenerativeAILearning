import os
from huggingface_hub import login
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList

# os.environ["HF_TOKEN"] = "XXX"
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
# login(os.environ["HF_TOKEN"])

# create our examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny one-liner responses to the users questions. Here are some
examples: 
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

print(few_shot_prompt_template.format(query="What is the meaning of life?"))


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
        
stop_tokens = ["###", "\n\n", "User:"]
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

chain = few_shot_prompt_template | llm
while(1):
    query = input("\n> Please, type your next query: ")
    answer = chain.invoke({"query" : query})
    print(f">>> Answer: {answer}")
    print("******************************************")            
# Few shot prompting

Create a few prompts like below as an example and provide context with some example and eventually it will predict the texts, and **one of the most important problem** with the few short prompting is context window length, so your given examples as well as your query should fit in context windows size current llm capacity is 128k tokens 

Reference - https://www.pinecone.io/learn/series/langchain/langchain-prompt-templates/

```from langchain import FewShotPromptTemplate


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
creative  and funny responses to the users questions. Here are some
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

query = "What is the meaning of life?"

print(few_shot_prompt_template.format(query=query))

```




### Stop Criteria

Defining stopping criteria is the most essential thing to define to tell the model where to stop text generation, below is an example of stopping generation where it encounter "User:" 
word otherwise it will keep generating the examples similar to few shot prompting
```
# Define custom stopping criteria
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

```

### Prompt Techniques from LLAMA papers

![image](https://github.com/user-attachments/assets/496645ad-4ac0-4e75-9b05-89970990bd35)

Guidelines from META - https://www.llama.com/docs/how-to-guides/prompting/


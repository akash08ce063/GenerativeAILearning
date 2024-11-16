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
)```

query = "What is the meaning of life?"

print(few_shot_prompt_template.format(query=query))

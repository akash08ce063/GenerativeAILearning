from huggingface_hub import login
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, StoppingCriteria, StoppingCriteriaList

# Example reviews
examples = [
    {
        "Review": "The ambience was great, specificaly the Shivaji graffiti on the wall was spectacular. Foods were yummy too.",
        "Reply": "Thank you so much for your insightful review. We are glad that you noticed the graffiti and appreciated our efforts. Hope you visit us soon."
    }, {
        "Review": "Vada Pav was so delicious, but Pav bhaji was too salty that it is not acceptable.",
        "Reply": "We're ever so sorry to hear that part of your food was salty and you didn't enjoy it. We are taking notes and will let our chef know about it. Please let us know whenever you revisit us, and we will extra take care of your order. As a token of our politeness, we would like to offer you an extra meal."
    }, {
        "Review": "The movie was too lengthy, and hard to follow. There were some unnecessary songs.",
        "Reply": "Sorry to hear that you didn't like the story and making of it. However, for my next movie I will keep in mind about your concerns."
    }, {
        "Review": "The quantity of the food was so less that it is not worth of the price.",
        "Reply": "We are sorry that you feel that way. We will try to address your concern on food proportion to price ratio in our next budget discussion."
    }, {
        "Review": "I am a fan of Benedict cumberbatch. I watched all of his movies and shows.",
        "Reply": "Thank you for your review. We will make sure there are enough Benedict's plays in our theatre."
    }
]

# create a example template
example_template = """
Customer: {Review}
AI: {Reply}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["Review", "Reply"],
    template=example_template
)

# the prefix is our instructions
prefix = """You are an AI assistant and act like you are an owner of multi-purpose public services 
    (e.g., restaurants, movie theatres, superstores etc.). Based on the semantic meaning of the customer reviews, 
    please generate replies. If it is Negative review, reply with politeness, take the accountability and provide a 
    resolution steps of customer's concern. If it is Positive review, reply with funny and witty tone so that they feel engaged 
    and revisit the service. Here are some examples: """


# and the suffix our user input and output indicator
suffix = """
Customer: {Review}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["review"],
    example_separator="\n\n"
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
        
stop_tokens = ["###", "\n", "Customer:"]
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
    review = input("\n> Please, type your review: ")
    reply = chain.invoke({"Review" : review})
    print(f">>> Answer: {reply}")
    print("******************************************")            
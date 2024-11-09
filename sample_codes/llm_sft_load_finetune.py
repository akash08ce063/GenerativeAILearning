import os
import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline, HfArgumentParser, TrainingArguments, logging, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig


os.environ["HF_TOKEN"] = "XXX"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
login(os.environ["HF_TOKEN"])

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

fine_tune_model = PeftModel.from_pretrained(
    base_model,
    "fine_tuned_instruction_model"
)

fine_tune_model = fine_tune_model.merge_and_unload()

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=fine_tune_model, truncation=True, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
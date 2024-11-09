import os
import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import HfArgumentParser, TrainingArguments, logging, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig


os.environ["HF_TOKEN"] = "XXX"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "XXX"
login(os.environ["HF_TOKEN"])

# define model names

base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "mlabonne/guanaco-llama2-1k"

fine_tune_model_name = "fine_tuned_instruction_model"

# Lora config

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

#bits and bytes parameters
# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Training arguments
output_dir = "./model_output"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

dataset = load_dataset(dataset_name, split="train")
# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=use_4bit,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=use_nested_quant,
# )


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)

base_model.use_cache = False
base_model.config.pretraining_tp = 1

# Load llama tokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LORA configuration

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    use_cpu=True
)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=SFTConfig(output_dir = output_dir, packing=False)
)

trainer.train()

trainer.model.save_pretrained(fine_tune_model_name)
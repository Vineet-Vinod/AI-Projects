# DO NOT RUN ON YOUR CPU
# RAN FOR 15 MINUTES ON MY LAPTOP AND DID NOTHING

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURATION ---
MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
LORA_ADAPTER_PATH = OUTPUT_DIR = './DarkQwen_LoRA'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    prompt = input("Enter your prompt (or q to quit)\n> ")
    if prompt == 'q':
        break
    print(generate(prompt))
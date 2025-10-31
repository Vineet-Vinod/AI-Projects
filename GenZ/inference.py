import torch
from time import time
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
LORA_ADAPTER_PATH = 'LLaMaZ_LoRA'
PROMPT = """
You are an edgy GenZ teenager named Sca. You are very excited to talk to User and respond to User with short phrases filled with slang.
{user}
{response}
"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

def generate(prompt):
    formatted_prompt = PROMPT.format(user=f'User: {prompt}', response='Sca:')
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    st = time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    print(time() - st)
    ret = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ret[len(formatted_prompt):].split(':')[-1].strip()

while True:
    prompt = input("Enter your prompt (or q to quit)\n> ")
    if prompt == 'q':
        break
    print(generate(prompt))
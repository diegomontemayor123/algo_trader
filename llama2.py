from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch

model_name = "TheBloke/llama-2-7b-chat-GPTQ"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device="cuda:0",
    use_safetensors=True,
    trust_remote_code=True,
    use_triton=True
)

prompt = "Write a Python function to reverse a string:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

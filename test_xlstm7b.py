import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use proper HuggingFace model ID instead of raw path
model_name = "NX-AI/xLSTM-7b"

# Load the model with Triton kernels which should work on AMD
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Print model info
print(f"Model loaded. Device: {next(model.parameters()).device}")

# Test with a simple prompt
prompt = "In a world where technology and nature coexist,"
print(f"Prompt: {prompt}")

# Tokenize
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# Generate
print("Generating...")
start_time = time.time()

with torch.no_grad():
    # Warmup
    _ = model.generate(input_ids, max_new_tokens=10)
    torch.cuda.synchronize()
    
    # Actual generation with timing
    start_time = time.time()
    output = model.generate(
        input_ids, 
        max_new_tokens=100,
        temperature=0.7
    )
    torch.cuda.synchronize()

end_time = time.time()

# Get generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated text:")
print(generated_text)

# Print timing info
tokens_generated = output.shape[1] - input_ids.shape[1]
time_taken = end_time - start_time
tokens_per_second = tokens_generated / time_taken

print(f"\nGeneration stats:")
print(f"Tokens generated: {tokens_generated}")
print(f"Time taken: {time_taken:.2f} seconds")
print(f"Tokens per second: {tokens_per_second:.2f}") 
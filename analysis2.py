import torch
from optimum.quanto import freeze, qint8, quantize
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-410m"

# The tokenizer turns text into token ids and back
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Causal LM: predicts the next token given the tokens before it
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()



def memory_footprint(model, label):
    # Sum the bytes actually stored for each parameter and buffer
    total_bytes = 0
    for param in model.parameters():
        if hasattr(param, "_data") and hasattr(param, "_scale"):
            # Quanto quantized weight: the int8 values plus their scale factors
            total_bytes += param._data.numel() * param._data.element_size()
            total_bytes += param._scale.numel() * param._scale.element_size()
        else:
            total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()

    print(f"{label} model size: {total_bytes:,} bytes")
    print(f"{label} model size: {total_bytes / 1024**2:.1f} MB")
    return total_bytes


# Print the layer structure and size before quantization
print(model)
memory_footprint(model, "Original")

# Quantize the weights to int8 (qint8) in place, then freeze them so the quantized weights are stored
quantize(model, weights=qint8)
freeze(model)

# Print the quantized architecture and size for comparison with the original
print(model)
memory_footprint(model, "Quantized (int8)")

prompt = "This is Winter"

# Convert the prompt to token ids and PyTorch tensors
inputs = tokenizer(prompt, return_tensors="pt")

# Generate the continuation without tracking gradients
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)

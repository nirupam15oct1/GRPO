# Step-by-Step Guide: Quantizing Vision-Language Models (VLMs) from Hugging Face

Quantizing Vision-Language Models (such as LLaVA, PaliGemma, or Qwen2-VL) reduces GPU VRAM consumption and inference latency by mapping high-precision floating-point weights (FP32/FP16/BF16) into lower-bit integer representations (INT8 or INT4).

---

## 1. Prerequisites and Installation

Install Hugging Face `transformers`, `accelerate`, `bitsandbytes`, and visual dependencies (`pillow`, `torchvision`):

```bash
pip install --upgrade torch transformers accelerate bitsandbytes torchvision pillow
2. Theoretical Overview: Linear (Affine / Uniform) QuantizationLinear quantization maps a continuous range $[r_{min}, r_{max}]$ to a discrete integer grid $[q_{min}, q_{max}]$.Mathematical FormulationGiven scale $S$ and zero-point $Z$:$$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$$$$Z = \text{round}\left( \frac{-r_{min}}{S} \right) + q_{min}$$Quantization:$$q = \text{clip}\left( \text{round}\left(\frac{r}{S}\right) + Z, \; q_{min}, \; q_{max} \right)$$Dequantization (Reconstruction):$$\hat{r} = S \times (q - Z)$$For symmetric quantization, $Z = 0$ and $S = \frac{\max(\vert{}r\vert{})}{q_{max}}$.3. Snippet: Linear Quantization from Scratch (PyTorch)Below is a self-contained PyTorch implementation demonstrating both symmetric and asymmetric affine linear quantization on an arbitrary weight tensor.Pythonimport torch


def linear_quantize(tensor: torch.Tensor, num_bits: int = 8, symmetric: bool = True):
    """Performs uniform affine/linear quantization and dequantization on a tensor.

    Args:
        tensor: Input FP32/FP16 tensor.
        num_bits: Bit-width (e.g., 8 for INT8, 4 for INT4).
        symmetric: If True, uses zero-point = 0.

    Returns:
        quantized_tensor: Integer tensor.
        dequantized_tensor: Reconstructed floating-point tensor.
        scale: Computed scale factor (S).
        zero_point: Computed zero-point offset (Z).
    """
    if symmetric:
        q_min = -(2 ** (num_bits - 1))
        q_max = (2 ** (num_bits - 1)) - 1
        
        # Determine maximum absolute range
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / q_max
        zero_point = 0
    else:
        q_min = 0
        q_max = (2 ** num_bits) - 1
        
        r_min = torch.min(tensor)
        r_max = torch.max(tensor)
        scale = (r_max - r_min) / (q_max - q_min)
        zero_point = torch.round(-r_min / scale).clamp(q_min, q_max).to(torch.int32)

    # Quantize: q = clip(round(r / S) + Z)
    scaled = tensor / (scale + 1e-8)
    quantized_tensor = torch.clamp(torch.round(scaled) + zero_point, q_min, q_max).to(torch.int8)

    # Dequantize: r_approx = (q - Z) * S
    dequantized_tensor = (quantized_tensor.to(torch.float32) - zero_point) * scale

    return quantized_tensor, dequantized_tensor, scale, zero_point


# Verification Example
if __name__ == "__main__":
    weights = torch.randn(4, 4)
    q_weights, deq_weights, s, z = linear_quantize(weights, num_bits=8, symmetric=False)
    
    mae_loss = torch.mean(torch.abs(weights - deq_weights)).item()
    print("Original Weights:\n", weights[:2, :2])
    print("\nQuantized (INT8):\n", q_weights[:2, :2])
    print("\nDequantized Approximation:\n", deq_weights[:2, :2])
    print(f"\nMean Absolute Quantization Error: {mae_loss:.6f}")
4. Step-by-Step VLM Quantization Pipeline in Hugging FaceBecause Vision-Language Models consist of three distinct components:Vision Tower / Encoder (e.g., CLIP-ViT, SigLIP)Multimodal Projector / Connector (Linear projection / MLP)Autoregressive Language Decoder (e.g., Llama, Vicuna, Qwen)Quantizing them on-the-fly via bitsandbytes automatically replaces all linear layers in the transformer layers with INT8/INT4 modules.Step 4.1: Define Quantization ConfigurationUse BitsAndBytesConfig to specify the bit-width and compute dtype.Pythonimport torch
from transformers import BitsAndBytesConfig

# 4-bit NormalFloat (NF4) configuration with double quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",               # Options: "fp4" or "nf4"
    bnb_4bit_use_double_quant=True,          # Quantizes quantization constants (saves extra VRAM)
    bnb_4bit_compute_dtype=torch.bfloat16    # Unquantized compute type during matrix ops
)
(For standard INT8 Linear Quantization, pass BitsAndBytesConfig(load_in_8bit=True) instead).Step 4.2: Load Processor and Quantized ModelLoad the multimodal checkpoint using AutoProcessor and the appropriate model class (such as AutoModelForVision2Seq or architecture-specific classes like LlavaForConditionalGeneration).Pythonfrom transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"

# 1. Load Vision-Language Processor (Tokenizer + Image Feature Extractor)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Load Model in 4-bit Quantized format
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"                        # Automatically balances layers across GPU/CPU
)

print(f"Model footprint on GPU: {model.get_memory_footprint() / 1e9:.2f} GB")
Step 4.3: Run Multimodal InferenceFormat prompt templates with image tokens and run inference through the quantized model:Pythonimport requests
from PIL import Image

# Load sample image
image_url = "[https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png](https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png)"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# Construct chat prompt
prompt = "USER: <image>\nDescribe what you see in this image in detail.\nASSISTANT:"

# Preprocess inputs
inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to("cuda")

# Generate output tokens
with torch.inference_mode():
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )

# Decode response
response = processor.batch_decode(
    generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False
)[0]

print(response.split("ASSISTANT:")[-1].strip())
5. Practical Tips for VLMsSelective Layer Exclusion (llm_int8_skip_modules):Vision encoders often degrade more aggressively under low-bit quantization than text decoders. If visual fidelity drops, preserve vision layers in 16-bit by setting:Pythonbnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["vision_tower", "multi_modal_projector"]
)
Flash Attention 2:Accelerate memory bandwidth and context processing by passing attn_implementation="flash_attention_2" during from_pretrained(...).Saving & Deploying:On-the-fly bitsandbytes models cannot be exported as raw static INT4 weights via .save_pretrained() directly for standalone execution; use tools like AutoAWQ or vLLM if you need fully serialized, standalone INT4/AWQ checkpoints for high-throughput serving engines.

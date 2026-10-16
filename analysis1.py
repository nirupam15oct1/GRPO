import copy
import textwrap
import urllib.request

import torch
from PIL import Image

from load_blip import load_model

processor, model = load_model()

# Keep the float32 model as is, and make an independent bfloat16 copy for later comparison
model_fp32 = model
model_bf16 = copy.deepcopy(model).to(torch.bfloat16)

# get_memory_footprint() returns the total size of the weights and buffers in bytes
for label, m in [("float32", model_fp32), ("bfloat16", model_bf16)]:
    footprint_bytes = m.get_memory_footprint()
    footprint_mb = footprint_bytes / 1024**2
    print(f"Model memory footprint ({label}): {footprint_bytes:,} bytes")
    print(f"Model memory footprint ({label}): {footprint_mb:.1f} MB")

# Five sample images hosted on the Hugging Face Hub
IMAGE_URLS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
    "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png",
]


def load_image(url):
    with urllib.request.urlopen(url) as response:
        return Image.open(response).convert("RGB")


def caption(m, image):
    # Preprocess the image on the same device as the model, then generate and decode the caption
    device = next(m.parameters()).device
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = m.generate(**inputs, max_new_tokens=30)
    return processor.decode(output_ids[0], skip_special_tokens=True)


# Run each image through both models and print the captions side by side
model_fp32.eval()
model_bf16.eval()
column_width = 50
print()
print(f"{'#':<3} {'float32':<{column_width}} {'bfloat16':<{column_width}}")
print("-" * (4 + 2 * column_width))
for i, url in enumerate(IMAGE_URLS, start=1):
    image = load_image(url)
    cap_fp32 = caption(model_fp32, image)
    cap_bf16 = caption(model_bf16, image)

    # Wrap long captions so both columns stay aligned
    lines_fp32 = textwrap.wrap(cap_fp32, column_width) or [""]
    lines_bf16 = textwrap.wrap(cap_bf16, column_width) or [""]
    for j in range(max(len(lines_fp32), len(lines_bf16))):
        left = lines_fp32[j] if j < len(lines_fp32) else ""
        right = lines_bf16[j] if j < len(lines_bf16) else ""
        number = str(i) if j == 0 else ""
        print(f"{number:<3} {left:<{column_width}} {right:<{column_width}}")
    print()


def compare_parameters(m_ref, m_other):
    # Compare the two models parameter by parameter, using the float32 model as the reference
    ref_params = dict(m_ref.named_parameters())
    other_params = dict(m_other.named_parameters())
    assert ref_params.keys() == other_params.keys(), "parameter names differ"

    total_elements = 0
    total_abs_diff = 0.0
    max_abs_diff = 0.0
    worst = []  # (max abs diff, name) for each parameter

    for name, p_ref in ref_params.items():
        p_other = other_params[name]
        # Cast both to float32 so the subtraction is exact enough to measure the rounding error
        diff = (p_ref.detach().float() - p_other.detach().float()).abs()
        total_elements += diff.numel()
        total_abs_diff += diff.sum().item()
        max_abs_diff = max(max_abs_diff, diff.max().item())
        worst.append((diff.max().item(), name))

    worst.sort(reverse=True)

    print("\nParameter comparison (float32 vs bfloat16)")
    print(f"  parameters compared : {len(ref_params)}")
    print(f"  elements compared   : {total_elements:,}")
    print(f"  mean abs difference : {total_abs_diff / total_elements:.6e}")
    print(f"  max abs difference  : {max_abs_diff:.6e}")
    print("  largest per-parameter max abs differences:")
    for value, name in worst[:5]:
        print(f"    {value:.6e}  {name}")


compare_parameters(model_fp32, model_bf16)

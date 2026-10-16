import copy
import sys

import torch

sys.path.insert(0, "/Users/nirupam/Rebuild/Quantization/Deeplearning_Course")
from helper import DummyModel

x = torch.LongTensor([[1, 0], [0, 1]])

base = DummyModel()
with torch.no_grad():
    ref = base(x).float()

    outputs = {"float32": ref}
    outputs["float16"] = copy.deepcopy(base).half()(x).float()
    outputs["bfloat16"] = copy.deepcopy(base).to(torch.bfloat16)(x).float()

    for name, out in outputs.items():
        print(f"\n== {name} ==")
        print(out)
        diff = (out - ref).abs()
        print(f"max abs diff vs float32: {diff.max().item():.6f}")
        print(f"mean abs diff vs float32: {diff.mean().item():.6f}")

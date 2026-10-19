"""
inspect_qwen.py — decompose an LLM's weights, layer by layer.

Goal: see how the dotted tensor names map to
  stack (model.layers) -> layer N -> attention (self_attn) -> Q/K/V matrices,
and confirm with real shapes everything we discussed.

Run it (no project setup needed) with uv:

    uv run --with "transformers>=4.45" --with torch --with safetensors --with huggingface_hub \
        python inspect_qwen.py

First run downloads ~1 GB to your HF cache (~/.cache/huggingface). Later runs are instant.
On Apple Silicon this loads fine in RAM; we never move it to GPU, we just read the numbers.
"""

from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ----------------------------------------------------------------------
# 1. THE CONFIG — the architecture spec. Everything below follows from these.
# ----------------------------------------------------------------------
section("1. CONFIG  (the blueprint — all shapes are derived from these)")

cfg = AutoConfig.from_pretrained(MODEL)
hidden = cfg.hidden_size
n_layers = cfg.num_hidden_layers
n_heads = cfg.num_attention_heads
n_kv = cfg.num_key_value_heads          # < n_heads  => grouped-query attention
head_dim = getattr(cfg, "head_dim", hidden // n_heads)

print(f"hidden_size            = {hidden}      (width of a token vector)")
print(f"num_hidden_layers      = {n_layers}       (how many stacked blocks)")
print(f"num_attention_heads    = {n_heads}       (Q heads)")
print(f"num_key_value_heads    = {n_kv}        (K/V heads  -> GQA if < Q heads)")
print(f"head_dim               = {head_dim}       (size of one head)")
print(f"intermediate_size      = {cfg.intermediate_size}     (MLP inner width)")
print(f"vocab_size             = {cfg.vocab_size}")
print(f"hidden activation      = {cfg.hidden_act}     (the MLP nonlinearity)")
print(f"tie_word_embeddings    = {cfg.tie_word_embeddings}   (lm_head reuses embed_tokens if True)")
print(f"\nSanity: n_heads * head_dim = {n_heads*head_dim} should equal hidden_size = {hidden}")
print(f"K/V width = n_kv * head_dim = {n_kv*head_dim}  (smaller than hidden -> K,V are smaller than Q)")


# ----------------------------------------------------------------------
# 2. LOAD THE WEIGHTS and list EVERY tensor, grouped by role.
#    named_parameters() gives exactly the dotted names from the tree.
# ----------------------------------------------------------------------
section("2. ALL TENSORS, grouped  (this is the weight file, decomposed)")

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

# Group names so 24 identical layers don't flood the screen: we collapse
# "model.layers.<N>.rest" -> "model.layers.*.rest" and show it once.
groups = defaultdict(list)
total_params = 0
for name, p in model.named_parameters():
    total_params += p.numel()
    key = name
    parts = name.split(".")
    if parts[:2] == ["model", "layers"]:
        parts[2] = "*"                 # collapse the layer index
        key = ".".join(parts)
    groups[key].append((name, tuple(p.shape)))

for key, members in groups.items():
    shape = members[0][1]
    count = len(members)
    tag = f"   (x{count} layers)" if count > 1 else ""
    print(f"{key:<48} {str(shape):<16}{tag}")


# ----------------------------------------------------------------------
# 3. ZOOM INTO LAYER 0'S ATTENTION — identify Q, K, V by hand.
# ----------------------------------------------------------------------
section("3. LAYER 0 ATTENTION  (Q / K / V / O identified)")

sa = model.model.layers[0].self_attn
# PyTorch Linear stores weight as [out_features, in_features] and computes
# y = x @ W.T + b  — i.e. the STORED matrix is the transpose of the W in q = x·W_Q.

def describe(linear, label, heads):
    w = linear.weight
    out, inp = w.shape
    has_bias = linear.bias is not None
    print(f"{label:<8} weight {str(tuple(w.shape)):<14} "
          f"-> {out} outputs = {heads} heads x {head_dim} head_dim | bias={has_bias}")

describe(sa.q_proj, "q_proj", n_heads)   # makes all Q heads
describe(sa.k_proj, "k_proj", n_kv)      # makes fewer K heads  (GQA)
describe(sa.v_proj, "v_proj", n_kv)      # makes fewer V heads
describe(sa.o_proj, "o_proj", n_heads)   # merges heads back to hidden

print("\nNote: k_proj/v_proj are SHORTER than q_proj -> grouped-query attention.")
print("Same embedding feeds all three, but each matrix is its own learned object.")

# Peek at a few actual numbers so Q/K/V stop being abstract:
print("\nFirst 5 numbers of q_proj row 0:", sa.q_proj.weight[0, :5].tolist())
print("First 5 numbers of k_proj row 0:", sa.k_proj.weight[0, :5].tolist())


# ----------------------------------------------------------------------
# 4. LAYER 0 MLP — where the SiLU/SwiGLU nonlinearity lives.
# ----------------------------------------------------------------------
section("4. LAYER 0 MLP  (gate/up feed the activation; down projects back)")

mlp = model.model.layers[0].mlp
print(f"gate_proj weight {tuple(mlp.gate_proj.weight.shape)}  -> feeds the {cfg.hidden_act} activation")
print(f"up_proj   weight {tuple(mlp.up_proj.weight.shape)}  -> expands to inner width")
print(f"down_proj weight {tuple(mlp.down_proj.weight.shape)}  -> back to hidden")
print("Computation: down_proj( SiLU(gate_proj(x)) * up_proj(x) )")
print("SiLU itself has NO weights — it is a fixed function, invisible in the file.")


# ----------------------------------------------------------------------
# 5. PARAMETER ACCOUNTING — note: NOTHING here depends on sequence length.
# ----------------------------------------------------------------------
section("5. PARAMETER COUNT  (independent of how many tokens you feed)")

per_category = defaultdict(int)
for name, p in model.named_parameters():
    if "embed_tokens" in name:
        cat = "embedding table (per-token)"
    elif ".self_attn." in name:
        cat = "attention projections (shared across tokens)"
    elif ".mlp." in name:
        cat = "MLP projections (shared across tokens)"
    elif "layernorm" in name or name.endswith("model.norm.weight"):
        cat = "RMSNorm scales"
    else:
        cat = "other (lm_head, etc.)"
    per_category[cat] += p.numel()

for cat, n in per_category.items():
    print(f"{cat:<46} {n:>12,}  ({n/total_params:5.1%})")
print("-" * 70)
print(f"{'TOTAL':<46} {total_params:>12,}")
print("\nFeed 4 tokens or 4000 tokens -> this number does not change.")
print("Sequence length changes the COMPUTE, never the PARAMETER COUNT.")


# ----------------------------------------------------------------------
# 6. (Optional) read the raw .safetensors file directly, no model object.
#    This is the most literal 'decompose the weight file' view.
# ----------------------------------------------------------------------
section("6. RAW FILE VIEW  (optional — reading model.safetensors directly)")
try:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(MODEL, filename="model.safetensors")
    print(f"file: {path}\n")
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        print(f"{len(keys)} tensors stored in the file. First 12 names:")
        for k in keys[:12]:
            print("  ", k, tuple(f.get_slice(k).get_shape()))
except Exception as e:
    print("Skipped raw view:", e)

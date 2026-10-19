This is the right way to cement it — reading real weights turns all the abstract Q/K/V talk into tensors you can print. Let me set you up with a specific model, explain the full top-down structure, show you exactly how tensor names map to "stack → layer → attention → Q/K/V," and give you a script that dumps everything.

## The model to download: Qwen2.5-0.5B

I'm picking **`Qwen/Qwen2.5-0.5B`** deliberately:

- **Ungated** — downloads with no license click or login (unlike Llama, which gates behind an access request).
- **Small** — ~1 GB, loads comfortably in RAM on your Apple Silicon Mac.
- **Modern architecture** — it's a Llama-style decoder-only Transformer with *separate* `q_proj`, `k_proj`, `v_proj` tensors. This matters for your goal: older models like GPT-2 fuse Q/K/V into one blob called `c_attn`, which hides exactly the thing you're trying to see. Qwen keeps them separate and even named.
- It uses the current standard pieces: **RMSNorm**, **SwiGLU** MLP, **RoPE** positions, and **grouped-query attention (GQA)** — and GQA will visibly show you that K and V are *smaller* than Q, which is a beautiful confirmation that key-space and query-space are genuinely different spaces.

(If you'd rather use Llama-3.2-1B later, the architecture and tensor names are nearly identical — same mental map applies. Qwen just avoids the gating friction today.)

## 1. The top-down nesting

A decoder-only LLM is a set of boxes inside boxes. Here is the whole thing for Qwen2.5-0.5B (24 layers):

```
Qwen2ForCausalLM                          ← the whole model
│
├── model.embed_tokens        [vocab × hidden]   ← per-token embedding table (the ONE per-token thing)
│
├── model.layers.0            ← "decoder layer" / "block" 0  ┐
├── model.layers.1                                           │  24 identical-shaped
├── ...                                                      │  blocks, stacked
├── model.layers.23                                          ┘
│       each block contains:
│       ├── input_layernorm            (RMSNorm, scale only)
│       ├── self_attn                  ← the attention sub-block
│       │     ├── q_proj   ← Q matrix  (+ bias in Qwen)
│       │     ├── k_proj   ← K matrix  (+ bias)
│       │     ├── v_proj   ← V matrix  (+ bias)
│       │     └── o_proj   ← output projection (no bias)
│       ├── post_attention_layernorm  (RMSNorm)
│       └── mlp                        ← the feed-forward sub-block
│             ├── gate_proj
│             ├── up_proj
│             └── down_proj
│
├── model.norm               (final RMSNorm)
│
└── lm_head                  [hidden × vocab]   ← projects back to vocabulary
        (in 0.5B this is TIED to embed_tokens — same weights reused, so it
         may not appear as a separate tensor in the file)
```

Read it as a hierarchy of names: **`model` → `layers` → `N` → `self_attn` → `q_proj`**. That dotted path *is* "by stack, by layer, by attention, which matrix." There's no mystery registry — the name literally encodes the location. "Stack" = the `model.layers` list; "layer N" = the index; "attention" = `self_attn`; "Q/K/V" = `q_proj / k_proj / v_proj`.

Every block has the **same shape**; they differ only in their learned numbers. Layer 0 and layer 23 have identical structure.

## 2. How the weights actually live on disk

Qwen ships its weights as a **`.safetensors`** file (Qwen2.5-0.5B is small enough to be a single file, `model.safetensors`). Safetensors is just a flat dictionary:

```
{ "tensor_name"  :  raw_numbers_with_a_shape_and_dtype, ... }
```

So the file is literally a mapping from those dotted names above to arrays of numbers. "Decomposing the weight file" = iterating that dictionary and reading each name + shape. That's the whole job, and the script does exactly this.

## 3. One attention block, every tensor named and sized

Here's layer 0's attention, with the shapes you'll see. Config: `hidden = 896`, `num_attention_heads = 14`, `num_key_value_heads = 2`, `head_dim = 64` (because 896 / 14 = 64).

| Tensor name | Shape `[out, in]` | What it is |
|---|---|---|
| `model.layers.0.self_attn.q_proj.weight` | `[896, 896]` | **Q matrix** — makes 14 query heads (14×64 = 896) |
| `model.layers.0.self_attn.q_proj.bias` | `[896]` | query bias |
| `model.layers.0.self_attn.k_proj.weight` | `[128, 896]` | **K matrix** — makes only **2** key heads (2×64 = 128) |
| `model.layers.0.self_attn.k_proj.bias` | `[128]` | key bias |
| `model.layers.0.self_attn.v_proj.weight` | `[128, 896]` | **V matrix** — only 2 value heads |
| `model.layers.0.self_attn.v_proj.bias` | `[128]` | value bias |
| `model.layers.0.self_attn.o_proj.weight` | `[896, 896]` | combines heads back to hidden size |

Two things jump out, and both are teaching gold:

**(a) K and V are smaller than Q** (`128` out vs `896` out). That's **grouped-query attention**: 14 query heads share just 2 key/value heads (7 queries per KV group). It's a memory optimization. But conceptually it's a direct, physical proof of the point from before — *the matrix that makes keys is a genuinely different, separately-sized object from the matrix that makes queries.* You're seeing Q-space and K-space have different dimensions in the raw file.

**(b) The shape is `[out, in]`, i.e. the transpose of what I wrote earlier.** Earlier I used $q = x \cdot W_Q$ with $W_Q$ shaped `[in, out]`. PyTorch's `Linear` layer stores the weight as `[out, in]` and computes $y = x W^\top + b$. So when you print `q_proj.weight` and see `[896, 896]`, read it as "896 outputs, 896 inputs." This trips up everyone reading raw weights the first time — flag it mentally now. Same math, stored transposed.

The MLP block of the same layer:

| Tensor name | Shape | Role |
|---|---|---|
| `model.layers.0.mlp.gate_proj.weight` | `[4864, 896]` | expands + feeds the gate |
| `model.layers.0.mlp.up_proj.weight` | `[4864, 896]` | expands to the wide hidden dim |
| `model.layers.0.mlp.down_proj.weight` | `[896, 4864]` | projects back down |

## 4. Where activation functions fit — and why they're *not* in the weight file

This answers your activation question directly. There are exactly **two nonlinearities**, and neither is a learned matrix:

**Softmax — inside attention.** After you compute $QK^\top/\sqrt{d_k}$ (scores), softmax turns them into the weights that sum to 1. It has **no parameters** — it's a pure function. You will *not* find a "softmax" tensor in the file. It's applied at runtime between the Q/K multiply and the V multiply.

**SiLU / SwiGLU — inside the MLP.** The MLP computes, roughly:
$$\text{down\_proj}\big(\ \text{SiLU}(\text{gate\_proj}(x)) \ \odot\ \text{up\_proj}(x)\ \big)$$
SiLU (also called Swish) is $\text{SiLU}(z) = z \cdot \sigma(z)$ — the actual nonlinearity. The `gate_proj` exists *to feed* that nonlinearity; the element-wise product $\odot$ is the "gating." SiLU itself has **no weights** — it's a fixed function applied to the gate's output.

So the division of labor is clean, and worth burning in:

- **Q, K, V, O, gate, up, down projections** = *linear* maps = the learned matrices = what's in the file.
- **softmax and SiLU** = *nonlinear* functions = fixed, parameter-free = applied at runtime, invisible in the weight file.
- **RMSNorm weights** (`input_layernorm`, `post_attention_layernorm`, `model.norm`) = a learned *scale vector* only (shape `[896]`, no matrix, no bias) — they rescale, they don't mix dimensions.

The attention mechanism's *only* nonlinearity is softmax; every Q/K/V step around it is pure linear algebra. The per-token "thinking" nonlinearity lives in the MLP's SiLU. That split — linear attention projections + nonlinear MLP — is the basic rhythm of every Transformer layer.

## 5. Why the parameter count is independent of sequence length

When you run the script and sum parameters, notice: the total is just the sum of those matrix sizes × 24 layers, plus the embedding table. **Nowhere does the number of input tokens appear.** A 4-token prompt and a 4000-token prompt flow through the *identical* frozen matrices. Sequence length affects *how much computation* happens (more tokens = more rows in $X$), never *how many parameters exist*. This is the concrete payoff of "shared across tokens" from the last message — you'll see it in the arithmetic.

---

I've written you a fully-commented inspection script. It prints the config, lists every tensor grouped by role, zooms into layer 0's Q/K/V with annotations, and does the parameter-count arithmetic so the points above become numbers on your screen.

Run `inspect_qwen.py` and read its output section by section against what's above — the printed shapes are ground truth, and I quoted the config numbers from memory, so let the script correct any I got slightly off (head counts and `intermediate_size` especially). Everything structural — the dotted names, the `[out, in]` transpose, K/V being smaller than Q — will hold.

## Teacher's notes — how to think while you read the weights

**1. The whole model is one function composed of alternating linear and nonlinear steps.** Strip away the names and a decoder layer is: normalize → linear (Q,K,V) → softmax-mix → linear (O) → add back → normalize → linear (gate,up) → nonlinear (SiLU) → linear (down) → add back. The "residual add-back" (the `+` that reinjects the block's input) is why you can stack 24 of them without the signal degrading — each block *edits* the running vector rather than replacing it. Watch for it conceptually; it has no weights, so it won't show in the file, but it's the reason depth works.

**2. Reading shapes is reading a composition of maps.** With your linear-algebra eye: a token starts as a vector in $\mathbb{R}^{896}$. `q_proj` maps it to $\mathbb{R}^{896}$ (14 heads of 64), `k_proj` to $\mathbb{R}^{128}$ (2 heads of 64). Different output dimensions = different target spaces. Every `.weight` you print is just "which space does this map send the vector into, and from where" — read `[out, in]` as "from $\mathbb{R}^{in}$ to $\mathbb{R}^{out}$."

**3. Parameter count ≈ a few matrix areas × depth.** Roughly, one layer's weight budget is $4 \times (\text{attn projections}) + 3 \times (\text{MLP projections})$, and the MLP usually dominates because `intermediate_size` (4864) is much bigger than hidden (896). When you see the per-category percentages in section 5, you'll find the MLP holds most of the parameters — attention is cheaper than people assume. That's a useful fact when you later think about quantization or memory (which, given your quantization project, is directly relevant — the biggest tensors to compress are the MLP's `up`/`gate`/`down`, not Q/K/V).

**4. Once you can name a tensor, you can edit it.** The real power of this map: `model.model.layers[5].self_attn.q_proj.weight` is a live, indexable array. You can zero a head, scale a matrix, measure a norm, swap in a quantized version — all by walking this exact path. Decomposing the file isn't just inspection; it's the handle for every intervention you'll do later.

Once you've run it and the shapes are concrete, the self-attention math ($\text{softmax}(QK^\top/\sqrt{d_k})V$ across these very matrices) will drop straight into place — say the word and we'll walk a sentence through layer 0 end to end.

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

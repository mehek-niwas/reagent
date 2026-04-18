# SELFIE on K2-V2 writer/editor agents

Minimum-viable version. All files sit in one directory; open `demo.ipynb` in
Jupyter from that directory.

## Layout

```
selfie_k2v2/
├── selfie_fork/
│   ├── __init__.py
│   └── interpret.py        # Ported SELFIE (modern transformers, LLaMA-arch)
├── k2v2_backend.py         # Loads K2-V2-Instruct, wraps the 3 ops
├── agents_graph.py         # LangGraph: writer → editor + 3 interpretation nodes
├── demo.ipynb              # Run this
└── README.md
```

## Install

```bash
pip install 'transformers>=4.45,<4.60' accelerate bitsandbytes langgraph pandas torch
```

## Hardware

K2-V2-Instruct is 70B. bf16 needs ~140GB VRAM. Options:

- 2×A100-80GB or 4×A6000-48GB with `device_map="auto"` (default in the notebook)
- Single 48GB GPU with 4-bit quantization — see the commented-out
  `BitsAndBytesConfig` block in `demo.ipynb`

## What runs

The LangGraph flow:

1. **writer_node** — K2-V2 writes a short paragraph from the user's task.
   Hidden states at all layers captured via a second forward pass on the
   full (prompt + output) sequence.
2. **editor_node** — K2-V2 evaluates the draft. Writer's raw token IDs are
   spliced directly into the editor's prompt (no detokenize/retokenize) so
   token positions are preserved for Experiment 3. Hidden states captured.
3. **selfie_writer** (Experiment 1) — SELFIE on writer's hidden states at
   each output-token position, across 3 probe layers (early-mid, mid, late).
4. **selfie_editor** (Experiment 3) — SELFIE on editor's hidden states at
   the draft-span positions, same 3 probe layers.
5. **injected_editor** (Experiment 2) — Editor's prompt rebuilt with
   placeholder tokens at the draft position; their embeddings are overwritten
   with the writer's final-layer hidden states. Editor generates a verdict
   without ever seeing the surface text.

The final notebook state contains:
- `writer_output_text`, `editor_verdict`, `editor_injected_verdict`
- `selfie_writer` (DataFrame), `selfie_editor_on_draft` (DataFrame)
- A merged side-by-side DataFrame aligning both SELFIEs by `(layer, rel_pos)`

## Known sharp edges (debug tips)

- **Interpretation prompt is plain text, not chat-templated.** SELFIE works
  better when the interpretation prompt is a flat sequence you control
  position-by-position. If the interpretations come out empty or nonsensical,
  try increasing `num_placeholders` in `make_interpretation_prompt`, or
  reword the prompt in `k2v2_backend.py`.
- **Experiment 2 can fail silently** if the writer's final hidden state is
  too "late" to be useful input at layer 0 of the editor. If the injected
  verdict is nonsense, change `inject_layer=0` to `inject_layer=1` or a
  middle layer in `agents_graph.injected_editor_node`. Matching layer
  indices between writer-source and editor-inject is usually more coherent
  than crossing layer depths.
- **Memory per SELFIE call**: every `(layer, token_idx)` row runs a full
  generation of the interpretation prompt. With 3 layers × ~30 writer
  tokens = 90 extra short generations. Shrink `max_writer_tokens` if this
  gets slow.
- **Token alignment** depends on the marker splice trick in
  `agents_graph.editor_node`. If the chat template changes across
  `transformers` versions and the marker tokens render differently, the
  `_find_sublist` lookup will fail with a clear error message.

## Why this should work at all

K2-V2 on HF is tagged as a `llama` architecture — same `model.model.layers[i]`
structure that original SELFIE exercises. Hidden size 8192, 80 layers.
Injection across positions in the residual stream uses the same math as
Llama; GQA doesn't affect this because we patch the residual, not K/V.

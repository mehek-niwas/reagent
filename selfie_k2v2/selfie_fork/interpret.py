"""
SELFIE interpretation, ported for modern transformers (4.45+) and K2-V2.

Key idea (unchanged from original SELFIE):
  - Build an "interpretation prompt" with placeholder tokens.
  - Run the target prompt once with output_hidden_states=True, capture the
    hidden state H at (layer L, token t) that we want to interpret.
  - Run the interpretation prompt with H injected at the placeholders.
  - The decoded continuation is the natural-language interpretation.

Performance improvement over the original:
  All token positions that share the same layer are batched into a SINGLE
  model.generate() call.  The batch dimension maps one-to-one to the tokens
  being interpreted, so N tokens × K layers → K generate() calls instead of
  N×K calls.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd


# ---------- Interpretation prompt ----------

class InterpretationPrompt:
    """
    Build the interpretation prompt with placeholder slots.

    Pass a tuple of strings and zeros, e.g.:
        ("Summarize the message: '", 0, 0, 0, "'. The message says:")

    Each 0 is one placeholder token whose embedding will be overwritten with
    the captured hidden state during the patched forward pass.
    """

    def __init__(self, tokenizer, template: Tuple):
        self.tokenizer = tokenizer
        self.template = template

        dummy_id = 0  # embedding will be overwritten
        ids: List[int] = []
        placeholder_positions: List[int] = []

        bos = tokenizer.bos_token_id
        if bos is not None:
            ids.append(bos)

        for part in template:
            if part == 0:
                placeholder_positions.append(len(ids))
                ids.append(dummy_id)
            else:
                sub = tokenizer.encode(part, add_special_tokens=False)
                ids.extend(sub)

        self.input_ids = torch.tensor([ids], dtype=torch.long)
        self.placeholder_positions = placeholder_positions
        self.num_placeholders = len(placeholder_positions)


# ---------- Capturing hidden states from a target prompt ----------

@torch.no_grad()
def capture_hidden_states(model, tokenizer, prompt_text: str, device=None):
    """
    Run the target prompt once and return (input_ids, hidden_states).

    hidden_states[L] has shape (1, seq_len, hidden_size).
    Index 0 = embedding output; index L = output of decoder layer L-1.
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return input_ids, out.hidden_states


# ---------- Batched patched forward ----------

@torch.no_grad()
def _generate_batch_patched(
    model,
    tokenizer,
    interp_input_ids: torch.Tensor,          # (1, prompt_len)
    placeholder_positions: List[int],
    injected_hidden_batch: torch.Tensor,     # (n_tokens, hidden_size)
    layer_idx: int,
    max_new_tokens: int = 20,
    chunk_size: int = 32,
) -> List[str]:
    """
    Run SELFIE for a BATCH of hidden states in one (or a few) generate() calls.

    injected_hidden_batch[i] is injected at the placeholder positions for the
    i-th element of the batch.  All elements use the same interpretation prompt
    (interp_input_ids repeated n_tokens times).

    Because the prompts are identical the only difference between batch elements
    is which hidden state is injected, so no padding is needed.

    chunk_size: maximum batch size per generate() call to avoid OOM.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n = injected_hidden_batch.shape[0]
    if n == 0:
        return []

    prompt_len = interp_input_ids.shape[1]
    interp_ids = interp_input_ids.to(device)   # (1, prompt_len)
    positions = list(placeholder_positions)
    pos_idx = torch.as_tensor(positions, device=device, dtype=torch.long)
    n_ph = pos_idx.shape[0]

    results: List[str] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch_inj = injected_hidden_batch[start:end].to(device=device, dtype=dtype)
        bs = end - start

        batch_ids = interp_ids.expand(bs, -1).contiguous()  # (bs, prompt_len)

        # Pre-expand batch_inj across the placeholder span once per chunk so
        # the patch becomes a single scatter: (bs, n_ph, H) → hidden[:, pos_idx, :]
        batch_inj_expanded = batch_inj.unsqueeze(1).expand(bs, n_ph, -1)

        def _patch(hidden: torch.Tensor) -> torch.Tensor:
            # hidden: (bs, seq_len, hidden_size)
            # Only patch during the prompt-length prefill pass, not decode steps
            if hidden.shape[1] >= prompt_len:
                hidden[:, pos_idx, :] = batch_inj_expanded
            return hidden

        handles = []
        if layer_idx == 0:
            embed = model.get_input_embeddings()
            handles.append(embed.register_forward_hook(lambda m, i, o: _patch(o)))
        else:
            # Locate the right decoder layer stack (same logic as model_backend.py)
            layers = _get_layers(model)
            target = min(layer_idx - 1, len(layers) - 1)
            layer = layers[target]

            def _pre_hook(module, args, kwargs):
                if args and torch.is_tensor(args[0]):
                    return (_patch(args[0]),) + args[1:], kwargs
                if "hidden_states" in kwargs:
                    return args, {**kwargs, "hidden_states": _patch(kwargs["hidden_states"])}
                return args, kwargs

            handles.append(layer.register_forward_pre_hook(_pre_hook, with_kwargs=True))

        # Explicit all-ones attention mask: the placeholder token id (0) may
        # coincide with the pad_token_id, which would cause generate() to
        # auto-mask our injected positions.  Forcing ones guarantees the model
        # attends to them.
        attention_mask = torch.ones_like(batch_ids)

        try:
            out_ids = model.generate(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        finally:
            for h in handles:
                h.remove()

        for i in range(bs):
            gen_ids = out_ids[i, prompt_len:]
            results.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    return results


def _get_layers(model) -> torch.nn.ModuleList:
    """Best-effort decoder layer lookup (mirrors model_backend._get_decoder_layers)."""
    for probe in (
        lambda m: m.model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ):
        try:
            layers = probe(model)
            if layers is not None:
                return layers
        except AttributeError:
            pass
    raise RuntimeError(f"Cannot locate decoder layers in {type(model).__name__}")


# ---------- High-level: interpret a list of (layer, token) positions ----------

@torch.no_grad()
def interpret(
    original_prompt: str,
    tokens_to_interpret: List[Tuple[int, int]],  # [(layer, token_idx), ...]
    model,
    tokenizer,
    interpretation_prompt: InterpretationPrompt,
    max_new_tokens: int = 20,
    hidden_states_override: Optional[Tuple[torch.Tensor, ...]] = None,
    input_ids_override: Optional[torch.Tensor] = None,
    chunk_size: int = 32,
) -> pd.DataFrame:
    """
    For each (layer, token_idx) in tokens_to_interpret, produce a natural-
    language interpretation by patching the interpretation prompt.

    Batched: all positions sharing the same layer are processed in a single
    model.generate() call, which is dramatically faster than the original
    one-call-per-position loop.

    Positions that exceed the hidden_states sequence length are silently
    skipped (avoids the off-by-one error from inline hidden-state capture
    where the final EOS token has no recorded hidden state).

    If hidden_states_override / input_ids_override are provided, skips the
    capture step and uses those directly.

    Returns a DataFrame: layer | token_idx | token | interpretation
    """
    device = next(model.parameters()).device

    if hidden_states_override is None:
        input_ids, hidden_states = capture_hidden_states(
            model, tokenizer, original_prompt
        )
    else:
        input_ids = input_ids_override
        hidden_states = hidden_states_override

    # Group by layer; filter out positions that exceed the sequence length
    layer_to_idxs: Dict[int, List[int]] = defaultdict(list)
    for (layer, token_idx) in tokens_to_interpret:
        hs_len = hidden_states[layer].shape[1]
        if token_idx < hs_len:
            layer_to_idxs[layer].append(token_idx)
        # else: silently skip (EOS / off-by-one from inline capture)

    rows = []
    for layer, token_idxs in layer_to_idxs.items():
        # Gather all hidden states for this layer as a batch
        hs_batch = hidden_states[layer][0, token_idxs, :].detach()  # (n, hidden_size)

        texts = _generate_batch_patched(
            model=model,
            tokenizer=tokenizer,
            interp_input_ids=interpretation_prompt.input_ids,
            placeholder_positions=interpretation_prompt.placeholder_positions,
            injected_hidden_batch=hs_batch,
            layer_idx=layer,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
        )

        for token_idx, text in zip(token_idxs, texts):
            tok_str = tokenizer.decode(
                [input_ids[0, token_idx].item()], skip_special_tokens=True
            )
            rows.append({
                "layer": layer,
                "token_idx": token_idx,
                "token": tok_str,
                "interpretation": text,
            })

    return pd.DataFrame(rows, columns=["layer", "token_idx", "token", "interpretation"])

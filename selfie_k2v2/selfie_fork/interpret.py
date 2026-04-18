"""
SELFIE interpretation, ported for modern transformers (4.45+) and K2-V2.

K2-V2 is registered as a `llama` architecture on HF, so model.model.layers[i]
is the decoder layer stack, same as LlamaForCausalLM. This file assumes that
structure.

Key idea (unchanged from original SELFIE):
  - Build an "interpretation prompt" with placeholder tokens (we use the
    first vocab token, id 0, as a dummy; its embedding will be overwritten).
  - Run the target prompt once with output_hidden_states=True, capture the
    hidden state H at (layer L, token t) that we want to interpret.
  - Run the interpretation prompt. On the forward of layer L, overwrite the
    hidden states at the placeholder positions with H.
  - Continue generation. The decoded text is the interpretation.

This implementation uses a forward *pre-hook* on the chosen layer that patches
the incoming hidden states. We repeat the placeholder `k` times (= the number
of tokens we want to interpret together, usually 1 for single-token probing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import pandas as pd


# ---------- Interpretation prompt ----------

class InterpretationPrompt:
    """
    Build the interpretation prompt with placeholder slots.

    Pass a tuple of strings and zeros, e.g.:
        ("Summarize the message: '", 0, 0, 0, "'. The message says:")

    Each 0 is one placeholder token. All placeholders are filled with the
    same captured hidden state H at layer L during the patched forward pass.
    """

    def __init__(self, tokenizer, template: Tuple):
        self.tokenizer = tokenizer
        self.template = template

        # Build input_ids by tokenizing string segments and inserting a dummy
        # token id for each placeholder.
        dummy_id = 0  # we'll overwrite its embedding anyway
        ids: List[int] = []
        placeholder_positions: List[int] = []

        # Make sure we start with BOS if the tokenizer uses one.
        bos = tokenizer.bos_token_id
        if bos is not None:
            ids.append(bos)

        for part in template:
            if part == 0:
                placeholder_positions.append(len(ids))
                ids.append(dummy_id)
            else:
                # no special tokens — we handle BOS ourselves
                sub = tokenizer.encode(part, add_special_tokens=False)
                ids.extend(sub)

        self.input_ids = torch.tensor([ids], dtype=torch.long)
        self.placeholder_positions = placeholder_positions
        self.num_placeholders = len(placeholder_positions)


# ---------- Capturing hidden states from a target prompt ----------

@torch.no_grad()
def capture_hidden_states(model, tokenizer, prompt_text: str, device: Optional[str] = None):
    """
    Run the target prompt once and return (input_ids, hidden_states).

    hidden_states is a tuple of length num_layers+1, each of shape
      (1, seq_len, hidden_size).
    Index 0 is the embedding output; index L is the output of decoder layer L-1.
    We treat `layer=L` downstream as "the output of layer L-1", matching the
    original SELFIE convention (they index hidden_states directly).
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    return input_ids, out.hidden_states


# ---------- The patched forward with the hook ----------

@torch.no_grad()
def _generate_with_patched_layer(
    model,
    tokenizer,
    interp_input_ids: torch.Tensor,
    placeholder_positions: List[int],
    injected_hidden: torch.Tensor,  # shape (hidden_size,) or (num_placeholders, hidden_size)
    layer_idx: int,
    max_new_tokens: int = 20,
):
    """
    Generate from the interpretation prompt while patching the decoder layer
    at layer_idx to overwrite placeholder positions with injected_hidden.

    layer_idx is the 1-based "after this layer" index matching hidden_states
    from capture_hidden_states (i.e. layer_idx=0 means patch the embeddings).
    Concretely: we patch the INPUT to decoder layer `layer_idx` so that
    hidden_states going into that layer have the injected vector at the
    placeholder positions.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    interp_input_ids = interp_input_ids.to(device)

    # Normalize injected_hidden to shape (num_placeholders, hidden_size)
    if injected_hidden.dim() == 1:
        injected = injected_hidden.unsqueeze(0).expand(len(placeholder_positions), -1)
    else:
        injected = injected_hidden
    injected = injected.to(device=device, dtype=dtype)

    positions = list(placeholder_positions)
    prompt_len = interp_input_ids.shape[1]

    # Hook target:
    #   layer_idx == 0  -> patch the embedding output (before layer 0)
    #   layer_idx >= 1  -> patch the input to decoder layer layer_idx-1 ...
    #       Actually, to match SELFIE's convention where hidden_states[L]
    #       is "the output of layer L-1" (with [0] being embeddings), and
    #       the user says "interpret hidden_states[L]" meaning the state
    #       that FLOWS OUT of layer L-1 and INTO layer L, we patch the
    #       input of layer L. Equivalently: pre-hook on model.model.layers[L]
    #       for L>=1, and for L==0 we patch embed_tokens output.
    #
    #   For layer_idx == num_layers (i.e. the final hidden state after the
    #   last layer), there is no "next layer" to pre-hook. In that case we
    #   hook the LAST layer's post-forward and overwrite its output — but
    #   this won't affect downstream logits meaningfully because the model
    #   is about to project to vocab. For interpretation we typically want
    #   L in [1, num_layers-1], with middle layers being most interesting.

    handles = []

    def _patch(hidden):
        # hidden: (batch, seq, hidden). Patch only on the prompt tokens,
        # not on tokens generated past prompt_len (KV cache handles those
        # separately — on generation steps past the prompt, `seq` may be 1).
        if hidden.shape[1] >= prompt_len:
            for i, pos in enumerate(positions):
                hidden[:, pos, :] = injected[i]
        return hidden

    if layer_idx == 0:
        # Patch the embedding layer output.
        embed = model.get_input_embeddings()

        def embed_hook(module, inputs, output):
            return _patch(output)

        handles.append(embed.register_forward_hook(embed_hook))
    else:
        num_layers = len(model.model.layers)
        target_layer = min(layer_idx, num_layers - 1)
        layer = model.model.layers[target_layer]

        def pre_hook(module, args, kwargs):
            # In modern HF LLaMA, the first positional arg (or kwarg
            # 'hidden_states') is the residual stream.
            if len(args) > 0 and torch.is_tensor(args[0]):
                new_args = (_patch(args[0]),) + args[1:]
                return new_args, kwargs
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = _patch(kwargs["hidden_states"])
                return args, kwargs
            return args, kwargs

        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))

    try:
        out_ids = model.generate(
            input_ids=interp_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        for h in handles:
            h.remove()

    # Decode only the generated continuation.
    gen_ids = out_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---------- High-level: interpret a list of (layer, token) positions ----------

@torch.no_grad()
def interpret(
    original_prompt: str,
    tokens_to_interpret: List[Tuple[int, int]],  # list of (layer, token_idx)
    model,
    tokenizer,
    interpretation_prompt: InterpretationPrompt,
    max_new_tokens: int = 20,
    hidden_states_override: Optional[Tuple[torch.Tensor, ...]] = None,
    input_ids_override: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """
    For each (layer, token_idx) in tokens_to_interpret, produce a natural
    language interpretation by patching the interpretation prompt.

    If hidden_states_override / input_ids_override are provided, skip the
    capture step and use those (useful when you've already run the target
    model and don't want to re-run it — e.g. Experiment 3, where the
    "target prompt" is actually the editor's full forward pass).

    Returns a DataFrame with columns: layer, token_idx, token, interpretation.
    """
    device = next(model.parameters()).device

    if hidden_states_override is None:
        input_ids, hidden_states = capture_hidden_states(model, tokenizer, original_prompt)
    else:
        input_ids = input_ids_override
        hidden_states = hidden_states_override

    rows = []
    for (layer, token_idx) in tokens_to_interpret:
        # hidden_states[layer] has shape (1, seq_len, hidden_size)
        h = hidden_states[layer][0, token_idx].detach()

        interp_text = _generate_with_patched_layer(
            model=model,
            tokenizer=tokenizer,
            interp_input_ids=interpretation_prompt.input_ids,
            placeholder_positions=interpretation_prompt.placeholder_positions,
            injected_hidden=h,
            layer_idx=layer,
            max_new_tokens=max_new_tokens,
        )

        tok_str = tokenizer.decode([input_ids[0, token_idx].item()])
        rows.append({
            "layer": layer,
            "token_idx": token_idx,
            "token": tok_str,
            "interpretation": interp_text.strip(),
        })

    return pd.DataFrame(rows)

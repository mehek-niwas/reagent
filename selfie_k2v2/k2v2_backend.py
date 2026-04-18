"""
K2V2Backend: single shared model instance used by both writer and editor
agents, plus utilities for SELFIE and embedding injection.

Designed for LLM360/K2-V2-Instruct at reasoning_effort="medium".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfie_fork.interpret import (
    InterpretationPrompt,
    interpret,
    capture_hidden_states,
)


@dataclass
class GenerationResult:
    prompt_ids: torch.Tensor          # (1, prompt_len)
    output_ids: torch.Tensor          # (1, prompt_len + gen_len)
    output_text: str                  # decoded generated continuation only
    gen_token_ids: List[int]          # generated tokens only
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None  # full sequence, all layers
    prompt_len: int = 0


class K2V2Backend:
    def __init__(
        self,
        model_name: str = "LLM360/K2-V2-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        print(f"Loading {model_name} (this may take a few minutes)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = len(self.model.model.layers)
        print(f"Model loaded. hidden_size={self.hidden_size}, num_layers={self.num_layers}")

    # ---------- chat template helper ----------

    def build_chat_prompt(
        self,
        system: Optional[str],
        user: str,
        reasoning_effort: str = "medium",
    ) -> torch.Tensor:
        """
        Render a chat prompt through K2-V2's chat template. Returns input_ids
        on the model's device.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        # K2-V2-Instruct chat template supports reasoning_effort kwarg.
        try:
            ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=reasoning_effort,
            )
        except TypeError:
            # Fallback if template doesn't accept that kwarg
            ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        return ids.to(self.device)

    # ---------- normal generation with hidden-state capture ----------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        capture_hidden: bool = True,
    ) -> GenerationResult:
        """
        Greedy generation. If capture_hidden=True, after generation we run one
        final forward pass on the full (prompt + output) sequence to get
        hidden states at every layer for every position — this is simpler
        and more reliable than trying to collect per-step hidden states
        during generation.
        """
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]

        out_ids = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        gen_ids = out_ids[0, prompt_len:].tolist()
        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        hidden_states = None
        if capture_hidden:
            # Single forward on the full sequence to get all hidden states.
            out = self.model(
                input_ids=out_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = tuple(h.detach() for h in out.hidden_states)

        return GenerationResult(
            prompt_ids=prompt_ids,
            output_ids=out_ids,
            output_text=output_text,
            gen_token_ids=gen_ids,
            hidden_states=hidden_states,
            prompt_len=prompt_len,
        )

    # ---------- SELFIE wrapper ----------

    def make_interpretation_prompt(self, num_placeholders: int = 4) -> InterpretationPrompt:
        """
        A simple interpretation prompt for K2-V2. Uses plain text (no chat
        template) because SELFIE's placeholder substitution is simpler with
        a flat token stream. Primed to complete with a description.
        """
        template = ("The following is a message. Summarize it in one sentence.\nMessage: '",)
        template = template + tuple([0] * num_placeholders)
        template = template + ("'\nSummary: The message is about",)
        return InterpretationPrompt(self.tokenizer, template)

    def selfie_on_positions(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        positions: List[Tuple[int, int]],  # [(layer, token_idx), ...]
        max_new_tokens: int = 20,
        num_placeholders: int = 4,
    ):
        interp_prompt = self.make_interpretation_prompt(num_placeholders)
        return interpret(
            original_prompt="",  # unused because we pass overrides
            tokens_to_interpret=positions,
            model=self.model,
            tokenizer=self.tokenizer,
            interpretation_prompt=interp_prompt,
            max_new_tokens=max_new_tokens,
            hidden_states_override=hidden_states,
            input_ids_override=input_ids,
        )

    # ---------- embedding injection generation (Experiment 2) ----------

    @torch.no_grad()
    def generate_with_injected_embeds(
        self,
        prompt_ids: torch.Tensor,
        placeholder_positions: List[int],     # positions in prompt_ids where we inject
        injected_hidden: torch.Tensor,        # (num_placeholders, hidden_size)
        inject_layer: int = 0,                # 0 = input embeds; L = after layer L-1
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate an editor response where the placeholder tokens in the prompt
        have their embeddings (or their residual-stream state at a chosen
        layer) overwritten with the writer's captured hidden state.

        inject_layer=0 → hook the embedding layer output.
        inject_layer>=1 → hook pre-forward of model.model.layers[inject_layer-1]
          so that the residual stream entering layer (inject_layer-1) is
          patched. Matches hidden_states indexing where [0]=embeds, [L]=output
          of layer L-1.
        """
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        dtype = next(self.model.parameters()).dtype
        injected = injected_hidden.to(device=self.device, dtype=dtype)
        assert injected.shape[0] == len(placeholder_positions), (
            f"injected rows ({injected.shape[0]}) != num placeholders "
            f"({len(placeholder_positions)})"
        )

        def _patch(hidden):
            # Only patch during the initial prompt forward. On subsequent
            # generation steps the model is fed one token at a time (seq=1)
            # so there are no placeholder positions to patch.
            if hidden.shape[1] >= prompt_len:
                for i, pos in enumerate(placeholder_positions):
                    hidden[:, pos, :] = injected[i]
            return hidden

        handles = []
        if inject_layer == 0:
            embed = self.model.get_input_embeddings()

            def embed_hook(module, inputs, output):
                return _patch(output)

            handles.append(embed.register_forward_hook(embed_hook))
        else:
            target_idx = min(inject_layer - 1, self.num_layers - 1)
            layer = self.model.model.layers[target_idx]

            def pre_hook(module, args, kwargs):
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
            out_ids = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            for h in handles:
                h.remove()

        gen_ids = out_ids[0, prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

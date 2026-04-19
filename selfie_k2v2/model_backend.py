"""
Generic ModelBackend for any HuggingFace decoder-only model, plus
HiddenStateProjector for cross-model hidden-state injection.

Supported architectures (auto-detected):
  - LLaMA 3.x family  (includes K2-V2, registered as llama-arch)
  - Qwen 2 / 2.5 / 3 / 3.5 family  (Qwen3/3.5: <think>…</think> blocks are
                                      stripped from output_text automatically
                                      while reasoning stays ON)
  - Gemma 2 / 3 / 4 family
  - Mistral / Mixtral family
  Falls back to .transformer.h (GPT-2/Falcon), .gpt_neox.layers (Pythia), etc.

The SELFIE patching mechanism (hook on residual stream at layer L) works for all
architectures above because HF standardises the .model.model.layers[i] path.
For cross-model injection (writer and editor have different architectures or
sizes) HiddenStateProjector handles the dimension mismatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .selfie_fork.interpret import (
    InterpretationPrompt,
    interpret,
    capture_hidden_states,
)

# Regex that matches an entire <think>…</think> block plus trailing whitespace.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    prompt_ids: torch.Tensor                                    # (1, prompt_len)
    output_ids: torch.Tensor                                    # (1, full_len) incl. thinking
    output_text: str                                            # clean answer (thinking stripped)
    gen_token_ids: List[int]                                    # answer-only tokens
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None   # all layers, full sequence
    prompt_len: int = 0
    # How many tokens separate prompt_end from the start of gen_token_ids in output_ids.
    # 0 for non-thinking models; > 0 when <think>…</think> prefix was stripped.
    answer_offset: int = 0
    # Raw thinking content (empty string if model has no thinking or none was generated).
    thinking_text: str = ""


# ---------------------------------------------------------------------------
# Cross-model hidden-state projection
# ---------------------------------------------------------------------------

class HiddenStateProjector:
    """
    Projects hidden states from a writer model's space into an editor model's space.

    When writer.hidden_size == editor.hidden_size, this is a no-op.

    When they differ, a random orthogonal projection is used as a structural
    baseline so the injection is dimensionally compatible.  NOTE: without
    training (e.g. via CKA / CCA alignment), this projection is semantically
    arbitrary.  It is provided so the plumbing works across architectures; a
    trained projection is needed for genuinely meaningful cross-model transfer.
    """

    def __init__(self, src_size: int, tgt_size: int):
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.needs_projection = src_size != tgt_size
        self._matrix: Optional[torch.Tensor] = None

        if self.needs_projection:
            raw = torch.randn(src_size, tgt_size)
            if src_size >= tgt_size:
                q, _ = torch.linalg.qr(raw)
                self._matrix = q[:, :tgt_size]          # (src, tgt)
            else:
                q, _ = torch.linalg.qr(raw.T)
                self._matrix = q[:tgt_size, :].T        # (src, tgt)
            # Rescale to approximately preserve vector norms
            self._scale = float((tgt_size / src_size) ** 0.5)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., src_size) → (..., tgt_size)"""
        if not self.needs_projection:
            return h
        mat = self._matrix.to(device=h.device, dtype=h.dtype)
        return (h @ mat) * self._scale

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return self.project(h)

    @classmethod
    def between(cls, writer: "ModelBackend", editor: "ModelBackend") -> "HiddenStateProjector":
        """Convenience constructor: infer sizes from two backends."""
        return cls(writer.hidden_size, editor.hidden_size)


# ---------------------------------------------------------------------------
# Main backend
# ---------------------------------------------------------------------------

class ModelBackend:
    """
    Loads any HuggingFace decoder-only model and exposes the generation,
    SELFIE, and hidden-state injection methods needed by the probe graph.

    Parameters
    ----------
    model_name : str
        HuggingFace model repo ID, e.g.:
          "Qwen/Qwen3.5-9B"
          "meta-llama/Llama-3.3-70B-Instruct"
          "google/gemma-3-27b-it"
          "LLM360/K2-V2-Instruct"
    dtype : torch.dtype
        Default bfloat16 for most modern models.
    device_map : str
        "auto" shards across all visible GPUs.
    quantization_config : optional
        Pass a BitsAndBytesConfig (or other HF quantization config) for 4-bit
        / 8-bit loading on smaller GPUs.
    chat_template_kwargs : dict
        Extra kwargs forwarded to apply_chat_template (e.g. reasoning_effort
        for K2-V2, enable_thinking for Qwen3/3.5).  Defaults to {} so thinking
        stays ON for Qwen3/3.5 (thinking blocks are stripped from output_text
        automatically — see _STRIP_THINKING_FAMILIES).
    """

    # Model families whose <think>…</think> blocks should be stripped from
    # output_text / gen_token_ids while keeping reasoning enabled.
    _STRIP_THINKING_FAMILIES: frozenset[str] = frozenset(["qwen3"])

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        quantization_config=None,
        chat_template_kwargs: Optional[dict] = None,
    ):
        self.model_name = model_name
        self._chat_template_kwargs = chat_template_kwargs or {}

        # Auto-detect whether this model family emits <think> blocks.
        name_lower = model_name.lower()
        self._strip_thinking: bool = any(
            f in name_lower for f in self._STRIP_THINKING_FAMILIES
        )

        print(f"Loading {model_name} ...")
        if self._strip_thinking:
            print("  → Qwen3/3.5 detected: reasoning ON, <think> blocks stripped from output.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs: dict = dict(torch_dtype=dtype, device_map=device_map)
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        self.device = next(self.model.parameters()).device
        self.hidden_size: int = self._get_hidden_size(self.model)
        self._layers = self._get_decoder_layers(self.model)
        self.num_layers: int = len(self._layers)

        print(
            f"Ready. hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, device={self.device}"
        )

    # ------------------------------------------------------------------
    # Architecture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_hidden_size(model) -> int:
        """
        Resolve hidden_size for any supported architecture.
        Multimodal models (e.g. Qwen3.5) nest their text config under
        config.text_config; fall back to config.hidden_size otherwise.
        """
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
            return text_cfg.hidden_size
        return cfg.hidden_size

    @staticmethod
    def _get_decoder_layers(model) -> torch.nn.ModuleList:
        """
        Auto-detect the decoder layer stack.  Tries common HF attribute paths
        in priority order so the same code works for LLaMA, Qwen, Gemma,
        Mistral, GPT-2, Falcon, GPT-NeoX, etc.
        """
        probes = [
            # LLaMA / Qwen2 / Qwen3 / Gemma / Mistral / K2-V2
            lambda m: m.model.layers,
            # Qwen3.5 multimodal loaded via AutoModelForCausalLM or
            # AutoModelForImageTextToText — text decoder sits under .language_model
            lambda m: m.language_model.model.layers,
            # GPT-2, Falcon (legacy)
            lambda m: m.transformer.h,
            # GPT-NeoX / Pythia
            lambda m: m.gpt_neox.layers,
            # Falcon (newer API)
            lambda m: m.transformer.blocks,
        ]
        for probe in probes:
            try:
                layers = probe(model)
                if layers is not None:
                    return layers
            except AttributeError:
                continue
        raise RuntimeError(
            f"Cannot auto-detect decoder layers for {type(model).__name__}. "
            "Please subclass ModelBackend and override _get_decoder_layers()."
        )

    # ------------------------------------------------------------------
    # Thinking-token handling  (Qwen3 / Qwen3.5)
    # ------------------------------------------------------------------

    def _find_think_boundary(self, gen_ids: List[int]) -> int:
        """
        Return the index into gen_ids where the ANSWER starts (i.e. after the
        </think> block and any trailing whitespace tokens).

        Strategy
        --------
        1. Try to find `</think>` as a known special token ID (O(n) scan).
        2. Fall back to decoding a prefix of increasing length until the
           decoded text contains `</think>` (O(n) calls to the tokenizer).

        Returns 0 if no thinking block is found (safe default for all models).
        """
        end_tag = "</think>"

        # --- Strategy 1: special-token lookup ----------------------------
        for candidate in ("</think>", "<|/think|>"):
            tid = self.tokenizer.convert_tokens_to_ids(candidate)
            unk = self.tokenizer.unk_token_id
            if tid is not None and tid != unk:
                try:
                    idx = gen_ids.index(tid)
                except ValueError:
                    continue
                # Skip trailing whitespace/newline tokens after </think>
                j = idx + 1
                while j < len(gen_ids):
                    tok_text = self.tokenizer.decode(
                        [gen_ids[j]], skip_special_tokens=False
                    )
                    if tok_text.strip():
                        break
                    j += 1
                return j

        # --- Strategy 2: prefix-decode scan ------------------------------
        buf = ""
        for i, tok_id in enumerate(gen_ids):
            buf += self.tokenizer.decode([tok_id], skip_special_tokens=False)
            if end_tag in buf:
                j = i + 1
                while j < len(gen_ids):
                    tok_text = self.tokenizer.decode(
                        [gen_ids[j]], skip_special_tokens=False
                    )
                    if tok_text.strip():
                        break
                    j += 1
                return j

        return 0  # no thinking block found

    def _post_process(self, gen_ids: List[int]) -> Tuple[List[int], int, str, str]:
        """
        Split gen_ids into answer tokens + metadata.

        Returns
        -------
        answer_ids     : List[int]  — token IDs of the answer (thinking stripped)
        answer_offset  : int        — how many gen tokens precede the answer
        output_text    : str        — decoded clean answer
        thinking_text  : str        — decoded thinking content (empty if none)
        """
        if not self._strip_thinking:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return gen_ids, 0, text, ""

        boundary = self._find_think_boundary(gen_ids)
        answer_ids = gen_ids[boundary:]
        output_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        # Extract thinking text for inspection
        if boundary > 0:
            raw = self.tokenizer.decode(gen_ids[:boundary], skip_special_tokens=False)
            m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            thinking_text = m.group(1).strip() if m else raw.strip()
        else:
            thinking_text = ""

        return answer_ids, boundary, output_text, thinking_text

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def build_chat_prompt(
        self,
        system: Optional[str],
        user: str,
        **extra_template_kwargs,
    ) -> torch.Tensor:
        """
        Render a chat prompt and return input_ids on the model's device.

        Extra keyword arguments (e.g. reasoning_effort="medium" for K2-V2,
        enable_thinking=False to hard-disable Qwen3 thinking) are merged with
        self._chat_template_kwargs and passed to apply_chat_template; a
        TypeError fallback silently drops unknown kwargs so the same calling
        code works across all model families.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        template_kw = {**self._chat_template_kwargs, **extra_template_kwargs}
        common = dict(add_generation_prompt=True, return_tensors="pt")
        try:
            ids = self.tokenizer.apply_chat_template(messages, **common, **template_kw)
        except TypeError:
            ids = self.tokenizer.apply_chat_template(messages, **common)
        # apply_chat_template may return a BatchEncoding (multimodal tokenizers
        # like Qwen3.5) rather than a plain tensor — always unwrap to input_ids.
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        return ids.to(self.device)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_hidden_states(
        gen_hs: Tuple,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Reassemble hidden states from model.generate(output_hidden_states=True)
        into the same shape as model(output_hidden_states=True).hidden_states:
        a tuple of (num_layers+1,) tensors, each (batch, full_seq_len, hidden_size).

        With KV-cache (the generate() default):
          gen_hs[step][layer] has shape (batch, step_len, hidden_size)
          step 0  → step_len = prompt_len   (all prompt tokens, prefill)
          step k  → step_len = 1            (one new token per decode step)
        Concatenating across steps gives (batch, prompt_len + T, hidden_size).
        """
        if not gen_hs:
            return ()
        num_layers = len(gen_hs[0])
        assembled: List[torch.Tensor] = []
        for layer_idx in range(num_layers):
            parts = [gen_hs[step][layer_idx] for step in range(len(gen_hs))]
            assembled.append(torch.cat(parts, dim=1).detach())
        return tuple(assembled)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 64,
        capture_hidden: bool = True,
        capture_all_tokens: bool = True,
    ) -> GenerationResult:
        """
        Greedy generation.

        For Qwen3 / Qwen3.5 models reasoning stays enabled (enable_thinking is
        NOT suppressed).  After generation the <think>…</think> prefix is split
        off: gen_token_ids and output_text contain only the answer, while
        thinking_text and answer_offset let you recover the full picture.

        hidden_states indexing (HF standard with output_hidden_states=True):
          [0]   = embedding layer output      (before any decoder layer)
          [L]   = output of decoder layer L-1 (L ≥ 1)
          [-1]  = output of the final decoder layer

        Positions in hidden_states correspond to output_ids (the full sequence
        including any thinking tokens).  Use answer_offset to locate the answer
        span:  answer_positions = range(prompt_len + answer_offset,
                                        prompt_len + answer_offset + len(gen_token_ids))

        capture_all_tokens : bool (default True)
            Inline HS capture during generate() naturally misses the VERY LAST
            generated token (the decode loop emits it and exits before running
            another forward pass on it).  Usually that token is <eos>, but when
            max_new_tokens is reached mid-answer it can be a content token
            (e.g. "fetch" at the end of "loves to play fetch").  When True we
            run ONE additional forward pass on the full output sequence to
            compute the missing HS for the final position, so the returned
            hidden_states tensor has the same length as output_ids.
            Cost: ~1 prefill pass (fast on modern GPUs).
        """
        prompt_ids = prompt_ids.to(self.device)
        # Unwrap BatchEncoding in case caller passed apply_chat_template output directly
        if hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        prompt_len = prompt_ids.shape[1]

        gen_kwargs: dict = dict(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if capture_hidden:
            # Capture hidden states inline during the generation forward passes.
            # This avoids the expensive second model() call that would re-run the
            # entire output sequence through the model just to get hidden states.
            #
            # With KV-cache (default), generate() returns:
            #   hidden_states[step][layer] = (batch, step_len, hidden_size)
            #   step 0  → step_len = prompt_len  (prefill of the full prompt)
            #   step k  → step_len = 1           (one new token per decode step)
            # We reassemble them into one tensor per layer over the full sequence.
            gen_kwargs.update(
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
            raw = self.model.generate(**gen_kwargs)
            out_ids = raw.sequences
            hidden_states = self._assemble_hidden_states(raw.hidden_states)

            # Inline capture misses the last generated token's HS.  When that
            # token carries real content (max_new_tokens reached before EOS),
            # skipping it loses information.  One extra forward pass fixes it.
            hs_len = hidden_states[-1].shape[1] if hidden_states else 0
            full_len = out_ids.shape[1]
            if capture_all_tokens and hs_len < full_len:
                attn = torch.ones_like(out_ids)
                extra = self.model(
                    input_ids=out_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                )
                hidden_states = tuple(
                    torch.cat(
                        [hs, extra.hidden_states[layer_idx][:, hs_len:full_len, :].detach()],
                        dim=1,
                    )
                    for layer_idx, hs in enumerate(hidden_states)
                )
        else:
            out_ids = self.model.generate(**gen_kwargs)
            hidden_states = None

        all_gen_ids = out_ids[0, prompt_len:].tolist()
        answer_ids, answer_offset, output_text, thinking_text = self._post_process(all_gen_ids)

        return GenerationResult(
            prompt_ids=prompt_ids,
            output_ids=out_ids,
            output_text=output_text,
            gen_token_ids=answer_ids,
            hidden_states=hidden_states,
            prompt_len=prompt_len,
            answer_offset=answer_offset,
            thinking_text=thinking_text,
        )

    # ------------------------------------------------------------------
    # SELFIE
    # ------------------------------------------------------------------

    def make_interpretation_prompt(self, num_placeholders: int = 4) -> InterpretationPrompt:
        """
        Plain-text interpretation prompt (no chat template) so SELFIE's
        placeholder substitution works position-by-position.
        Works for all supported model families.
        """
        template = (
            "The following is a message. Summarize it in one sentence.\nMessage: '",
            *([0] * num_placeholders),
            "'\nSummary: The message is about",
        )
        return InterpretationPrompt(self.tokenizer, template)

    def selfie_on_positions(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        positions: List[Tuple[int, int]],   # [(layer, token_idx), ...]
        max_new_tokens: int = 20,
        num_placeholders: int = 4,
    ):
        interp_prompt = self.make_interpretation_prompt(num_placeholders)
        return interpret(
            original_prompt="",
            tokens_to_interpret=positions,
            model=self.model,
            tokenizer=self.tokenizer,
            interpretation_prompt=interp_prompt,
            max_new_tokens=max_new_tokens,
            hidden_states_override=hidden_states,
            input_ids_override=input_ids,
        )

    # ------------------------------------------------------------------
    # Hidden-state injection generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_with_injected_embeds(
        self,
        prompt_ids: torch.Tensor,
        placeholder_positions: List[int],
        injected_hidden: torch.Tensor,          # (num_ph, hidden_size_src)
        inject_layer: int = 0,
        max_new_tokens: int = 128,
        projector: Optional[HiddenStateProjector] = None,
    ) -> str:
        """
        Generate from `prompt_ids` where placeholder token slots in the
        residual stream are overwritten with `injected_hidden`.

        inject_layer=0  → patch the embedding layer output (before layer 0)
        inject_layer=L  → patch the input to decoder layer L  (1-indexed,
                          matches HF hidden_states[L] convention)

        projector : if writer and editor have different hidden_sizes, pass
                    HiddenStateProjector.between(writer_backend, editor_backend)
                    to map the writer's hidden states into the editor's space.

        The returned text has thinking blocks stripped (same as generate()).
        """
        prompt_ids = prompt_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]
        dtype = next(self.model.parameters()).dtype

        # Optional dimension projection (cross-model)
        h = injected_hidden
        if projector is not None:
            h = projector(h)
        injected = h.to(device=self.device, dtype=dtype)

        if injected.shape[0] != len(placeholder_positions):
            raise ValueError(
                f"injected rows {injected.shape[0]} != "
                f"num placeholder positions {len(placeholder_positions)}"
            )
        if injected.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Hidden dim after projection ({injected.shape[-1]}) != "
                f"model hidden_size ({self.hidden_size}). "
                "Use HiddenStateProjector.between(writer, editor)."
            )

        # Pre-build an index tensor on-device so the patch becomes a single
        # vectorized scatter — every placeholder position is overwritten in one
        # tensor op rather than an N-length Python loop.
        ph_idx = torch.as_tensor(placeholder_positions, device=self.device, dtype=torch.long)

        def _patch(hidden: torch.Tensor) -> torch.Tensor:
            # Only patch during the prompt-length prefill, not decode steps
            if hidden.shape[1] >= prompt_len:
                # hidden: (batch, seq_len, H)  →  assign all N injected rows at once
                hidden[:, ph_idx, :] = injected
            return hidden

        handles = []
        if inject_layer == 0:
            embed = self.model.get_input_embeddings()
            handles.append(embed.register_forward_hook(lambda m, i, o: _patch(o)))
        else:
            target_idx = min(inject_layer - 1, self.num_layers - 1)
            layer = self._layers[target_idx]

            def _pre_hook(module, args, kwargs):
                if args and torch.is_tensor(args[0]):
                    return (_patch(args[0]),) + args[1:], kwargs
                if "hidden_states" in kwargs:
                    return args, {**kwargs, "hidden_states": _patch(kwargs["hidden_states"])}
                return args, kwargs

            handles.append(layer.register_forward_pre_hook(_pre_hook, with_kwargs=True))

        # IMPORTANT: pass explicit attention_mask of all ones.  Placeholder
        # positions use pad_token_id; if we let generate() auto-derive the
        # attention mask it would mask them out, and the rest of the prompt
        # would never attend to our injected hidden states (the editor would
        # literally say "the draft is empty").
        attention_mask = torch.ones_like(prompt_ids)

        try:
            out_ids = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            for h in handles:
                h.remove()

        all_gen_ids = out_ids[0, prompt_len:].tolist()
        _, _, output_text, _ = self._post_process(all_gen_ids)
        return output_text

    # ------------------------------------------------------------------
    # Teacher-forced forward pass with hidden-state injection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_with_injected_embeds(
        self,
        full_ids: torch.Tensor,                 # (1, prompt_len + T) — prompt + reference response
        prompt_len: int,
        placeholder_positions: List[int],
        injected_hidden: torch.Tensor,          # (num_ph, hidden_size_src)
        inject_layer: int = 0,
        projector: Optional[HiddenStateProjector] = None,
    ) -> torch.Tensor:                          # returns (T, vocab_size) logits
        """
        Single teacher-forced forward pass over `full_ids = prompt + response`
        with the same placeholder-injection mechanism used by
        generate_with_injected_embeds.

        Returns the next-token logits aligned to each response position:
            logits[t] = p(response_token[t] | prompt, response[:t])

        where T = full_ids.shape[1] - prompt_len.

        This is the workhorse for the Communication Gap metric: it lets you
        score the *same* reference response under two different channels (raw
        text vs injected HS) on matched logits — no autoregressive drift
        between channels.
        """
        full_ids = full_ids.to(self.device)
        if hasattr(full_ids, "input_ids"):
            full_ids = full_ids.input_ids
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)

        dtype = next(self.model.parameters()).dtype

        h = injected_hidden
        if projector is not None:
            h = projector(h)
        injected = h.to(device=self.device, dtype=dtype)

        if injected.shape[0] != len(placeholder_positions):
            raise ValueError(
                f"injected rows {injected.shape[0]} != "
                f"num placeholder positions {len(placeholder_positions)}"
            )
        if injected.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Hidden dim after projection ({injected.shape[-1]}) != "
                f"model hidden_size ({self.hidden_size})."
            )

        ph_idx = torch.as_tensor(placeholder_positions, device=self.device, dtype=torch.long)

        def _patch(hidden: torch.Tensor) -> torch.Tensor:
            # Single forward pass covers the full sequence, so this runs once.
            if hidden.shape[1] >= prompt_len:
                hidden[:, ph_idx, :] = injected
            return hidden

        handles = []
        if inject_layer == 0:
            embed = self.model.get_input_embeddings()
            handles.append(embed.register_forward_hook(lambda m, i, o: _patch(o)))
        else:
            target_idx = min(inject_layer - 1, self.num_layers - 1)
            layer = self._layers[target_idx]

            def _pre_hook(module, args, kwargs):
                if args and torch.is_tensor(args[0]):
                    return (_patch(args[0]),) + args[1:], kwargs
                if "hidden_states" in kwargs:
                    return args, {**kwargs, "hidden_states": _patch(kwargs["hidden_states"])}
                return args, kwargs

            handles.append(layer.register_forward_pre_hook(_pre_hook, with_kwargs=True))

        attention_mask = torch.ones_like(full_ids)
        try:
            out = self.model(input_ids=full_ids, attention_mask=attention_mask)
        finally:
            for h_ in handles:
                h_.remove()

        # logits[:, i, :] predicts token i+1.  To score each response token
        # we take positions [prompt_len - 1, prompt_len, ..., full_len - 2].
        full_len = full_ids.shape[1]
        return out.logits[0, prompt_len - 1 : full_len - 1, :].detach()

    @torch.no_grad()
    def forward_plain(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass with NO injection.  Mirror of
        forward_with_injected_embeds() for the text channel of the CG metric.
        Returns (T, vocab_size) next-token logits aligned to response tokens.
        """
        full_ids = full_ids.to(self.device)
        if hasattr(full_ids, "input_ids"):
            full_ids = full_ids.input_ids
        if full_ids.dim() == 1:
            full_ids = full_ids.unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        out = self.model(input_ids=full_ids, attention_mask=attention_mask)
        full_len = full_ids.shape[1]
        return out.logits[0, prompt_len - 1 : full_len - 1, :].detach()

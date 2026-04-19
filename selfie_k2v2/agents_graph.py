"""
LangGraph probe graph: writer → editor with three comparison arms.

Three-arm comparison
────────────────────
  Arm A  (editor_verdict)
      Editor receives the raw essay text in its prompt.

  Arm B  (editor_selfhs_verdict)
      Editor's OWN hidden states over the draft span (captured in Arm A)
      are injected back at the embedding layer in place of the draft tokens.
      Isolates "how much of the editor's verdict is driven by its own internal
      representation of the draft vs. the surface text tokens."

  Arm C  (editor_writerhs_verdict)
      The WRITER's final-layer hidden states (at its output positions) are
      injected into the editor at the draft placeholder positions.
      Isolates "how much information the writer communicates via its hidden
      states vs. its decoded surface tokens."

Comparing A, B, and C lets you characterise agent communication loss:
  A ≈ B  →  the editor's internal representation of the text carries the same
            information as the surface tokens (no representational bottleneck).
  A ≈ C  →  the writer's final hidden states carry the same info as its text.
  B ≈ C  →  the writer and editor form compatible internal representations.

Pluggable API
─────────────
  make_probe_nodes(writer_backend, editor_backend, **kwargs)
      Returns a dict[str, callable] of node functions.  Drop them into any
      existing StateGraph whose state schema extends SelfieProbeMixin.

  add_probe_to_graph(graph, writer_backend, editor_backend, entry_node)
      Attaches all probe nodes to `graph` as a linear chain after `entry_node`.

  make_graph(writer_backend, editor_backend, **kwargs)
      Builds a complete standalone compiled graph (writer + all probe nodes).

Future metric hook
──────────────────
  AgentState carries 'arm_divergence' (placeholder dict) so a metric can be
  added later without changing the state schema.  Fill it in a post-processing
  node or after graph.invoke().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pandas as pd
import torch
from langgraph.graph import END, StateGraph

from .model_backend import GenerationResult, HiddenStateProjector, ModelBackend


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_WRITER_SYSTEM = (
    "You are a helpful writing assistant. "
    "Write a clear, concise paragraph (3–4 sentences) on the topic provided."
)

_EDITOR_SYSTEM = (
    "You are a critical editor. "
    "Evaluate the draft between <DRAFT> and </DRAFT>. "
    "Give one short sentence of feedback: is it clear? Is anything missing?"
)

_DRAFT_MARKER = "<<<DRAFT_TOKENS>>>"
_INJECT_MARKER = "<<<INJECT_HERE>>>"

_EDITOR_USER_TEXT = (
    f"Please evaluate this draft:\n<DRAFT>{_DRAFT_MARKER}</DRAFT>\nYour feedback:"
)
_EDITOR_USER_INJECT = (
    f"Please evaluate this draft:\n<DRAFT>{_INJECT_MARKER}</DRAFT>\nYour feedback:"
)

# Writer self-probe: inject the writer's own final HS back into the writer and
# ask it to repeat/describe the encoded content.  Tests whether the HS actually
# carry the information of the original output (sanity check before trusting
# cross-agent injection in Arm C).
#
# Prompt format mirrors the SELFIE interpretation template (single-quote frame
# + continuation primer) which is proven to work for hidden-state probing.
# XML-style tags like <MSG>...</MSG> were tried first but caused the model to
# fixate on the framing tokens and emit "<MSG><MSG><MSG>..." loops.
#
# These defaults can be overridden by passing writer_selfprobe_system /
# writer_selfprobe_user through make_probe_nodes / add_probe_to_graph /
# make_graph.  The user-supplied prompt MUST contain exactly one instance of
# `_INJECT_MARKER` (default: "<<<INJECT_HERE>>>") — that location is where the
# writer's hidden states get spliced into the residual stream.
DEFAULT_WRITER_SELFPROBE_SYSTEM = "You are a helpful assistant."
DEFAULT_WRITER_SELFPROBE_USER = (
    f"The following is a message: '{_INJECT_MARKER}'.\n"
    "Repeated word-for-word, the message says:"
)
# Backwards-compat aliases (internal use only)
_WRITER_SELFPROBE_SYSTEM = DEFAULT_WRITER_SELFPROBE_SYSTEM
_WRITER_SELFPROBE_USER = DEFAULT_WRITER_SELFPROBE_USER


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class SelfieProbeMixin(TypedDict, total=False):
    """
    All keys written/read by the probe nodes.  Extend your own TypedDict with
    this to make it compatible with make_probe_nodes():

        class MyState(SelfieProbeMixin):
            my_extra_field: str
    """
    # ── writer outputs (must be set before probe nodes run) ──────────────────
    writer_result: GenerationResult
    writer_output_text: str
    writer_output_token_ids: List[int]

    # ── editor shared ────────────────────────────────────────────────────────
    editor_prompt_ids: torch.Tensor
    editor_draft_start: int         # index of first draft token in editor prompt
    editor_draft_end: int           # exclusive

    # ── Arm A: raw text → editor ─────────────────────────────────────────────
    editor_result: GenerationResult
    editor_verdict: str

    # ── Arm B: editor's own hidden states re-injected ────────────────────────
    editor_selfhs_verdict: str

    # ── Arm C: writer's hidden states injected into editor ───────────────────
    editor_writerhs_verdict: str

    # ── Writer self-probe: writer's HS injected back into writer ─────────────
    writer_selfprobe_output: str

    # ── SELFIE analyses ───────────────────────────────────────────────────────
    selfie_writer: Optional[pd.DataFrame]           # Exp 1
    selfie_editor_on_draft: Optional[pd.DataFrame]  # Exp 3

    # ── Future divergence metric (placeholder) ───────────────────────────────
    # Populated externally once a metric is chosen.
    arm_divergence: Optional[Dict[str, Any]]


class AgentState(SelfieProbeMixin, total=False):
    """Full state for the standalone make_graph() pipeline."""
    task: str
    writer_prompt_ids: torch.Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_sublist(haystack: List[int], needle: List[int]) -> int:
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i: i + n] == needle:
            return i
    return -1


def _probe_layers(num_layers: int, count: int = 2) -> List[int]:
    """
    Pick `count` representative probe layers, evenly spread from mid to late.
    count=1 → just a single late layer (fastest)
    count=2 → mid + late (default, balanced)
    count=3 → early-mid + mid + late
    """
    count = max(1, count)
    if num_layers < 4:
        return [max(1, num_layers - 1)]
    # Evenly spaced between num_layers//2 and num_layers-1
    late = num_layers - 1
    if count == 1:
        return [(3 * num_layers) // 4]
    mid = num_layers // 2
    if count == 2:
        return [mid, (3 * num_layers) // 4]
    # count >= 3
    step = max(1, (late - mid) // (count - 1))
    return [mid + step * i for i in range(count - 1)] + [late]


def _sample_positions(positions: List[int], max_n: int) -> List[int]:
    """Evenly subsample `positions` to at most `max_n` entries."""
    if len(positions) <= max_n:
        return positions
    step = len(positions) / max_n
    return [positions[int(i * step)] for i in range(max_n)]


def _split_prompt_at_marker(
    backend: ModelBackend,
    system: str,
    user: str,
    marker: str,
) -> Tuple[List[int], List[int]]:
    """
    Render the chat prompt as plain text (tokenize=False), split at ``marker``,
    then tokenize each half independently.

    Returns (pre_ids, post_ids) so that::

        spliced = pre_ids + <content_ids> + post_ids

    gives a coherent prompt sequence without ever needing _find_sublist.

    This sidesteps the BPE tokenization-context problem where ``marker`` can
    tokenise differently when embedded in the full string vs. when encoded in
    isolation (which caused the old _find_sublist approach to fail on Qwen3.5).
    """
    tok = backend.tokenizer
    messages: List[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    template_kw = {**backend._chat_template_kwargs}
    common: dict = dict(add_generation_prompt=True, tokenize=False)
    try:
        full_text: str = tok.apply_chat_template(messages, **common, **template_kw)
    except TypeError:
        full_text = tok.apply_chat_template(messages, **common)

    if marker not in full_text:
        raise RuntimeError(
            f"Marker {marker!r} absent from rendered prompt text. "
            "The chat template may be escaping or altering the marker string."
        )

    pre_text, post_text = full_text.split(marker, 1)

    # Tokenize each half as a plain string; the special tokens are already
    # embedded as text by apply_chat_template, so add_special_tokens=False.
    # Multimodal processors can return nested lists or tensors — normalize.
    def _tokenize_flat(text: str) -> List[int]:
        out = tok(text, add_special_tokens=False)
        ids = out["input_ids"]
        # Handle: tensor → list, nested list [[...]] → [...], list → list
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    pre_ids = _tokenize_flat(pre_text)
    post_ids = _tokenize_flat(post_text)

    # CPU-side vocab range check — catches out-of-range IDs before they hit
    # the GPU embedding lookup (which would produce an opaque async CUDA assert).
    vocab_size = backend.model.get_input_embeddings().num_embeddings
    for label, ids in (("pre", pre_ids), ("post", post_ids)):
        for tid in ids:
            if tid < 0 or tid >= vocab_size:
                raise RuntimeError(
                    f"Token id {tid} out of range [0, {vocab_size}) in {label}_ids. "
                    f"Tokenizer/embedding mismatch for {type(tok).__name__}."
                )
    return pre_ids, post_ids



def _build_inject_prompt(
    backend: ModelBackend,
    marker: str,
    num_placeholders: int,
    system: str = _EDITOR_SYSTEM,
    user_template: str = _EDITOR_USER_INJECT,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build an editor prompt where `marker` is replaced by `num_placeholders`
    dummy tokens.  Returns (prompt_ids tensor, list of placeholder positions).
    """
    pre_ids, post_ids = _split_prompt_at_marker(
        backend, system, user_template, marker
    )

    tok = backend.tokenizer
    ph_id = tok.pad_token_id if tok.pad_token_id is not None else tok.bos_token_id
    if ph_id is None:
        ph_id = 0

    marker_pos = len(pre_ids)
    spliced = pre_ids + [ph_id] * num_placeholders + post_ids
    ph_positions = list(range(marker_pos, marker_pos + num_placeholders))
    prompt_ids = torch.tensor([spliced], dtype=torch.long, device=backend.device)
    return prompt_ids, ph_positions


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_probe_nodes(
    writer_backend: ModelBackend,
    editor_backend: ModelBackend,
    max_writer_tokens: int = 40,
    max_editor_tokens: int = 60,
    selfie_max_new_tokens: int = 15,
    inject_layer: int = 0,
    run_selfie: bool = True,
    run_self_probe: bool = True,
    run_arm_b: bool = True,
    selfie_max_positions: int = 8,
    selfie_num_layers: int = 2,
    writer_selfprobe_system: Optional[str] = None,
    writer_selfprobe_user: Optional[str] = None,
    verbose_timing: bool = False,
) -> Dict[str, Any]:
    """
    Build all probe node functions and return them as a dict.

    The returned node names are:
      "probe_editor"          – Arm A (raw text → editor, also captures HS)
      "probe_editor_selfhs"   – Arm B (editor's own HS re-injected)
      "probe_editor_writerhs" – Arm C (writer's HS injected)
      "probe_selfie_writer"   – SELFIE on writer's hidden states
      "probe_selfie_editor"   – SELFIE on editor's hidden states at draft span

    State keys READ by the probe nodes (must already exist):
      writer_result, writer_output_text, writer_output_token_ids

    State keys WRITTEN by the probe nodes:
      editor_prompt_ids, editor_draft_start, editor_draft_end,
      editor_result, editor_verdict,
      editor_selfhs_verdict, editor_writerhs_verdict,
      selfie_writer, selfie_editor_on_draft, arm_divergence (None placeholder)

    Parameters
    ----------
    writer_backend, editor_backend : ModelBackend
        Can be the same instance (same model for both agents) or different
        instances (different models/sizes).  HiddenStateProjector handles
        hidden-size mismatches automatically.
    inject_layer : int
        Layer at which to inject hidden states.  0 = embedding layer output
        (most analogous to replacing the token embeddings entirely).
        Set to a mid-layer index to inject deeper representations.
    writer_selfprobe_system, writer_selfprobe_user : Optional[str]
        Override the default writer self-probe system/user prompts.  The user
        string MUST contain exactly one occurrence of the placeholder marker
        (default "<<<INJECT_HERE>>>" — see _INJECT_MARKER); that location is
        where the writer's final hidden states get spliced in.  Pass None to
        use DEFAULT_WRITER_SELFPROBE_SYSTEM / DEFAULT_WRITER_SELFPROBE_USER.

    Speed controls
    --------------
    run_selfie : bool (default True)
        Run SELFIE analysis on writer+editor HS.  Adds 2 * selfie_num_layers
        generate() calls.  Set False for pure 3-arm comparison (skips analytics).
    run_self_probe : bool (default True)
        Run writer self-probe (inject writer's HS back into writer + ask to
        repeat).  Adds 1 generate() call.  Set False to skip entirely.
    run_arm_b : bool (default True)
        Run Arm B (editor's own HS re-injected).  Adds 1 generate() call.  Set
        False if you only care about the A↔C comparison.
    selfie_num_layers : int (default 2)
        Number of SELFIE probe layers.  Each layer costs one generate() call.
        Set 1 for fastest SELFIE.
    selfie_max_positions : int (default 8)
        Max token positions to probe per layer in SELFIE.  Drives batch size,
        not call count.
    verbose_timing : bool (default False)
        Print wall-clock time for each probe node so you can see where the
        time goes.

    Minimum work per call (generate()-count reference):
      writer (entry)              : 1
      probe_editor (Arm A)        : 1  (captures HS)
      probe_editor_selfhs (Arm B) : 1  (skipped if run_arm_b=False)
      probe_editor_writerhs (C)   : 1
      probe_writer_selfprobe      : 1  (skipped if run_self_probe=False)
      probe_selfie_writer         : selfie_num_layers  (skipped if run_selfie=False)
      probe_selfie_editor         : selfie_num_layers  (skipped if run_selfie=False)

    Lean mode (run_selfie=False, run_self_probe=False, run_arm_b=False):
        writer + Arm A + Arm C = 3 generate() calls.
    """
    projector = HiddenStateProjector.between(writer_backend, editor_backend)

    # Timer wrapper: lets callers see where the time goes without per-node
    # instrumentation in user code.  Nodes that short-circuit (run_* = False)
    # still print so the skip is visible in the trace.
    import time as _time

    def _timed(name: str, fn):
        if not verbose_timing:
            return fn

        def wrapped(state):
            t0 = _time.time()
            out = fn(state)
            dt = _time.time() - t0
            print(f"  [{name:<24s}] {dt:6.2f}s")
            return out

        return wrapped

    selfprobe_system = (
        writer_selfprobe_system
        if writer_selfprobe_system is not None
        else DEFAULT_WRITER_SELFPROBE_SYSTEM
    )
    selfprobe_user = (
        writer_selfprobe_user
        if writer_selfprobe_user is not None
        else DEFAULT_WRITER_SELFPROBE_USER
    )
    if _INJECT_MARKER not in selfprobe_user:
        raise ValueError(
            f"writer_selfprobe_user must contain the marker {_INJECT_MARKER!r} "
            "exactly once — that location is where the writer's hidden states "
            "are spliced into the prompt."
        )

    # ── Arm A: editor with raw text ──────────────────────────────────────────

    def probe_editor(state: SelfieProbeMixin) -> SelfieProbeMixin:
        device = editor_backend.device

        # Build the editor prompt by splitting at the draft marker in plain text
        # and splicing writer token IDs in between.  Avoids the BPE context
        # problem that caused _find_sublist to fail on Qwen3.5.
        pre_ids, post_ids = _split_prompt_at_marker(
            editor_backend, _EDITOR_SYSTEM, _EDITOR_USER_TEXT, _DRAFT_MARKER
        )

        writer_toks = state["writer_output_token_ids"]

        # If writer and editor have different tokenizers, the writer's token IDs
        # may be out of range for the editor's embedding table.  Detect this and
        # fall back to decoding the writer's answer and re-tokenizing with the
        # editor's tokenizer (round-trip through text).
        editor_vocab = editor_backend.model.get_input_embeddings().num_embeddings
        if writer_backend is not editor_backend and any(
            t < 0 or t >= editor_vocab for t in writer_toks
        ):
            writer_text = state.get("writer_output_text") or \
                          writer_backend.tokenizer.decode(writer_toks, skip_special_tokens=True)
            writer_toks = editor_backend.tokenizer(
                writer_text, add_special_tokens=False
            )["input_ids"]
            if hasattr(writer_toks, "tolist"):
                writer_toks = writer_toks.tolist()
            if writer_toks and isinstance(writer_toks[0], list):
                writer_toks = writer_toks[0]
            writer_toks = [int(x) for x in writer_toks]

        spliced = pre_ids + writer_toks + post_ids
        draft_start = len(pre_ids)
        draft_end = draft_start + len(writer_toks)

        editor_prompt_ids = torch.tensor([spliced], dtype=torch.long, device=device)

        result = editor_backend.generate(
            prompt_ids=editor_prompt_ids,
            max_new_tokens=max_editor_tokens,
            capture_hidden=True,
        )

        return {
            **state,
            "editor_prompt_ids": editor_prompt_ids,
            "editor_draft_start": draft_start,
            "editor_draft_end": draft_end,
            "editor_result": result,
            "editor_verdict": result.output_text,
            "arm_divergence": None,   # placeholder for future metric
        }

    # ── Arm B: editor's own hidden states re-injected ────────────────────────

    def probe_editor_selfhs(state: SelfieProbeMixin) -> SelfieProbeMixin:
        if not run_arm_b:
            return {**state, "editor_selfhs_verdict": "(skipped: run_arm_b=False)"}
        er = state["editor_result"]
        # draft_start/end were set from answer-only token IDs, so they point
        # directly at answer positions inside editor's output_ids.  Cap to
        # hs_len because the final EOS token has no hidden state captured by
        # inline generate() (loop exits before its forward pass).
        hs_len = er.hidden_states[-1].shape[1]
        draft_positions = [t for t in range(state["editor_draft_start"], state["editor_draft_end"])
                           if t < hs_len]
        num_ph = len(draft_positions)

        # Extract the editor's FINAL hidden state at the draft span.
        # final layer = hidden_states[-1] (output of the last decoder layer).
        final_hs = er.hidden_states[-1][0, draft_positions, :]  # (num_ph, hidden_size)

        prompt_ids, ph_positions = _build_inject_prompt(
            editor_backend, _INJECT_MARKER, num_ph,
        )

        verdict = editor_backend.generate_with_injected_embeds(
            prompt_ids=prompt_ids,
            placeholder_positions=ph_positions,
            injected_hidden=final_hs,
            inject_layer=inject_layer,
            max_new_tokens=max_editor_tokens,
            projector=None,  # same model, no projection needed
        )
        return {**state, "editor_selfhs_verdict": verdict}

    # ── Arm C: writer's hidden states injected into editor ───────────────────

    def probe_editor_writerhs(state: SelfieProbeMixin) -> SelfieProbeMixin:
        wr = state["writer_result"]
        # Positions of the answer tokens (skip any thinking prefix).  Cap to
        # hs_len because the final EOS token has no hidden state captured by
        # inline generate() (loop exits before its forward pass).
        hs_len = wr.hidden_states[-1].shape[1]
        ans_start = wr.prompt_len + wr.answer_offset
        writer_out_positions = [t for t in range(ans_start, ans_start + len(wr.gen_token_ids))
                                if t < hs_len]
        num_ph = len(writer_out_positions)

        # Writer's FINAL hidden state at its output positions.
        writer_final_hs = wr.hidden_states[-1][0, writer_out_positions, :]  # (num_ph, src_hs)

        prompt_ids, ph_positions = _build_inject_prompt(
            editor_backend, _INJECT_MARKER, num_ph,
        )

        verdict = editor_backend.generate_with_injected_embeds(
            prompt_ids=prompt_ids,
            placeholder_positions=ph_positions,
            injected_hidden=writer_final_hs,
            inject_layer=inject_layer,
            max_new_tokens=max_editor_tokens,
            projector=projector if projector.needs_projection else None,
        )
        return {**state, "editor_writerhs_verdict": verdict}

    # ── Writer self-probe: writer's HS injected back into writer ─────────────
    # Tests whether the writer can decode its OWN final hidden states back into
    # the original output.  If the self-probe reconstructs the essay well, the
    # HS genuinely encode the content (and any divergence between Arm A and
    # Arm C is due to cross-agent communication, not an encoding limitation).

    def probe_writer_selfprobe(state: SelfieProbeMixin) -> SelfieProbeMixin:
        if not run_self_probe:
            return {**state, "writer_selfprobe_output": "(skipped: run_self_probe=False)"}
        wr = state["writer_result"]
        hs_len = wr.hidden_states[-1].shape[1]
        ans_start = wr.prompt_len + wr.answer_offset
        writer_out_positions = [t for t in range(ans_start, ans_start + len(wr.gen_token_ids))
                                if t < hs_len]
        num_ph = len(writer_out_positions)

        writer_final_hs = wr.hidden_states[-1][0, writer_out_positions, :]

        prompt_ids, ph_positions = _build_inject_prompt(
            writer_backend,
            _INJECT_MARKER,
            num_ph,
            system=selfprobe_system,
            user_template=selfprobe_user,
        )

        output = writer_backend.generate_with_injected_embeds(
            prompt_ids=prompt_ids,
            placeholder_positions=ph_positions,
            injected_hidden=writer_final_hs,
            inject_layer=inject_layer,
            max_new_tokens=max_writer_tokens,
            projector=None,  # writer → writer, same space
        )
        return {**state, "writer_selfprobe_output": output}

    # ── SELFIE on writer's hidden states ─────────────────────────────────────

    def probe_selfie_writer(state: SelfieProbeMixin) -> SelfieProbeMixin:
        if not run_selfie:
            return {**state, "selfie_writer": pd.DataFrame(
                columns=["layer", "token_idx", "token", "interpretation"]
            )}
        wr = state["writer_result"]
        # Positions of the answer tokens inside output_ids (skip any thinking prefix).
        # Cap to hs.shape[1]-1 because the final EOS token has no hidden state
        # recorded by the inline generate() capture (loop exits before its fwd pass).
        hs_len = wr.hidden_states[0].shape[1]
        ans_start = wr.prompt_len + wr.answer_offset
        raw_positions = [t for t in range(ans_start, ans_start + len(wr.gen_token_ids))
                         if t < hs_len]
        out_positions = _sample_positions(raw_positions, selfie_max_positions)
        probe_layers = _probe_layers(writer_backend.num_layers, count=selfie_num_layers)
        positions = [(L, t) for L in probe_layers for t in out_positions]

        df = writer_backend.selfie_on_positions(
            hidden_states=wr.hidden_states,
            input_ids=wr.output_ids,
            positions=positions,
            max_new_tokens=selfie_max_new_tokens,
        )
        return {**state, "selfie_writer": df}

    # ── SELFIE on editor's hidden states at the draft span ───────────────────

    def probe_selfie_editor(state: SelfieProbeMixin) -> SelfieProbeMixin:
        if not run_selfie:
            return {**state, "selfie_editor_on_draft": pd.DataFrame(
                columns=["layer", "token_idx", "token", "interpretation"]
            )}
        er = state["editor_result"]
        hs_len = er.hidden_states[0].shape[1]
        # draft_start/end align with answer-only token positions in output_ids
        raw_positions = [t for t in range(state["editor_draft_start"], state["editor_draft_end"])
                         if t < hs_len]
        draft_positions = _sample_positions(raw_positions, selfie_max_positions)
        probe_layers = _probe_layers(editor_backend.num_layers, count=selfie_num_layers)
        positions = [(L, t) for L in probe_layers for t in draft_positions]

        df = editor_backend.selfie_on_positions(
            hidden_states=er.hidden_states,
            input_ids=er.output_ids,
            positions=positions,
            max_new_tokens=selfie_max_new_tokens,
        )
        return {**state, "selfie_editor_on_draft": df}

    return {
        "probe_editor":            _timed("probe_editor",           probe_editor),
        "probe_editor_selfhs":     _timed("probe_editor_selfhs",    probe_editor_selfhs),
        "probe_editor_writerhs":   _timed("probe_editor_writerhs",  probe_editor_writerhs),
        "probe_writer_selfprobe":  _timed("probe_writer_selfprobe", probe_writer_selfprobe),
        "probe_selfie_writer":     _timed("probe_selfie_writer",    probe_selfie_writer),
        "probe_selfie_editor":     _timed("probe_selfie_editor",    probe_selfie_editor),
    }


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def add_probe_to_graph(
    graph: StateGraph,
    writer_backend: ModelBackend,
    editor_backend: ModelBackend,
    entry_node: str,
    max_writer_tokens: int = 40,
    max_editor_tokens: int = 60,
    selfie_max_new_tokens: int = 15,
    inject_layer: int = 0,
    run_selfie: bool = True,
    run_self_probe: bool = True,
    run_arm_b: bool = True,
    selfie_max_positions: int = 8,
    selfie_num_layers: int = 2,
    writer_selfprobe_system: Optional[str] = None,
    writer_selfprobe_user: Optional[str] = None,
    verbose_timing: bool = False,
) -> StateGraph:
    """
    Attach all probe nodes to `graph` as a linear chain starting from
    `entry_node`.  The final probe node edges to END.

    Usage in an existing workflow
    ─────────────────────────────
        g = StateGraph(MyState)
        g.add_node("my_writer", my_writer_fn)   # populates writer_result etc.
        g.set_entry_point("my_writer")
        add_probe_to_graph(g, writer_backend, editor_backend, entry_node="my_writer")
        app = g.compile()

    The graph topology added:
        entry_node
          → probe_editor            (Arm A — raw text, captures HS)
          → probe_selfie_writer
          → probe_selfie_editor
          → probe_editor_selfhs     (Arm B — editor's own HS re-injected)
          → probe_editor_writerhs   (Arm C — writer's HS injected into editor)
          → probe_writer_selfprobe  (writer's HS injected back into writer)
          → END
    """
    nodes = make_probe_nodes(
        writer_backend,
        editor_backend,
        max_writer_tokens=max_writer_tokens,
        max_editor_tokens=max_editor_tokens,
        selfie_max_new_tokens=selfie_max_new_tokens,
        inject_layer=inject_layer,
        run_selfie=run_selfie,
        run_self_probe=run_self_probe,
        run_arm_b=run_arm_b,
        selfie_max_positions=selfie_max_positions,
        selfie_num_layers=selfie_num_layers,
        writer_selfprobe_system=writer_selfprobe_system,
        writer_selfprobe_user=writer_selfprobe_user,
        verbose_timing=verbose_timing,
    )

    order = [
        "probe_editor",
        "probe_selfie_writer",
        "probe_selfie_editor",
        "probe_editor_selfhs",
        "probe_editor_writerhs",
        "probe_writer_selfprobe",
    ]

    for name in order:
        graph.add_node(name, nodes[name])

    graph.add_edge(entry_node, "probe_editor")
    for a, b in zip(order, order[1:]):
        graph.add_edge(a, b)
    graph.add_edge(order[-1], END)

    return graph


def make_graph(
    writer_backend: ModelBackend,
    editor_backend: ModelBackend,
    max_writer_tokens: int = 40,
    max_editor_tokens: int = 60,
    selfie_max_new_tokens: int = 15,
    inject_layer: int = 0,
    run_selfie: bool = False,
    run_self_probe: bool = True,
    run_arm_b: bool = True,
    selfie_max_positions: int = 8,
    selfie_num_layers: int = 2,
    writer_selfprobe_system: Optional[str] = None,
    writer_selfprobe_user: Optional[str] = None,
    verbose_timing: bool = False,
):
    """
    Build and compile a complete standalone graph:
        writer → probe_editor (A) → probe_selfie_writer
                                  → probe_selfie_editor
                                  → probe_editor_selfhs (B)
                                  → probe_editor_writerhs (C) → END

    Parameters
    ----------
    writer_backend, editor_backend : ModelBackend
        Pass the same instance for same-model writer/editor, or different
        ModelBackend instances for cross-model experiments.

    Returns
    -------
    Compiled LangGraph app.  Call with:
        final = app.invoke({"task": "your task here"})
    """
    g = StateGraph(AgentState)

    import time as _time

    # ── writer node ───────────────────────────────────────────────────────────
    def writer_node(state: AgentState) -> AgentState:
        t0 = _time.time()
        prompt_ids = writer_backend.build_chat_prompt(
            system=_WRITER_SYSTEM,
            user=state["task"],
        )
        result = writer_backend.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_writer_tokens,
            capture_hidden=True,
        )
        if verbose_timing:
            print(f"  [{'writer':<24s}] {_time.time() - t0:6.2f}s")
        return {
            **state,
            "writer_prompt_ids": prompt_ids,
            "writer_result": result,
            "writer_output_text": result.output_text,
            "writer_output_token_ids": result.gen_token_ids,
        }

    g.add_node("writer", writer_node)
    g.set_entry_point("writer")

    add_probe_to_graph(
        g,
        writer_backend,
        editor_backend,
        entry_node="writer",
        max_writer_tokens=max_writer_tokens,
        max_editor_tokens=max_editor_tokens,
        selfie_max_new_tokens=selfie_max_new_tokens,
        inject_layer=inject_layer,
        run_selfie=run_selfie,
        run_self_probe=run_self_probe,
        run_arm_b=run_arm_b,
        selfie_max_positions=selfie_max_positions,
        selfie_num_layers=selfie_num_layers,
        writer_selfprobe_system=writer_selfprobe_system,
        writer_selfprobe_user=writer_selfprobe_user,
        verbose_timing=verbose_timing,
    )

    return g.compile()

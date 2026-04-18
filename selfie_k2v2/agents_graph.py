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


def _probe_layers(num_layers: int) -> List[int]:
    """Pick three representative probe layers (early-mid, mid, late)."""
    if num_layers >= 60:
        return [num_layers // 4, num_layers // 2, (3 * num_layers) // 4]
    if num_layers >= 20:
        return [num_layers // 3, num_layers // 2, (2 * num_layers) // 3]
    return [max(1, num_layers // 2)]


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
    pre_ids: List[int] = tok(pre_text, add_special_tokens=False)["input_ids"]
    post_ids: List[int] = tok(post_text, add_special_tokens=False)["input_ids"]
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
    """
    projector = HiddenStateProjector.between(writer_backend, editor_backend)

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
        er = state["editor_result"]
        # draft_start/end were set from answer-only token IDs, so they point
        # directly at answer positions inside editor's output_ids.
        draft_positions = list(range(state["editor_draft_start"], state["editor_draft_end"]))
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
        # Positions of the answer tokens (skip any thinking prefix)
        ans_start = wr.prompt_len + wr.answer_offset
        writer_out_positions = list(range(ans_start, ans_start + len(wr.gen_token_ids)))
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

    # ── SELFIE on writer's hidden states ─────────────────────────────────────

    def probe_selfie_writer(state: SelfieProbeMixin) -> SelfieProbeMixin:
        wr = state["writer_result"]
        # Positions of the answer tokens inside output_ids (skip any thinking prefix)
        ans_start = wr.prompt_len + wr.answer_offset
        out_positions = list(range(ans_start, ans_start + len(wr.gen_token_ids)))
        probe_layers = _probe_layers(writer_backend.num_layers)
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
        er = state["editor_result"]
        # draft_start/end align with answer-only token positions in output_ids
        draft_positions = list(range(state["editor_draft_start"], state["editor_draft_end"]))
        probe_layers = _probe_layers(editor_backend.num_layers)
        positions = [(L, t) for L in probe_layers for t in draft_positions]

        df = editor_backend.selfie_on_positions(
            hidden_states=er.hidden_states,
            input_ids=er.output_ids,
            positions=positions,
            max_new_tokens=selfie_max_new_tokens,
        )
        return {**state, "selfie_editor_on_draft": df}

    return {
        "probe_editor":          probe_editor,
        "probe_editor_selfhs":   probe_editor_selfhs,
        "probe_editor_writerhs": probe_editor_writerhs,
        "probe_selfie_writer":   probe_selfie_writer,
        "probe_selfie_editor":   probe_selfie_editor,
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
          → probe_editor       (Arm A — raw text, captures HS)
          → probe_selfie_writer
          → probe_selfie_editor
          → probe_editor_selfhs   (Arm B — editor's own HS re-injected)
          → probe_editor_writerhs (Arm C — writer's HS injected)
          → END
    """
    nodes = make_probe_nodes(
        writer_backend,
        editor_backend,
        max_writer_tokens=max_writer_tokens,
        max_editor_tokens=max_editor_tokens,
        selfie_max_new_tokens=selfie_max_new_tokens,
        inject_layer=inject_layer,
    )

    order = [
        "probe_editor",
        "probe_selfie_writer",
        "probe_selfie_editor",
        "probe_editor_selfhs",
        "probe_editor_writerhs",
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

    # ── writer node ───────────────────────────────────────────────────────────
    def writer_node(state: AgentState) -> AgentState:
        prompt_ids = writer_backend.build_chat_prompt(
            system=_WRITER_SYSTEM,
            user=state["task"],
        )
        result = writer_backend.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_writer_tokens,
            capture_hidden=True,
        )
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
    )

    return g.compile()

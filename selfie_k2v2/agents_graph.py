"""
LangGraph probe graph: writer → editor with three comparison arms.

Three-arm comparison
────────────────────
  Arm A  (editor_verdict)
      Editor receives the raw essay text in its prompt.

  Arm B  (editor_selfhs_verdict)
      Editor's OWN hidden states over the draft span (captured in Arm A)
      are injected back at `inject_layer` in place of the draft tokens.
      Isolates "how much of the editor's verdict is driven by its own internal
      representation of the draft vs. the surface text tokens."

  Arm C  (editor_writerhs_verdict)
      The WRITER's final-layer hidden states (at its answer-token positions)
      are injected into the editor at the draft placeholder positions.
      Isolates "how much information the writer communicates via its hidden
      states vs. its decoded surface tokens."

  Arm D  (writer_selfhs_output)
      Writer self-probe: the WRITER's final-layer hidden states (full-sentence
      answer) are injected back into the WRITER itself with a user-configurable
      prompt (default: "repeat/describe this message").  Sanity check that the
      HS actually encode the essay content — if Arm D fails to produce coherent
      text, Arm C (cross-agent) is extremely unlikely to work either.

Comparing A, B, C, D lets you characterise agent communication loss:
  A ≈ B   →  the editor's internal representation of the text carries the same
             information as the surface tokens (no representational bottleneck).
  A ≈ C   →  the writer's final hidden states carry the same info as its text.
  B ≈ C   →  the writer and editor form compatible internal representations.
  D ≈ ans →  writer's HS genuinely encode the essay content (foundation for C).

Total cost per invoke(): 5 generate() calls — writer + A + B + C + D.

Pluggable API
─────────────
  make_probe_nodes(writer_backend, editor_backend, **kwargs)
      Returns a dict[str, callable] of the 3 probe nodes.  Drop them into any
      existing StateGraph whose state schema extends ProbeMixin.

  add_probe_to_graph(graph, writer_backend, editor_backend, entry_node)
      Attaches the 3 probe nodes to `graph` as a linear chain after `entry_node`.

  make_graph(writer_backend, editor_backend, **kwargs)
      Builds a complete standalone compiled graph (writer + 3 probe nodes).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from langgraph.graph import END, StateGraph

from .metrics import communication_gap
from .model_backend import GenerationResult, HiddenStateProjector, ModelBackend


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
#
# Every prompt below is overridable at graph-build time:
#
#   make_graph(..., writer_system=..., editor_system=...,
#              writer_selfprobe_system=..., writer_selfprobe_user=...)
#
# The editor's *user* prompts are not user-configurable because they embed
# special markers ("<<<DRAFT_TOKENS>>>", "<<<INJECT_HERE>>>") that are load-
# bearing for the token-splicing logic.

DEFAULT_WRITER_SYSTEM = (
    "You are a helpful writing assistant. "
    "Write a clear, concise paragraph (3–4 sentences) on the topic provided."
)

DEFAULT_EDITOR_SYSTEM = (
    "You are a critical editor. "
    "Evaluate the draft between <DRAFT> and </DRAFT>. "
    "Give one short sentence of feedback: is it clear? Is anything missing?"
)

# Internal aliases
_WRITER_SYSTEM = DEFAULT_WRITER_SYSTEM
_EDITOR_SYSTEM = DEFAULT_EDITOR_SYSTEM

_DRAFT_MARKER = "<<<DRAFT_TOKENS>>>"
_INJECT_MARKER = "<<<INJECT_HERE>>>"

_EDITOR_USER_TEXT = (
    f"Please evaluate this draft:\n<DRAFT>{_DRAFT_MARKER}</DRAFT>\nYour feedback:"
)
_EDITOR_USER_INJECT = (
    f"Please evaluate this draft:\n<DRAFT>{_INJECT_MARKER}</DRAFT>\nYour feedback:"
)

# ── Writer self-probe (Arm D) ────────────────────────────────────────────────
# The writer's own final-layer HS (over its full answer) are injected back into
# the writer.  The user-prompt below MUST contain exactly one `_INJECT_MARKER`
# — that's where the HS get spliced into the residual stream.
#
# Prompt format mirrors the SELFIE interpretation template (single-quote frame
# + continuation primer), which is proven to work for hidden-state probing.
# XML-style tags like <MSG>...</MSG> were tried first but caused the model to
# fixate on the framing tokens and emit "<MSG><MSG><MSG>..." loops.
DEFAULT_WRITER_SELFPROBE_SYSTEM = "You are a helpful assistant."
DEFAULT_WRITER_SELFPROBE_USER = (
    f"The following is a message: '{_INJECT_MARKER}'.\n"
    "Repeated word-for-word, the message says:"
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ProbeMixin(TypedDict, total=False):
    """
    All keys written/read by the probe nodes.  Extend your own TypedDict with
    this to make it compatible with make_probe_nodes():

        class MyState(ProbeMixin):
            my_extra_field: str
    """
    # writer outputs (must be set before probe nodes run)
    writer_result: GenerationResult
    writer_output_text: str
    writer_output_token_ids: List[int]

    # editor shared
    editor_prompt_ids: torch.Tensor
    editor_draft_start: int         # index of first draft token in editor prompt
    editor_draft_end: int           # exclusive

    # Arm A
    editor_result: GenerationResult
    editor_verdict: str

    # Arm B
    editor_selfhs_verdict: str

    # Arm C
    editor_writerhs_verdict: str

    # Arm D: writer's own HS re-injected into the writer (self-probe)
    writer_selfhs_output: str

    # Communication Gap metric (text channel vs. latent channel, teacher-forced)
    comm_gap: Dict[str, Any]


# Backwards-compat alias so existing imports keep working
SelfieProbeMixin = ProbeMixin


class AgentState(ProbeMixin, total=False):
    """Full state for the standalone make_graph() pipeline."""
    task: str
    writer_prompt_ids: torch.Tensor


# ---------------------------------------------------------------------------
# Prompt-splicing helpers
# ---------------------------------------------------------------------------

def _split_prompt_at_marker(
    backend: ModelBackend,
    system: str,
    user: str,
    marker: str,
) -> Tuple[List[int], List[int]]:
    """
    Render the chat prompt as plain text, split at ``marker``, then tokenize
    each half independently.  Returns (pre_ids, post_ids) so that::

        spliced = pre_ids + <content_ids> + post_ids

    gives a coherent prompt sequence.  Sidesteps BPE tokenization-context
    quirks (markers can tokenize differently when embedded in full text vs.
    encoded in isolation).
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

    def _tokenize_flat(text: str) -> List[int]:
        out = tok(text, add_special_tokens=False)
        ids = out["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    pre_ids = _tokenize_flat(pre_text)
    post_ids = _tokenize_flat(post_text)

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
    num_placeholders: int,
    system: str = _EDITOR_SYSTEM,
    user: str = _EDITOR_USER_INJECT,
    marker: str = _INJECT_MARKER,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Build a prompt where ``marker`` (inside ``user``) is replaced by
    ``num_placeholders`` dummy tokens.  Returns (prompt_ids, placeholder_positions).
    Used by Arm B / Arm C (editor prompts) and Arm D (writer self-probe prompt).
    """
    pre_ids, post_ids = _split_prompt_at_marker(backend, system, user, marker)
    tok = backend.tokenizer
    ph_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.bos_token_id or 0)

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
    max_editor_tokens: int = 60,
    max_writer_selfprobe_tokens: int = 80,
    inject_layer: int = 0,
    editor_system: Optional[str] = None,
    editor_user_text: Optional[str] = None,
    editor_inject_user_text: Optional[str] = None,
    writer_selfprobe_system: Optional[str] = None,
    writer_selfprobe_user: Optional[str] = None,
    run_comm_gap: bool = True,
    cg_alpha: float = 0.5,
    cg_beta: float = 0.5,
    verbose_timing: bool = False,
) -> Dict[str, Any]:
    """
    Build the probe nodes and return them as a dict:
      "probe_editor"          → Arm A (raw text → editor, captures editor HS for Arm B)
      "probe_editor_selfhs"   → Arm B (editor's own draft HS re-injected)
      "probe_editor_writerhs" → Arm C (writer's answer HS injected into editor)
      "probe_writer_selfhs"   → Arm D (writer's answer HS re-injected into writer)
      "probe_comm_gap"        → Communication Gap metric (Arm A vs Arm C logits,
                                 teacher-forced).  Included iff run_comm_gap=True.

    Parameters
    ----------
    writer_backend, editor_backend : ModelBackend
        Same instance for single-model experiments, or different instances for
        cross-model.  HiddenStateProjector handles hidden-size mismatches.
    inject_layer : int
        Single layer at which to inject hidden states in Arms B, C, and D.
        0 = embedding-layer output; L > 0 = input to decoder layer L
        (1-indexed, matches HF hidden_states[L] convention).
    writer_selfprobe_system / writer_selfprobe_user : str, optional
        Custom prompts for Arm D.  The user-prompt MUST contain exactly one
        `<<<INJECT_HERE>>>` marker — that's where the writer's HS are spliced.
        Defaults to DEFAULT_WRITER_SELFPROBE_SYSTEM / DEFAULT_WRITER_SELFPROBE_USER
        (a "repeat this message" reconstruction prompt).
    editor_user_text : str, optional
        User-turn template for the **text channel** (Arm A).  Must contain
        exactly one `<<<DRAFT_TOKENS>>>` marker where the writer's decoded text
        is spliced in.  Defaults to the built-in "evaluate this draft" prompt.
        Override for domain-specific tasks, e.g. a negotiation reply prompt.
    editor_inject_user_text : str, optional
        User-turn template for the **injection channel** (Arms B, C, CG latent).
        Must contain exactly one `<<<INJECT_HERE>>>` marker.
        Defaults to the built-in injection prompt (mirrors editor_user_text
        with the inject marker).  Should match editor_user_text semantically.
    run_comm_gap : bool
        Whether to include the probe_comm_gap node in the returned dict.
        Adds ~2 fast forward passes per invoke() (no generation loop).
    cg_alpha, cg_beta : float
        Weights on the cosine and JS terms of the Communication Gap metric.
        CG = mean_t (cg_beta * JS_t + cg_alpha * COS_t).
    verbose_timing : bool
        Print wall-clock time per arm so you can see where time goes.

    State keys read: writer_result, writer_output_text, writer_output_token_ids
    State keys written:
        editor_prompt_ids, editor_draft_start, editor_draft_end,
        editor_result, editor_verdict,
        editor_selfhs_verdict, editor_writerhs_verdict,
        writer_selfhs_output,
        comm_gap  (if run_comm_gap)
    """
    projector = HiddenStateProjector.between(writer_backend, editor_backend)

    ed_system = editor_system if editor_system is not None else DEFAULT_EDITOR_SYSTEM
    ed_user_text = editor_user_text if editor_user_text is not None else _EDITOR_USER_TEXT
    ed_inject_user_text = editor_inject_user_text if editor_inject_user_text is not None else _EDITOR_USER_INJECT

    if _DRAFT_MARKER not in ed_user_text:
        raise ValueError(
            f"editor_user_text must contain exactly one {_DRAFT_MARKER!r} marker "
            "indicating where the writer's decoded text should be spliced."
        )
    if _INJECT_MARKER not in ed_inject_user_text:
        raise ValueError(
            f"editor_inject_user_text must contain exactly one {_INJECT_MARKER!r} marker "
            "indicating where the injected hidden states should be placed."
        )

    wsp_system = writer_selfprobe_system if writer_selfprobe_system is not None \
        else DEFAULT_WRITER_SELFPROBE_SYSTEM
    wsp_user = writer_selfprobe_user if writer_selfprobe_user is not None \
        else DEFAULT_WRITER_SELFPROBE_USER
    if _INJECT_MARKER not in wsp_user:
        raise ValueError(
            f"writer_selfprobe_user must contain exactly one {_INJECT_MARKER!r} "
            "marker indicating where the writer's hidden states should be injected."
        )

    def _timed(name: str, fn):
        if not verbose_timing:
            return fn

        def wrapped(state):
            t0 = time.time()
            out = fn(state)
            print(f"  [{name:<24s}] {time.time() - t0:6.2f}s")
            return out

        return wrapped

    # ── Arm A: editor sees the raw essay text ────────────────────────────────
    def probe_editor(state: ProbeMixin) -> ProbeMixin:
        pre_ids, post_ids = _split_prompt_at_marker(
            editor_backend, ed_system, ed_user_text, _DRAFT_MARKER
        )

        writer_toks = state["writer_output_token_ids"]

        # Cross-tokenizer safety: if writer's ids don't fit editor's vocab,
        # round-trip through text and retokenize with the editor's tokenizer.
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
        editor_prompt_ids = torch.tensor([spliced], dtype=torch.long, device=editor_backend.device)

        result = editor_backend.generate(
            prompt_ids=editor_prompt_ids,
            max_new_tokens=max_editor_tokens,
            capture_hidden=True,       # Arm B needs editor HS at draft span
        )
        return {
            **state,
            "editor_prompt_ids": editor_prompt_ids,
            "editor_draft_start": draft_start,
            "editor_draft_end": draft_end,
            "editor_result": result,
            "editor_verdict": result.output_text,
        }

    # ── Arm B: editor's own final-layer HS over the draft span re-injected ───
    def probe_editor_selfhs(state: ProbeMixin) -> ProbeMixin:
        er = state["editor_result"]
        draft_positions = list(range(state["editor_draft_start"], state["editor_draft_end"]))
        num_ph = len(draft_positions)

        # Editor's FINAL decoder-layer HS at each draft-token position
        final_hs = er.hidden_states[-1][0, draft_positions, :]   # (N, hidden)

        prompt_ids, ph_positions = _build_inject_prompt(
            editor_backend, num_ph,
            system=ed_system, user=ed_inject_user_text, marker=_INJECT_MARKER,
        )

        verdict = editor_backend.generate_with_injected_embeds(
            prompt_ids=prompt_ids,
            placeholder_positions=ph_positions,
            injected_hidden=final_hs,
            inject_layer=inject_layer,
            max_new_tokens=max_editor_tokens,
            projector=None,            # same model, no projection
        )
        return {**state, "editor_selfhs_verdict": verdict}

    # ── Arm C: writer's final-layer HS over answer span injected into editor ─
    def probe_editor_writerhs(state: ProbeMixin) -> ProbeMixin:
        wr = state["writer_result"]
        # Answer positions only (skip prompt + thinking tokens)
        ans_start = wr.prompt_len + wr.answer_offset
        writer_out_positions = list(range(ans_start, ans_start + len(wr.gen_token_ids)))
        num_ph = len(writer_out_positions)

        writer_final_hs = wr.hidden_states[-1][0, writer_out_positions, :]  # (N, src_hs)

        prompt_ids, ph_positions = _build_inject_prompt(
            editor_backend, num_ph,
            system=ed_system, user=ed_inject_user_text, marker=_INJECT_MARKER,
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

    # ── Arm D: writer's own full-answer HS re-injected into the writer ───────
    def probe_writer_selfhs(state: ProbeMixin) -> ProbeMixin:
        wr = state["writer_result"]
        # Same answer positions as Arm C, but injected into writer_backend.
        ans_start = wr.prompt_len + wr.answer_offset
        writer_out_positions = list(range(ans_start, ans_start + len(wr.gen_token_ids)))
        num_ph = len(writer_out_positions)

        writer_final_hs = wr.hidden_states[-1][0, writer_out_positions, :]  # (N, hidden)

        prompt_ids, ph_positions = _build_inject_prompt(
            writer_backend, num_ph,
            system=wsp_system, user=wsp_user, marker=_INJECT_MARKER,
        )

        output = writer_backend.generate_with_injected_embeds(
            prompt_ids=prompt_ids,
            placeholder_positions=ph_positions,
            injected_hidden=writer_final_hs,
            inject_layer=inject_layer,
            max_new_tokens=max_writer_selfprobe_tokens,
            projector=None,            # same model, no projection
        )
        return {**state, "writer_selfhs_output": output}

    # ── Communication Gap: Arm A (text) vs Arm C (latent), teacher-forced ────
    def probe_comm_gap(state: ProbeMixin) -> ProbeMixin:
        """
        Score both channels on the SAME reference response (Arm A's verdict),
        then compute the Communication Gap metric on their logits.

        Teacher forcing is critical here: without it, each channel's logits
        at step t would be conditioned on its own previously-generated prefix,
        and divergence would be dominated by prefix drift rather than by the
        channel difference we actually want to measure.
        """
        er = state["editor_result"]
        wr = state["writer_result"]

        # Reference response tokens = Arm A's generated continuation (the
        # tokens AFTER the editor prompt).  Use the full raw generation so
        # we don't drop any tokens to thinking-block post-processing.
        prompt_len_text = state["editor_prompt_ids"].shape[1]
        ref_resp = er.output_ids[:, prompt_len_text:].to(editor_backend.device)

        if ref_resp.shape[1] == 0:
            # No response to score against (editor produced nothing) —
            # skip the metric gracefully.
            return {**state, "comm_gap": {"CG": float("nan"), "T": 0, "note": "empty reference response"}}

        # ── Text channel: plain teacher-forced forward pass ──────────────────
        full_ids_text = torch.cat([state["editor_prompt_ids"], ref_resp], dim=1)
        logits_text = editor_backend.forward_plain(
            full_ids=full_ids_text,
            prompt_len=prompt_len_text,
        )

        # ── Latent channel: rebuild the injection prompt, then append the
        #    exact same reference response and do a patched forward pass. ────
        ans_start = wr.prompt_len + wr.answer_offset
        writer_out_positions = list(range(ans_start, ans_start + len(wr.gen_token_ids)))
        num_ph = len(writer_out_positions)
        writer_final_hs = wr.hidden_states[-1][0, writer_out_positions, :]

        lat_prompt_ids, ph_positions = _build_inject_prompt(
            editor_backend, num_ph,
            system=ed_system, user=ed_inject_user_text, marker=_INJECT_MARKER,
        )
        prompt_len_lat = lat_prompt_ids.shape[1]
        full_ids_lat = torch.cat([lat_prompt_ids, ref_resp], dim=1)
        logits_lat = editor_backend.forward_with_injected_embeds(
            full_ids=full_ids_lat,
            prompt_len=prompt_len_lat,
            placeholder_positions=ph_positions,
            injected_hidden=writer_final_hs,
            inject_layer=inject_layer,
            projector=projector if projector.needs_projection else None,
        )

        cg = communication_gap(logits_lat, logits_text, alpha=cg_alpha, beta=cg_beta)
        return {**state, "comm_gap": cg}

    built = {
        "probe_editor":          _timed("probe_editor",          probe_editor),
        "probe_editor_selfhs":   _timed("probe_editor_selfhs",   probe_editor_selfhs),
        "probe_editor_writerhs": _timed("probe_editor_writerhs", probe_editor_writerhs),
        "probe_writer_selfhs":   _timed("probe_writer_selfhs",   probe_writer_selfhs),
    }
    if run_comm_gap:
        built["probe_comm_gap"] = _timed("probe_comm_gap", probe_comm_gap)
    return built


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def add_probe_to_graph(
    graph: StateGraph,
    writer_backend: ModelBackend,
    editor_backend: ModelBackend,
    entry_node: str,
    max_editor_tokens: int = 60,
    max_writer_selfprobe_tokens: int = 80,
    inject_layer: int = 0,
    editor_system: Optional[str] = None,
    editor_user_text: Optional[str] = None,
    editor_inject_user_text: Optional[str] = None,
    writer_selfprobe_system: Optional[str] = None,
    writer_selfprobe_user: Optional[str] = None,
    run_comm_gap: bool = True,
    cg_alpha: float = 0.5,
    cg_beta: float = 0.5,
    verbose_timing: bool = False,
) -> StateGraph:
    """
    Attach the probe nodes to `graph` as a linear chain starting from
    `entry_node`.  Final node edges to END.

        entry_node
          → probe_editor            (Arm A)
          → probe_editor_selfhs     (Arm B)
          → probe_editor_writerhs   (Arm C)
          → probe_writer_selfhs     (Arm D)
          → probe_comm_gap          (CG metric, iff run_comm_gap)
          → END
    """
    nodes = make_probe_nodes(
        writer_backend,
        editor_backend,
        max_editor_tokens=max_editor_tokens,
        max_writer_selfprobe_tokens=max_writer_selfprobe_tokens,
        inject_layer=inject_layer,
        editor_system=editor_system,
        editor_user_text=editor_user_text,
        editor_inject_user_text=editor_inject_user_text,
        writer_selfprobe_system=writer_selfprobe_system,
        writer_selfprobe_user=writer_selfprobe_user,
        run_comm_gap=run_comm_gap,
        cg_alpha=cg_alpha,
        cg_beta=cg_beta,
        verbose_timing=verbose_timing,
    )

    order = [
        "probe_editor",
        "probe_editor_selfhs",
        "probe_editor_writerhs",
        "probe_writer_selfhs",
    ]
    if run_comm_gap:
        order.append("probe_comm_gap")
    for name in order:
        graph.add_node(name, nodes[name])
    graph.add_edge(entry_node, order[0])
    for a, b in zip(order, order[1:]):
        graph.add_edge(a, b)
    graph.add_edge(order[-1], END)
    return graph


def make_graph(
    writer_backend: ModelBackend,
    editor_backend: ModelBackend,
    max_writer_tokens: int = 80,
    max_editor_tokens: int = 60,
    max_writer_selfprobe_tokens: int = 80,
    inject_layer: int = 0,
    writer_system: Optional[str] = None,
    editor_system: Optional[str] = None,
    editor_user_text: Optional[str] = None,
    editor_inject_user_text: Optional[str] = None,
    writer_selfprobe_system: Optional[str] = None,
    writer_selfprobe_user: Optional[str] = None,
    run_comm_gap: bool = True,
    cg_alpha: float = 0.5,
    cg_beta: float = 0.5,
    verbose_timing: bool = False,
):
    """
    Build and compile a complete standalone graph:
        writer → probe_editor (A) → probe_editor_selfhs (B)
               → probe_editor_writerhs (C) → probe_writer_selfhs (D)
               → probe_comm_gap (CG metric) → END

    Cost: 5 generate() calls + 2 fast forward passes per invoke()
    (drop to 5 generate() calls by passing run_comm_gap=False).

    Returns a compiled LangGraph app; call with:
        final = app.invoke({"task": "your task here"})
    """
    g = StateGraph(AgentState)

    wr_system = writer_system if writer_system is not None else DEFAULT_WRITER_SYSTEM

    def writer_node(state: AgentState) -> AgentState:
        t0 = time.time()
        prompt_ids = writer_backend.build_chat_prompt(
            system=wr_system,
            user=state["task"],
        )
        result = writer_backend.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_writer_tokens,
            capture_hidden=True,
        )
        if verbose_timing:
            print(f"  [{'writer':<24s}] {time.time() - t0:6.2f}s")
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
        max_editor_tokens=max_editor_tokens,
        max_writer_selfprobe_tokens=max_writer_selfprobe_tokens,
        inject_layer=inject_layer,
        editor_system=editor_system,
        editor_user_text=editor_user_text,
        editor_inject_user_text=editor_inject_user_text,
        writer_selfprobe_system=writer_selfprobe_system,
        writer_selfprobe_user=writer_selfprobe_user,
        run_comm_gap=run_comm_gap,
        cg_alpha=cg_alpha,
        cg_beta=cg_beta,
        verbose_timing=verbose_timing,
    )

    return g.compile()

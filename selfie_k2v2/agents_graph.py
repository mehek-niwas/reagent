"""
LangGraph writer → editor graph using the shared K2V2Backend.

The graph state carries the writer's hidden states through so the editor
(and the interpretation passes) can use them.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict, Tuple, Any

import torch
import pandas as pd
from langgraph.graph import StateGraph, END

from k2v2_backend import K2V2Backend, GenerationResult


WRITER_SYSTEM = (
    "You are K2, a helpful assistant. You are a WRITER agent. Write a short, "
    "clear paragraph (3-4 sentences) on the topic the user provides. Be concise."
)

EDITOR_SYSTEM = (
    "You are K2, a helpful assistant. You are an EDITOR agent. Evaluate the "
    "draft between <DRAFT> and </DRAFT> tags. Give one short sentence of "
    "feedback: is it clear? Is anything missing?"
)


class AgentState(TypedDict, total=False):
    # inputs
    task: str

    # writer outputs
    writer_prompt_ids: torch.Tensor
    writer_result: GenerationResult
    writer_output_text: str
    writer_output_token_ids: List[int]

    # editor outputs (normal — text passthrough)
    editor_prompt_ids: torch.Tensor
    editor_draft_start: int             # position in editor prompt where writer's tokens begin
    editor_draft_end: int               # exclusive
    editor_result: GenerationResult
    editor_verdict: str

    # editor with injected embeddings (Experiment 2)
    editor_injected_verdict: Optional[str]

    # SELFIE outputs
    selfie_writer: Optional[pd.DataFrame]                 # Experiment 1
    selfie_editor_on_draft: Optional[pd.DataFrame]        # Experiment 3


def make_graph(backend: K2V2Backend, max_writer_tokens: int = 40,
               max_editor_tokens: int = 60):

    # ----- writer node -----
    def writer_node(state: AgentState) -> AgentState:
        prompt_ids = backend.build_chat_prompt(
            system=WRITER_SYSTEM,
            user=state["task"],
            reasoning_effort="medium",
        )
        result = backend.generate(
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

    # ----- editor node (text-passthrough; also builds prompt we reuse for Exp 2/3) -----
    def editor_node(state: AgentState) -> AgentState:
        tok = backend.tokenizer
        device = backend.device

        # Build the editor prompt by splicing the writer's raw token IDs into
        # the draft tags. This keeps token alignment exact (no retokenization).
        draft_open_ids = tok.encode("<DRAFT>", add_special_tokens=False)
        draft_close_ids = tok.encode("</DRAFT>", add_special_tokens=False)

        # User message uses a text placeholder we'll splice around.
        # We build the prompt in two halves around a marker, then splice.
        MARKER = "<<<DRAFT_TOKENS>>>"
        user_msg = (
            f"Please evaluate this draft:\n<DRAFT>{MARKER}</DRAFT>\nYour feedback:"
        )
        # Render with chat template, then find the marker tokens and replace.
        full_ids = backend.build_chat_prompt(
            system=EDITOR_SYSTEM,
            user=user_msg,
            reasoning_effort="medium",
        )[0].tolist()

        marker_ids = tok.encode(MARKER, add_special_tokens=False)
        # Find marker in full_ids
        marker_pos = _find_sublist(full_ids, marker_ids)
        if marker_pos < 0:
            raise RuntimeError("Could not locate marker in rendered editor prompt.")

        # Replace marker_ids with the actual writer output token IDs.
        spliced = (
            full_ids[:marker_pos]
            + state["writer_output_token_ids"]
            + full_ids[marker_pos + len(marker_ids):]
        )
        draft_start = marker_pos
        draft_end = marker_pos + len(state["writer_output_token_ids"])

        editor_prompt_ids = torch.tensor([spliced], dtype=torch.long, device=device)

        result = backend.generate(
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
        }

    # ----- Experiment 1: SELFIE on writer's output positions -----
    def selfie_writer_node(state: AgentState) -> AgentState:
        wr = state["writer_result"]
        # Writer's output positions in its own full sequence:
        writer_out_positions = list(range(wr.prompt_len,
                                          wr.prompt_len + len(wr.gen_token_ids)))
        # Sweep a few layers (middle + late). For a 80-layer K2-V2, 20/40/60.
        probe_layers = _probe_layers(backend.num_layers)

        positions = [(L, t) for L in probe_layers for t in writer_out_positions]
        df = backend.selfie_on_positions(
            hidden_states=wr.hidden_states,
            input_ids=wr.output_ids,
            positions=positions,
            max_new_tokens=15,
        )
        return {**state, "selfie_writer": df}

    # ----- Experiment 3: SELFIE on editor's hidden states at the draft span -----
    def selfie_editor_node(state: AgentState) -> AgentState:
        er = state["editor_result"]
        draft_positions = list(range(state["editor_draft_start"], state["editor_draft_end"]))
        probe_layers = _probe_layers(backend.num_layers)
        positions = [(L, t) for L in probe_layers for t in draft_positions]

        df = backend.selfie_on_positions(
            hidden_states=er.hidden_states,
            input_ids=er.output_ids,
            positions=positions,
            max_new_tokens=15,
        )
        return {**state, "selfie_editor_on_draft": df}

    # ----- Experiment 2: editor with writer's hidden states injected at placeholders -----
    def injected_editor_node(state: AgentState) -> AgentState:
        tok = backend.tokenizer
        device = backend.device

        # Editor prompt with N placeholder tokens = number of writer output tokens.
        num_ph = len(state["writer_output_token_ids"])
        # Use a specific unused-ish token repeatedly; its embedding will be
        # overwritten anyway. We'll use the pad token if available, else BOS.
        ph_id = tok.pad_token_id if tok.pad_token_id is not None else tok.bos_token_id
        if ph_id is None:
            ph_id = 0

        MARKER = "<<<INJECT_HERE>>>"
        user_msg = (
            f"Please evaluate this draft:\n<DRAFT>{MARKER}</DRAFT>\nYour feedback:"
        )
        full_ids = backend.build_chat_prompt(
            system=EDITOR_SYSTEM,
            user=user_msg,
            reasoning_effort="medium",
        )[0].tolist()

        marker_ids = tok.encode(MARKER, add_special_tokens=False)
        marker_pos = _find_sublist(full_ids, marker_ids)
        spliced = (
            full_ids[:marker_pos]
            + [ph_id] * num_ph
            + full_ids[marker_pos + len(marker_ids):]
        )
        ph_positions = list(range(marker_pos, marker_pos + num_ph))
        prompt_ids = torch.tensor([spliced], dtype=torch.long, device=device)

        # Use writer's FINAL hidden state at the writer's output positions.
        wr = state["writer_result"]
        writer_out_positions = list(range(wr.prompt_len,
                                          wr.prompt_len + len(wr.gen_token_ids)))
        final_layer = len(wr.hidden_states) - 1  # last hidden state
        injected = wr.hidden_states[final_layer][0, writer_out_positions, :]  # (N, H)

        verdict = backend.generate_with_injected_embeds(
            prompt_ids=prompt_ids,
            placeholder_positions=ph_positions,
            injected_hidden=injected,
            inject_layer=0,     # patch at embedding layer
            max_new_tokens=60,
        )
        return {**state, "editor_injected_verdict": verdict}

    # ----- build graph -----
    g = StateGraph(AgentState)
    g.add_node("writer", writer_node)
    g.add_node("editor", editor_node)
    g.add_node("selfie_writer", selfie_writer_node)
    g.add_node("selfie_editor", selfie_editor_node)
    g.add_node("injected_editor", injected_editor_node)

    g.set_entry_point("writer")
    g.add_edge("writer", "editor")
    g.add_edge("editor", "selfie_writer")
    g.add_edge("selfie_writer", "selfie_editor")
    g.add_edge("selfie_editor", "injected_editor")
    g.add_edge("injected_editor", END)

    return g.compile()


def _find_sublist(haystack: List[int], needle: List[int]) -> int:
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def _probe_layers(num_layers: int) -> List[int]:
    # Pick three reasonable probe layers: early-middle, middle, late.
    # hidden_states has num_layers+1 entries (index 0 = embeds, num_layers = final).
    # We want meaningful content layers, not embeddings or the final.
    if num_layers >= 60:
        return [num_layers // 4, num_layers // 2, (3 * num_layers) // 4]
    if num_layers >= 20:
        return [num_layers // 3, num_layers // 2, (2 * num_layers) // 3]
    return [max(1, num_layers // 2)]

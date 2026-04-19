from .model_backend import ModelBackend, GenerationResult, HiddenStateProjector
from .k2v2_backend import K2V2Backend
from .agents_graph import (
    make_graph,
    make_probe_nodes,
    add_probe_to_graph,
    AgentState,
    ProbeMixin,
    SelfieProbeMixin,  # backwards-compat alias of ProbeMixin
    DEFAULT_WRITER_SYSTEM,
    DEFAULT_EDITOR_SYSTEM,
    DEFAULT_WRITER_SELFPROBE_SYSTEM,
    DEFAULT_WRITER_SELFPROBE_USER,
)
from .metrics import communication_gap
from .selfie_fork import InterpretationPrompt, interpret, capture_hidden_states

__all__ = [
    # backends
    "ModelBackend",
    "K2V2Backend",
    "GenerationResult",
    "HiddenStateProjector",
    # graph
    "make_graph",
    "make_probe_nodes",
    "add_probe_to_graph",
    "AgentState",
    "ProbeMixin",
    "SelfieProbeMixin",
    "DEFAULT_WRITER_SYSTEM",
    "DEFAULT_EDITOR_SYSTEM",
    "DEFAULT_WRITER_SELFPROBE_SYSTEM",
    "DEFAULT_WRITER_SELFPROBE_USER",
    # metrics
    "communication_gap",
    # selfie primitives (kept available for anyone who wants to use them
    # directly; not used by the 4-arm probe graph)
    "InterpretationPrompt",
    "interpret",
    "capture_hidden_states",
]

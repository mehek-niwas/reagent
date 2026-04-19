"""
Quantitative metrics for comparing agent communication channels.

The centrepiece is :func:`communication_gap` — a per-step scalar distance
between the editor's next-token distributions under two channels:

    TEXT CHANNEL     : editor reads the raw essay text in its prompt
    LATENT CHANNEL   : editor reads the writer's final-layer hidden states
                       (injected into placeholder positions)

Both channels MUST be evaluated via teacher forcing on the same reference
response; otherwise the logits at step t are conditioned on different prefixes
and any comparison is inflated by prefix drift rather than channel difference.
See :func:`agents_graph.make_probe_nodes` (``probe_comm_gap``) for the node
that wires this into the LangGraph workflow.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def communication_gap(
    logits_lat: torch.Tensor,   # (T_lat, vocab)
    logits_text: torch.Tensor,  # (T_text, vocab)
    alpha: float = 0.5,
    beta: float = 0.5,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Compute the Communication Gap (CG) metric between two logit streams
    produced by the same editor on the same reference response but under
    different input channels.

    For each decoding step t (up to T = min(T_lat, T_text)):
      JS_t  = Jensen-Shannon divergence between softmax(logits_lat[t])
              and softmax(logits_text[t])
      COS_t = 1 - cosine_similarity(logits_lat[t], logits_text[t])

    CG = mean_t (beta * JS_t + alpha * COS_t)

    Parameters
    ----------
    logits_lat, logits_text : torch.Tensor
        Next-token logits aligned to the same response positions.  Shapes
        (T, vocab).  If T differs between the two, both are truncated to
        the minimum length and a warning-free note is recorded in the output.
    alpha, beta : float
        Weights on the cosine term and JS term.  Defaults 0.5/0.5 reproduce
        the spec.
    eps : float
        Small constant added inside the log to avoid log(0) when a channel
        assigns zero probability to some vocab item.

    Returns
    -------
    dict with keys:
      "CG"          : float, scalar metric
      "JS_mean"     : float, mean JS over response positions
      "COS_mean"    : float, mean (1 - cos) over response positions
      "per_step"    : torch.Tensor of shape (T,), the CG breakdown per step
      "JS_per_step" : torch.Tensor of shape (T,)
      "COS_per_step": torch.Tensor of shape (T,)
      "T"           : int, effective length after truncation
      "alpha"       : float
      "beta"        : float

    Interpretation
    --------------
      Low CG  →  editor behaves similarly under raw text and latent injection
                 (writer and editor share compatible internal representations)
      High CG →  channels differ significantly (agent communication gap)
    """
    if logits_lat.dim() != 2 or logits_text.dim() != 2:
        raise ValueError(
            f"Expected (T, vocab) tensors, got {tuple(logits_lat.shape)} "
            f"and {tuple(logits_text.shape)}."
        )
    if logits_lat.shape[-1] != logits_text.shape[-1]:
        raise ValueError(
            f"Vocab sizes differ: {logits_lat.shape[-1]} vs "
            f"{logits_text.shape[-1]}.  CG requires the same editor model "
            "(and therefore the same tokenizer) on both channels."
        )

    T = min(logits_lat.shape[0], logits_text.shape[0])
    if T == 0:
        raise ValueError("Empty logits — T must be >= 1.")

    lat = logits_lat[:T].float()
    txt = logits_text[:T].float()

    # Probability distributions
    p_lat = lat.softmax(dim=-1)
    p_txt = txt.softmax(dim=-1)

    # Jensen-Shannon divergence: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M)
    # Computed in nats, then normalised by log(2) so the result is in [0, 1]
    # (identical to using log base-2; 0 = identical distributions, 1 = maximally different).
    import math as _math
    m = 0.5 * (p_lat + p_txt)
    log_p_lat = (p_lat + eps).log()
    log_p_txt = (p_txt + eps).log()
    log_m = (m + eps).log()
    kl_lat_m = (p_lat * (log_p_lat - log_m)).sum(dim=-1)
    kl_txt_m = (p_txt * (log_p_txt - log_m)).sum(dim=-1)
    js_t = 0.5 * (kl_lat_m + kl_txt_m) / _math.log(2)
    js_t = js_t.clamp(0.0, 1.0)  # numerical: JS in bits is in [0, 1]

    # Cosine-based logit distance
    cos_t = 1.0 - F.cosine_similarity(lat, txt, dim=-1)

    per_step = beta * js_t + alpha * cos_t

    return {
        "CG": float(per_step.mean().item()),
        "JS_mean": float(js_t.mean().item()),
        "COS_mean": float(cos_t.mean().item()),
        "per_step": per_step.detach().cpu(),
        "JS_per_step": js_t.detach().cpu(),
        "COS_per_step": cos_t.detach().cpu(),
        "T": int(T),
        "alpha": float(alpha),
        "beta": float(beta),
    }

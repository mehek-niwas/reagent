"""
Backward-compatibility shim.

K2V2Backend is now a thin alias for ModelBackend with K2-V2-Instruct defaults
pre-set (reasoning_effort="medium" forwarded to the chat template).

New code should use ModelBackend directly:
    from selfie_k2v2.model_backend import ModelBackend
"""

from .model_backend import GenerationResult, HiddenStateProjector, ModelBackend  # noqa: F401


class K2V2Backend(ModelBackend):
    """
    ModelBackend pre-configured for LLM360/K2-V2-Instruct.
    Passes reasoning_effort="medium" to the chat template automatically.
    """

    def __init__(
        self,
        model_name: str = "LLM360/K2-V2-Instruct",
        **kwargs,
    ):
        kwargs.setdefault("chat_template_kwargs", {"reasoning_effort": "medium"})
        super().__init__(model_name=model_name, **kwargs)

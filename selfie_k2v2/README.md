# Writer / Editor Hidden-State Probe

Three-arm comparison of agent communication via hidden states. Works across Qwen 2.5 / 3.5,
LLaMA 3.x, Gemma 3/4, and K2-V2 via a single generic `ModelBackend`.

## The three arms

| Arm | Key | What it does |
|-----|-----|--------------|
| **A** | `editor_verdict` | Editor receives the **raw essay text** in its prompt. |
| **B** | `editor_selfhs_verdict` | Editor's **own final-layer hidden states** over the draft span (captured during Arm A) are re-injected at `inject_layer`. |
| **C** | `editor_writerhs_verdict` | The **writer's final-layer hidden states** (over its answer tokens) are injected into the editor at `inject_layer`. |

All three arms share a single `inject_layer` parameter. Injection is done **all at once** in a single
vectorised forward pass — not one token at a time.

Cost: **4 `generate()` calls per run** — writer + A + B + C.

## Layout

```
selfie_k2v2/
├── model_backend.py    # Generic ModelBackend + HiddenStateProjector (cross-model)
├── k2v2_backend.py     # Thin K2-V2 subclass (adds reasoning_effort default)
├── agents_graph.py     # LangGraph: writer → A → B → C → END
├── selfie_fork/        # Standalone SELFIE primitives (not used by the 3-arm graph)
├── demo.ipynb          # Runnable demo
├── demo.py             # Plain-Python version of the demo
└── README.md
```

## Install

```bash
pip install .
# or, from GitHub:
pip install git+https://github.com/YOUR_USERNAME/reagent.git
```

For **Qwen 3.5 only** also install the linear-attention kernels (otherwise Qwen 3.5 is 10-20× slower):

```bash
pip install flash-linear-attention causal-conv1d
```

## Minimal use

```python
import torch
from selfie_k2v2 import ModelBackend, make_graph

backend = ModelBackend("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="auto")
graph = make_graph(backend, backend, max_writer_tokens=80, max_editor_tokens=48, inject_layer=0)
final = graph.invoke({"task": "Describe a golden retriever in 3 sentences."})

print("A:", final["editor_verdict"])
print("B:", final["editor_selfhs_verdict"])
print("C:", final["editor_writerhs_verdict"])
```

## Plugging into an existing LangGraph workflow

```python
from selfie_k2v2 import ProbeMixin, add_probe_to_graph
from langgraph.graph import StateGraph

class MyState(ProbeMixin, total=False):
    task: str

def my_writer(state):
    prompt_ids = backend.build_chat_prompt(system="...", user=state["task"])
    result = backend.generate(prompt_ids, max_new_tokens=80, capture_hidden=True)
    return {
        **state,
        "writer_result": result,
        "writer_output_text": result.output_text,
        "writer_output_token_ids": result.gen_token_ids,
    }

g = StateGraph(MyState)
g.add_node("my_writer", my_writer)
g.set_entry_point("my_writer")
add_probe_to_graph(g, backend, backend, entry_node="my_writer", inject_layer=0)
app = g.compile()
```

## Cross-model support

Writer and editor can be different models with different hidden sizes — `HiddenStateProjector`
handles the dimensionality mismatch via a random orthogonal projection:

```python
writer_backend = ModelBackend("Qwen/Qwen3.5-9B",  dtype=torch.bfloat16, device_map="auto")
editor_backend = ModelBackend("Qwen/Qwen3.5-27B", dtype=torch.bfloat16, device_map="auto")
graph = make_graph(writer_backend, editor_backend, inject_layer=10)
```

## Thinking models (Qwen 3 / 3.5)

Reasoning stays **ON** by default. `ModelBackend` automatically strips `<think>…</think>` from
`output_text` and `gen_token_ids` so injection uses only answer tokens. Raw thinking is still
accessible via `result.thinking_text`.

## Tuning notes

- `inject_layer=0` = embedding-layer output. Try `num_layers // 2` for mid-stack injection — often
  more coherent than very early or very late.
- If Arm C produces nonsense, the writer-side HS may be too late-layer to drive the editor's early
  layers. Move `inject_layer` later, or inject from an earlier writer layer by subclassing and
  overriding which HS you pull from `wr.hidden_states[-1]`.

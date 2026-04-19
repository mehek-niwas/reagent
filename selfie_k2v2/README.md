# Writer / Editor Hidden-State Probe

Four-arm comparison of agent communication via hidden states. Works across Qwen 2.5 / 3.5,
LLaMA 3.x, Gemma 3/4, and K2-V2 via a single generic `ModelBackend`.

## The four arms

| Arm | Key | What it does |
|-----|-----|--------------|
| **A** | `editor_verdict` | Editor receives the **raw essay text** in its prompt. |
| **B** | `editor_selfhs_verdict` | Editor's **own final-layer hidden states** over the draft span (captured during Arm A) are re-injected at `inject_layer`. |
| **C** | `editor_writerhs_verdict` | The **writer's final-layer hidden states** (over its answer tokens) are injected into the editor at `inject_layer`. |
| **D** | `writer_selfhs_output` | Writer's **own final-layer hidden states** (full-sentence answer) are re-injected into the **writer** itself with a user-configurable prompt (default: "repeat this message"). Sanity check that the HS encode the essay content. |

All four arms share a single `inject_layer` parameter. Injection is done **all at once** in a single
vectorised forward pass — not one token at a time.

### Communication Gap (CG) metric

In addition to the four arms, the graph computes a **Communication Gap** scalar: a weighted mix of
Jensen-Shannon divergence and logit-cosine distance between the editor's next-token distributions
on the **text channel** (Arm A input) vs. the **latent channel** (Arm C input). Both channels are
scored on the **same reference response** (Arm A's verdict) under **teacher forcing**, so the
number reflects channel difference — not autoregressive prefix drift.

```
CG = mean_t [ beta · JS_t  +  alpha · COS_t ]
```

- Low CG → editor behaves similarly under raw text and injected HS (agents speak the same language).
- High CG → channels disagree, a measurable communication gap between writer and editor.

Available as `final["comm_gap"]` — keys `CG`, `JS_mean`, `COS_mean`, `per_step`, `JS_per_step`,
`COS_per_step`, `T`. See `selfie_k2v2/metrics.py` for the standalone function and
`demo.ipynb` §5 for a per-step plot.

Cost: **5 `generate()` calls + 2 fast forward passes per run** — writer + A + B + C + D + CG.
Pass `run_comm_gap=False` to drop the 2 extra forwards.

## Layout

```
selfie_k2v2/
├── model_backend.py    # Generic ModelBackend + HiddenStateProjector (cross-model)
├── k2v2_backend.py     # Thin K2-V2 subclass (adds reasoning_effort default)
├── agents_graph.py     # LangGraph: writer → A → B → C → D → comm_gap → END
├── metrics.py          # communication_gap() — Jensen-Shannon + cosine logit distance
├── selfie_fork/        # Standalone SELFIE primitives (not used by the 4-arm graph)
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
print("D:", final["writer_selfhs_output"])
cg = final["comm_gap"]
print(f"CG: {cg['CG']:.4f}  (JS={cg['JS_mean']:.4f}, 1-cos={cg['COS_mean']:.4f})")
```

### Customising the writer self-probe prompt (Arm D)

```python
final = graph.invoke({"task": "..."}) if False else None  # illustration
# or, at graph-build time:
graph = make_graph(
    backend, backend, inject_layer=0,
    writer_selfprobe_user=(
        "Here is an encoded thought: '<<<INJECT_HERE>>>'.\n"
        "In one sentence, what topic does it cover?"
    ),
)
```

The user-prompt must contain exactly one `<<<INJECT_HERE>>>` marker — that's where the
writer's hidden states get spliced into the residual stream.

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

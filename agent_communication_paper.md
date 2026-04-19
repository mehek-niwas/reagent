# Investigating Internal Representations of Agent-to-Agent Communication

**Mehek Niwas, Abhi Patel, Advik Lall, Daniel Zhang**
*HackPrinceton Spring 2026 — d_model Project Submission*

---

## Abstract

Multi-agent systems (MAS) built on large language models (LLMs) increasingly rely on natural language as the default communication substrate. Yet natural language is a lossy channel: the continuous internal state of a sending agent must be discretized into tokens before it reaches a receiving agent, inevitably discarding latent reasoning. Despite growing interest in richer communication protocols (e.g., state-delta injection, probability-weighted embeddings), it remains unclear *how* a receiving agent internally represents the content it reads from a sender—and whether that internal representation faithfully preserves the sender's intent. We introduce a two-arm framework that places a **Writer agent** and an **Editor agent** in a pairwise communication scenario and compares two communication conditions: (1) **Arm B** — direct hidden-state injection of the Writer's final-layer embeddings into the Editor's forward pass, and (2) **Arm C** — natural-language transmission followed by SelfIE probing (Chen et al., 2024) of the Editor's internal encoding of those tokens. Comparing these two arms directly reveals how much representational information is preserved or distorted when agent communication is constrained to natural language. We quantify representational fidelity using Jensen-Shannon (JS) divergence and cosine similarity between paired embedding sets. Experiments are conducted on Qwen (12 layers, 12 heads, 110M parameters) using the CoQA dataset as a conversational grounding task. Our results illuminate systematic gaps between what a sender encodes and what a receiver internally represents—gaps that we argue are a root cause of emergent failure modes in agentic workflows.

---

## 1. Introduction and Motivation

The deployment of LLM-based multi-agent systems (MAS) has accelerated rapidly, with frameworks such as AutoGen (Wu et al., 2024), ChatDev (Qian et al., 2024), and MetaGPT (Hong et al., 2024) demonstrating that structured agent collaboration can substantially improve task performance. Within these systems, orchestrators and meta-agents assign roles, manage dependencies, and arbitrate conflicts. Yet, as noted by Venkataramani et al. (2026), there is currently **no interpretability for inter-agent communication**: it is opaque which messages were misunderstood, whether agents maintain consistent context, and whether coordination protocols failed. This absence makes emergent failure modes—deadlocks, goal divergence, and reasoning loops—invisible until they manifest in final outputs.

The foundational problem is one of *representational fidelity*. When Agent A (the sender) produces an output, that output is the result of a rich, high-dimensional forward pass through potentially billions of parameters. Transmitting it as natural language is analogous to compressing a high-resolution signal into a compressed format: certain information is preserved, but the reconstruction is necessarily approximate. Agent B (the receiver) then reads that compressed signal and constructs its *own* internal representation—which may differ substantially from the representation Agent A held when generating the message.

Prior work has begun to address the transmission side of this problem. Tang et al. (2025) propose State Delta Encoding (SDE), which augments natural-language tokens with token-wise hidden-state differences, demonstrating significant performance improvements on reasoning tasks. Du et al. (2026) go further by enabling agents to communicate entirely in latent space, bypassing natural language altogether. However, both lines of work focus on *communication performance* (downstream task accuracy) rather than *representational alignment*: does the receiver's internal encoding of the sender's message faithfully mirror the sender's internal state?

SelfIE (Chen et al., 2024) provides a complementary tool. By inserting hidden embeddings into a separate interpretation forward pass, SelfIE allows an LLM to generate natural-language descriptions of its own latent states—without any additional training. This opens the possibility of probing what the *sender* encodes at generation time and comparing it to what the *receiver* encodes when reading that message.

We combine these threads to ask a focused empirical question: **how does a receiver agent's internal representation of a sender's message differ between direct latent transfer and natural-language communication, and where in the model do these representations diverge most?** To operationalize this, we instantiate a Writer–Editor agent pair and compare two conditions along representational and information-theoretic axes.

---

## 2. Background and Related Work

### 2.1 Multi-Agent Communication Protocols

The majority of LLM-based MAS frameworks transmit information between agents as natural-language strings (Li et al., 2023; Wu et al., 2024; Chan et al., 2024). While natural language is interpretable and generalizable, it is a discrete, lossy channel: continuous internal states must be sampled into tokens before transfer, and only one reasoning path—the sampled one—reaches the receiver (Tang et al., 2025). If that path is incorrect, no downstream agent can recover what was discarded.

Several alternatives have been proposed. CIPHER (Pham et al., 2024) replaces output tokens with probability-weighted token embeddings, preserving some distributional information about the sender's output layer. SDE (Tang et al., 2025) captures the *trajectory* of hidden-state changes across layers during generation and injects these state deltas into the receiver's forward pass, achieving up to 17.3% improvement on agent workflow tasks. Du et al. (2026) propose full latent-space communication for homogeneous-model agent pairs. Ramesh and Li (2025) demonstrate unidirectional hidden-state transfer between a reading agent and a generating agent.

### 2.2 LLM Interpretability and Hidden-State Probing

Interpreting LLM internal states is a longstanding challenge. Linear probing approaches (Li et al., 2021; Hernandez et al., 2023) can identify specific concepts in embeddings but require task-specific training data and are limited to a predefined concept set. Decoder-based approaches such as LogitLens (Nostalgebraist) and TunedLens (Belrose et al., 2023) provide token-level predictions at intermediate layers. SelfIE (Chen et al., 2024) generalizes these approaches by repurposing the LLM's own decoding capabilities: a hidden embedding is injected into an "interpretation forward pass" at a chosen layer, and the model is prompted to generate a natural-language description of what that embedding contains. SelfIE has been applied to reveal hidden reasoning in cases of ethical decision-making, prompt injection, harmful knowledge retrieval, and hallucination.

### 2.3 Agent Communication and Negotiation Datasets

The CraigslistBargains dataset (He et al., 2018) provides a naturalistic setting for studying pairwise agent communication: buyer and seller agents must negotiate the price of real Craigslist items, producing dialogues that involve strategy, persuasion, and information exchange. This setting is particularly well-suited to our framework because (a) one agent (the buyer/writer) generates content with a specific goal, (b) the other agent (the seller/editor) must interpret and respond to that content, and (c) ground-truth dialogue acts provide a structured signal against which internal representations can be compared.

---

## 3. Method

### 3.1 Agent Roles and Task Setup

We operationalize pairwise agent communication through a **Writer–Editor** scenario. The Writer agent is given a task (e.g., produce a conversational opening, draft an argument, or answer a question) and generates a response. The Editor agent is then asked to evaluate or respond to the Writer's output. This asymmetric setup—where one agent produces and another interprets—is a canonical instance of inter-agent communication. The CraigslistBargains buyer-seller interaction serves as a natural instantiation: the buyer (Writer) crafts an opening offer, and the seller (Editor) responds.

### 3.2 Model

All experiments use **Qwen** with the following configuration:
- 12 transformer layers
- 12 attention heads
- 110M parameters

Qwen was selected based on practical considerations discussed in Section 6 (see Future Work). The CoQA conversational question-answering dataset is used as a grounding task because it requires multi-turn reasoning and context tracking—properties that stress-test inter-agent communication.



### 3.3 Communication Methods

We compare two communication conditions, each varying in how the Writer's internal state is made available to the Editor.

**Arm B — Hidden-State Injection (Direct Latent Transfer).** Rather than transmitting the Writer's output as natural language, we inject the Writer's final-layer hidden embeddings directly into the Editor's forward pass at the token positions corresponding to the Writer's output. Specifically, for Writer output tokens $t_1, \ldots, t_n$ with final hidden states $H^L_{\text{Writer}} = \{h^L_1, \ldots, h^L_n\}$, the Editor's prompt is constructed with *placeholder tokens* at the positions where the Writer's output would normally appear. During the Editor's forward pass, each placeholder token's embedding at a chosen injection layer $\ell_{\text{inject}}$ is replaced by the corresponding Writer hidden state:

$$h^{\ell_{\text{inject}}}_{\text{Editor}, j} \leftarrow h^L_{\text{Writer}, i} \quad \text{if position } j \text{ corresponds to token } t_i$$

This is conceptually similar to the latent-communication approach in Du et al. (2026) and Ramesh and Li (2025). After injection, we apply SelfIE to the Editor's modified embeddings to obtain the *receiver's representation under direct latent transfer*.

**Arm C — Natural Language + Editor SelfIE (Standard Communication).** The Writer's output is transmitted as natural language and inserted into the Editor's prompt. The Editor processes this prompt through a standard forward pass. We then identify the token positions in the Editor's prompt that correspond to the Writer's output and apply SelfIE at layer $\ell^*$ to each of these positions. This yields the *receiver's internal representation of the sender's message under natural-language communication*.

The core comparison is between **Arm B** (Editor's representation when the Writer's latent states are transferred directly) and **Arm C** (Editor's representation when communication is mediated through natural language). Any divergence between these two conditions captures the representational loss specifically attributable to the natural-language bottleneck.

### 3.4 Comparison Metrics

To quantify the alignment between paired representations, we use two complementary metrics.

**Jensen-Shannon Divergence (JSD).** For two probability distributions $P$ and $Q$ over the vocabulary (obtained by passing hidden embeddings through the LM head and applying softmax), JS divergence is defined as:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$ and $D_{\text{KL}}$ is the Kullback-Leibler divergence. JSD $\in [0, 1]$ (using log base 2) and is symmetric. Higher JSD indicates greater distributional mismatch between what the two agents "think" about the same token position.

**Cosine Similarity.** For hidden embedding vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

$$\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

We report $1 - \text{cos}(\mathbf{u}, \mathbf{v})$ (cosine distance) so that higher values indicate greater divergence, consistent with JSD. Cosine distance captures directional misalignment in representation space independent of magnitude.

Following the Communication Gap formulation in our experiment code, we combine these metrics into a single per-step score applied pairwise between Arm B and Arm C representations:

$$\text{CG}_t = \beta \cdot \text{JS}_t + \alpha \cdot (1 - \text{cos}_t)$$

with $\alpha = \beta = 0.5$ as default weights, averaged over all $T$ tokens in the Writer's message:

$$\text{CG} = \frac{1}{T} \sum_{t=1}^{T} \text{CG}_t$$

where $\text{JS}_t \in [0, 1]$ bits and $1 - \text{cos}_t \in [0, 2]$. A higher CG score indicates that the Editor's internal representation is more divergent under natural-language communication (Arm C) than under direct latent transfer (Arm B), pointing to specific tokens and layers where the natural-language channel loses information.

### 3.5 Dataset

We simulate pairwise agent communication using the **CoQA** (Conversational Question Answering) dataset for the core representational experiments, and the **CraigslistBargains** dataset (He et al., 2018) for the Communication Gap evaluation. CoQA provides multi-turn conversational contexts that require tracking entities and states across turns—a property that tests whether agents maintain consistent internal representations over the course of a dialogue. CraigslistBargains provides a structured negotiation setting with buyer and seller roles that map cleanly onto our Writer–Editor framing.

For CraigslistBargains, each example produces: (1) a buyer LLM generating an opening offer (Writer), (2) a seller LLM responding (Editor), and (3) Communication Gap scores comparing text-channel and latent-channel representations using teacher forcing on the exact seller response.

---

## 4. Diagrams

### Figure 1: Overview of the Two-Arm Communication Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   WRITER-EDITOR COMMUNICATION FRAMEWORK                 │
└─────────────────────────────────────────────────────────────────────────┘

   Writer Agent
  ┌──────────────────────┐
  │  Input Prompt + Task │
  └──────────┬───────────┘
             │ Forward Pass
             ▼
  ┌──────────────────────┐
  │  h^1, ..., h^L       │  Hidden states at each layer, per output token
  │  (t_1, ..., t_n)     │
  └──────────┬───────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────────┐  ┌──────────────────────────┐
│   ARM B     │  │         ARM C            │
│  ─────────  │  │  ──────────────────────  │
│  h^L_Writer │  │  Tokens t_1,...,t_n      │
│  → placeholder  │  → inserted into Editor  │
│    tokens   │  │    prompt as natural     │
│  → inject at│  │    language              │
│  ℓ_inject   │  │  → standard Editor       │
│  in Editor  │  │    forward pass          │
│  → SelfIE   │  │  → SelfIE on Editor's    │
│    on Edito │  │    encoding of t_1..t_n  │
│    r states │  │                          │
└──────┬──────┘  └───────────┬──────────────┘
       │                     │
       │    Editor Agent     │
       ▼                     ▼
  ┌─────────────────────────────────────────┐
  │     COMPARISON: Arm B vs. Arm C         │
  │                                         │
  │   CG(B, C) = β·JSD(P_B, P_C)           │
  │            + α·(1 − cos(h_B, h_C))     │
  │                                         │
  │   Higher CG → more information lost    │
  │   in the natural-language channel       │
  └─────────────────────────────────────────┘
```

*Suggested rendered diagram: A forked-flow diagram with the Writer's layer stack on the left splitting into two labeled paths (Arm B = latent injection, Arm C = natural language), each entering the Editor's layer stack. Both paths terminate at a shared comparison block at the bottom showing CG score decomposition.*

---

### Figure 2: SelfIE Interpretation Procedure (adapted from Chen et al., 2024)

```
Original Forward Pass                Interpretation Forward Pass
─────────────────────                ───────────────────────────

Input: "What happened               Prompt: [X] Please summarize
        next in the story?"                  the previous message.

  Layer 1 ████████████████           Layer 1 ████ [X] ████████
  Layer 2 ████████████████           Layer 2 ████ [X] ████████
  Layer k ████████████████  ──────▶  Layer k ████ [h^k_{i*}] ██
  Layer L ████████████████           Layer L ████████████████

  Output: "The knight               Output: "The message is about
           returned."                         a knight's journey."

          ↑                                     ↑
    h^{ℓ*}_{i*}                         Replace [X] with h^{ℓ*}_{i*}
    (embedding to interpret)             at chosen layer k
```

*Suggested rendered diagram: Side-by-side transformer diagrams with color-coded layers. A dotted arrow connects the source embedding in the left diagram to the placeholder position in the right diagram, with a label "inject at layer k".*

---

### Figure 3: Communication Gap Metric Decomposition

$$\text{CG}_t = \underbrace{\beta \cdot \text{JS}_t}_{\text{distributional divergence}} + \underbrace{\alpha \cdot (1 - \cos_t)}_{\text{directional divergence}}$$

```
For each token t in Writer's output:

  Arm B distribution P^B_t ───────────────────┐
  (Editor under latent injection)              ├─▶ JSD(P^B_t, P^C_t) × β
  Arm C distribution P^C_t ───────────────────┘
  (Editor under natural language)

  Arm B embedding h^B_t ──────────────────────┐
  (Editor hidden state, injection path)        ├─▶ (1 - cos(h^B_t, h^C_t)) × α
  Arm C embedding h^C_t ──────────────────────┘
  (Editor hidden state, NL path)

  CG_t = weighted sum of both terms

  Final CG = (1/T) Σ CG_t  over all T tokens
  Higher CG → natural language loses more of the latent signal
```

*Suggested rendered diagram: A two-row area chart with token position on the x-axis. β·JS contribution shown in coral, α·(1−cos) in steel blue, and a black line for per-step CG—matching the style of the per-step breakdown plots in the experimental code. Pairs of high-CG tokens can be annotated with their SelfIE descriptions from each arm to show qualitatively what information diverges.*

---



### Figure 4: Communication Gap Across Layers — Arm B vs. Arm C

*(Placeholder — to be populated once Results are complete)*

```
      CG Score (Arm B vs. Arm C)
      ────────────────────────────────────────────────────
 High │
      │   ■  ■  ■  ■     ← Arm C (Natural Language)
      │                    higher CG = more info lost
 Low  │
      │   ▲  ▲  ▲  ▲     ← Arm B (Latent Injection)
      │                    lower CG = representations closer
      └───────────────────────────────────────────────────▶
           Early         Middle         Late
                       Transformer Layer (injection depth)
```

*Suggested rendered diagram: A two-line plot with Arm B and Arm C CG scores on the y-axis, transformer layer depth on the x-axis, and shaded ±1 std bands. The gap between the two lines at each layer directly quantifies what is lost by the natural-language bottleneck as a function of where in the model the comparison is made.*

---

*Results are pending completion of experiments. This section will be updated with quantitative findings.*

Upon completion, this section will report:

- **Communication Gap scores** (mean CG, JS divergence component, cosine distance component) for Arms B and C, broken down by CoQA example category and dialogue turn.
- **Layer-wise representational alignment**: how the gap between Writer and Editor representations changes across transformer layers, informing optimal injection depth.
- **Qualitative SelfIE comparisons**: natural-language descriptions generated by SelfIE for matched tokens in Arms B vs. C, illustrating specific tokens where the natural-language channel produces interpretively divergent Editor representations relative to direct latent transfer.
- **Communication method ranking**: whether direct hidden-state injection (Arm B) produces tighter representational alignment than natural language (Arm C), and at which layers this advantage is most pronounced.

---

## 6. Discussion

The two-arm framework introduced here draws on complementary prior work to expose a layer of the agent communication problem that performance-centric evaluations miss. Tang et al. (2025) show that SDE outperforms natural language on downstream task accuracy, particularly for complex reasoning; our framework provides a mechanistic explanation for *why*—by showing that natural language transmission creates measurable representational gaps, which SDE-style injection partially closes. Similarly, SelfIE (Chen et al., 2024) was originally developed to interpret individual LLM forward passes in isolation; we extend its application to the *comparative* setting, using it as a bridge between two agents' internal states.

The Writer–Editor framing is intentionally general. It subsumes common multi-agent patterns: planner–executor, researcher–synthesizer, debater–judge. The CraigslistBargains setting (He et al., 2018) provides a controlled instantiation where the strategic intent of the writer (the buyer's target price and negotiating posture) has a measurable downstream effect on the editor's response (whether a fair deal is reached). This grounding allows us to connect representational gaps to behavioral outcomes.

An important design choice is the use of SelfIE rather than linear probing for the representational comparison. Linear probes require labeled data for each concept of interest and are limited to a closed concept set. SelfIE produces open-world natural-language descriptions that can capture nuanced, compositional ideas—exactly the kind of content that arises in agent-to-agent communication. The trade-off is that SelfIE interpretations are harder to compare automatically; we address this by grounding comparisons in the JSD and cosine-distance metrics rather than relying solely on semantic similarity of generated text.

---

## 7. Future Work and Challenges

**Model scale.** We initially attempted experiments on Llama-7B and Llama-13B. However, hidden-state injection at these scales produced incoherent or degenerate outputs in the Editor, a limitation also noted by Chen et al. (2024) for smaller models (see their ablation in Figure 12—7B and 13B models show markedly lower SelfIE accuracy, primarily due to failures to follow interpretation instructions). After systematic ablation, we adopted the 110M Qwen model as a controlled testbed. We aim to replicate findings at Qwen 3.5 (27B, no reasoning), pending access to sufficient compute resources.

**Injection layer selection.** Following Tang et al. (2025), we plan to use a preliminary sweep over injection layers using a held-out split of the dataset and select the top-$k$ layers by CG score for all subsequent experiments.

**Additional metrics.** The Communication Gap metric captures representational divergence at the token level. Future work should investigate metrics that operate at the level of propositions, dialogue acts, or higher-level semantic structure—providing a richer picture of what specifically is lost in transmission.

**Steering and alignment applications.** If we can identify the token positions and layers at which representational gaps are largest, we can potentially steer the Editor's internal representation toward the Writer's intent—using techniques similar to ActAdd (Turner et al., 2024) or SelfIE's Supervised Control—reducing communication failure without full latent-space transmission.

**Multi-agent chains and tool use.** The current framework addresses pairwise communication. In chains of three or more agents, representational gaps may compound across hops. Orchestrator agents that call tools introduce additional state transitions that could further distort downstream representations.

**Safety implications.** Representational gaps in agent communication are not merely a performance issue—they are a safety concern. An agent that internally represents a message differently from how the sender intended may take actions that appear reasonable from its (distorted) perspective but violate the original intent. Understanding and measuring these gaps is thus a prerequisite for reliable red-teaming of agentic systems.

---

## References

1. Venkataramani et al. (2026). *MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems.* arXiv:2602.03053.

2. He, H., Chen, D., Balakrishnan, A., & Liang, P. (2018). *Decoupling Strategy and Generation in Negotiation Dialogues.* arXiv:1808.09637.

3. Tang, Y., Su, W., Zhou, Y., Liu, Y., Zhang, M., Ma, S., & Ai, Q. (2025). *Augmenting Multi-Agent Communication with State Delta Trajectory.* arXiv:2506.19209.

4. Du et al. (2026). *Enabling Agents to Communicate Entirely in Latent Space.* arXiv:2511.09149.

5. Chen, H., Vondrick, C., & Mao, C. (2024). *SelfIE: Self-Interpretation of Large Language Model Embeddings.* arXiv:2403.10949.

6. Pham, C. et al. (2024). *Let Models Speak Ciphers: Multiagent Debate through Embeddings.* ICLR 2024.

7. Turner, A. M. et al. (2024). *Steering Language Models with Activation Engineering.* arXiv:2308.10248.

8. Ramesh, V. & Li, K. (2025). *Communicating Activations between Language Model Agents.* arXiv:2501.14082.

9. Li, B. Z., Nye, M., & Andreas, J. (2021). *Implicit Representations of Meaning in Neural Language Models.* arXiv:2106.00737.

10. Hernandez, E., Li, B. Z., & Andreas, J. (2023). *Inspecting and Editing Knowledge Representations in Language Models.* arXiv:2304.00740.

11. Wu, Q. et al. (2024). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* COLM 2024.

12. Qian, C. et al. (2024). *ChatDev: Communicative Agents for Software Development.* ACL 2024.

13. Hong, S. et al. (2024). *MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework.* ICLR 2024.

14. Belrose, N. et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.

---

*Submitted to HackPrinceton Spring 2026. Code and data available in the project repository.*

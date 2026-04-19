"""
Writer/editor hidden-state probe — minimum viable run.

    python demo.py

K2-V2-Instruct is 70B. bf16 needs ~140GB VRAM. The script uses
device_map="auto" so HF will shard across whatever GPUs are visible.
For 4-bit quantization on smaller GPUs, see the commented block below.
"""

import os
import sys
import time

import torch

from selfie_k2v2 import K2V2Backend, make_graph

_HERE = os.path.dirname(os.path.abspath(__file__))

BAR = "=" * 72
THIN = "-" * 72


def banner(title):
    print()
    print(BAR)
    print(f"  {title}")
    print(BAR)


def subbanner(title):
    print()
    print(THIN)
    print(f"  {title}")
    print(THIN)


def main():
    banner("STEP 1 / 4: Load K2-V2-Instruct")
    print("Loading LLM360/K2-V2-Instruct in bf16 with device_map='auto'.\n")

    t0 = time.time()
    backend = K2V2Backend(
        model_name="LLM360/K2-V2-Instruct",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model ready in {time.time() - t0:.1f}s.")
    print(f"  hidden_size = {backend.hidden_size}")
    print(f"  num_layers  = {backend.num_layers}")
    print(f"  device      = {backend.device}")

    # --- 4-bit alternative ----------------------------------------------------
    # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    # bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    #                          bnb_4bit_quant_type="nf4")
    # backend = K2V2Backend.__new__(K2V2Backend)
    # backend.tokenizer = AutoTokenizer.from_pretrained("LLM360/K2-V2-Instruct")
    # backend.model = AutoModelForCausalLM.from_pretrained(
    #     "LLM360/K2-V2-Instruct", device_map="auto", quantization_config=bnb)

    banner("STEP 2 / 4: Sanity check — plain generation")
    t0 = time.time()
    ids = backend.build_chat_prompt(
        system="You are K2, a helpful assistant.",
        user="Say hi in 5 words.",
        reasoning_effort="medium",
    )
    r = backend.generate(ids, max_new_tokens=20, capture_hidden=False)
    print(f"[took {time.time() - t0:.1f}s]")
    print(f"Output: {r.output_text!r}")

    if not r.output_text.strip():
        print("\n[!] Empty output. Check model load or chat template.")
        sys.exit(1)

    banner("STEP 3 / 4: Build graph and run 4-arm comparison + CG metric")
    print("Graph: writer → probe_editor (A) → probe_editor_selfhs (B)")
    print("                                  → probe_editor_writerhs (C)")
    print("                                  → probe_writer_selfhs  (D)")
    print("                                  → probe_comm_gap      (CG metric) → END")
    print("Cost: 5 generate() calls + 2 fast forward passes.\n")

    task = "Describe a golden retriever in 3 sentences."
    print(f"Task: {task!r}\n")

    graph = make_graph(
        backend, backend,
        max_writer_tokens=80, max_editor_tokens=60,
        inject_layer=0, verbose_timing=True,
    )

    t0 = time.time()
    final = graph.invoke({"task": task})
    print(f"\nFull graph run took {time.time() - t0:.1f}s.")

    banner("STEP 4 / 4: Four-arm comparison")
    subbanner("WRITER OUTPUT (clean — thinking stripped)")
    print(final["writer_output_text"])

    subbanner("Arm A — EDITOR VERDICT (raw text)")
    print(final["editor_verdict"])

    subbanner("Arm B — EDITOR VERDICT (editor's own hidden states re-injected)")
    print(final["editor_selfhs_verdict"])

    subbanner("Arm C — EDITOR VERDICT (writer's hidden states injected)")
    print(final["editor_writerhs_verdict"])

    subbanner("Arm D — WRITER SELF-PROBE (writer's hidden states → writer)")
    print(final["writer_selfhs_output"])

    print()
    print("A ≈ B       →  editor's internal repr and surface text carry the same info.")
    print("A ≈ C       →  writer's hidden states carry the same info as its text.")
    print("B ≈ C       →  writer and editor form compatible internal representations.")
    print("D ≈ answer  →  writer's HS genuinely encode the essay content.")

    cg = final.get("comm_gap")
    if cg and cg.get("T", 0) > 0:
        subbanner("Communication Gap (text channel vs. latent channel)")
        print(f"  response length T : {cg['T']} tokens")
        print(f"  alpha (cosine)    : {cg['alpha']}")
        print(f"  beta  (JS)        : {cg['beta']}")
        print(f"  CG                : {cg['CG']:.4f}")
        print(f"  mean JS           : {cg['JS_mean']:.4f}")
        print(f"  mean (1 - cos)    : {cg['COS_mean']:.4f}")
        print()
        print("Low CG  →  editor behaves similarly under raw text and latent injection.")
        print("High CG →  measurable communication gap between writer and editor.")

    out_dir = os.path.join(_HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    wr = final["writer_result"]
    with open(os.path.join(out_dir, "arm_verdicts.txt"), "w", encoding="utf-8") as f:
        f.write(f"task : {task}\n\n")
        if wr.thinking_text:
            f.write(f"=== Writer thinking ===\n{wr.thinking_text}\n\n")
        f.write(f"=== Writer output (clean) ===\n{final['writer_output_text']}\n\n")
        f.write(f"=== Arm A: editor_verdict (raw text) ===\n{final['editor_verdict']}\n\n")
        f.write(f"=== Arm B: editor_selfhs_verdict ===\n{final['editor_selfhs_verdict']}\n\n")
        f.write(f"=== Arm C: editor_writerhs_verdict ===\n{final['editor_writerhs_verdict']}\n\n")
        f.write(f"=== Arm D: writer_selfhs_output ===\n{final['writer_selfhs_output']}\n\n")
        if cg and cg.get("T", 0) > 0:
            f.write(f"=== Communication Gap ===\n")
            f.write(f"T         : {cg['T']}\n")
            f.write(f"alpha     : {cg['alpha']}\n")
            f.write(f"beta      : {cg['beta']}\n")
            f.write(f"CG        : {cg['CG']:.6f}\n")
            f.write(f"JS_mean   : {cg['JS_mean']:.6f}\n")
            f.write(f"COS_mean  : {cg['COS_mean']:.6f}\n")

    print(f"\nOutputs written to: {out_dir}/arm_verdicts.txt")
    banner("DONE")


if __name__ == "__main__":
    main()

"""
SELFIE on K2-V2 writer/editor agents — minimum viable run.

Run from inside the selfie_k2v2/ directory:

    python demo.py

Before first run:

    pip install 'transformers>=4.45,<4.60' accelerate bitsandbytes langgraph pandas torch

K2-V2-Instruct is 70B. bf16 needs ~140GB VRAM. The script uses
device_map="auto" so HF will shard across whatever GPUs are visible.
For 4-bit quantization on smaller GPUs, see the commented block below.
"""

import os
import sys
import time

import torch
import pandas as pd

from selfie_k2v2.k2v2_backend import K2V2Backend
from selfie_k2v2.agents_graph import make_graph

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---- pretty-print helpers ----
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


def timed(label):
    """Context manager-ish decorator used inline via start = time.time()."""
    pass  # kept for future use


def main():
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.width", 200)

    # ---------------------------------------------------------------------
    banner("STEP 1 / 6: Load K2-V2-Instruct")
    # ---------------------------------------------------------------------
    print("Loading LLM360/K2-V2-Instruct in bf16 with device_map='auto'.")
    print("First run will download ~140GB of weights to ~/.cache/huggingface/.")
    print("This can take several minutes even after weights are cached.\n")

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

    # --- If you need 4-bit on a smaller GPU, replace the K2V2Backend(...) call
    # --- above with something like:
    #
    # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    # bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    #                          bnb_4bit_quant_type="nf4")
    # backend = K2V2Backend.__new__(K2V2Backend)
    # backend.tokenizer = AutoTokenizer.from_pretrained("LLM360/K2-V2-Instruct")
    # backend.model = AutoModelForCausalLM.from_pretrained(
    #     "LLM360/K2-V2-Instruct", device_map="auto", quantization_config=bnb)
    # backend.model.eval()
    # backend.device = next(backend.model.parameters()).device
    # backend.hidden_size = backend.model.config.hidden_size
    # backend.num_layers = len(backend.model.model.layers)

    # ---------------------------------------------------------------------
    banner("STEP 2 / 6: Sanity check — plain generation")
    # ---------------------------------------------------------------------
    print("Running a tiny 'say hi' prompt to confirm the model decodes at all.\n")

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
        print("\n[!] Empty output from sanity check. Something is wrong with the ")
        print("    model load or chat template. Stopping before we waste time on ")
        print("    the full graph.")
        sys.exit(1)

    # ---------------------------------------------------------------------
    banner("STEP 3 / 6: Build graph and run writer → editor → 3 experiments")
    # ---------------------------------------------------------------------
    print("Graph structure:")
    print("  writer → editor → selfie_writer (Exp 1)")
    print("                 → selfie_editor  (Exp 3)")
    print("                 → injected_editor (Exp 2) → END")
    print()
    print("Caps: writer=40 tokens, editor=60 tokens (short on purpose).")
    print()
    task = "Describe a golden retriever in 3 sentences."
    print(f"Task: {task!r}")
    print()

    graph = make_graph(backend, max_writer_tokens=40, max_editor_tokens=60)

    t0 = time.time()
    final = graph.invoke({"task": task})
    print(f"\nFull graph run took {time.time() - t0:.1f}s.")

    # ---------------------------------------------------------------------
    banner("STEP 4 / 6: Agent outputs (text-level)")
    # ---------------------------------------------------------------------
    subbanner("WRITER OUTPUT")
    print(final["writer_output_text"])

    subbanner("EDITOR VERDICT — normal (sees writer's text)")
    print(final["editor_verdict"])

    subbanner("EDITOR VERDICT — injected (writer's hidden states at placeholders)")
    print(final["editor_injected_verdict"])

    print()
    print("Compare the two verdicts above.")
    print("  - If nearly identical: writer's final hidden states carry roughly the")
    print("    same info as its surface tokens.")
    print("  - If very different / injected is garbled: try inject_layer=1 or a")
    print("    middle layer in agents_graph.injected_editor_node.")

    # ---------------------------------------------------------------------
    banner("STEP 5 / 6: Experiment 1 — SELFIE on WRITER hidden states")
    # ---------------------------------------------------------------------
    print("Each row interprets the writer's hidden state at one (layer, token).")
    print("Probe layers were chosen at ~25%, 50%, 75% of the stack.")
    print()
    print(final["selfie_writer"].to_string(index=False))

    # ---------------------------------------------------------------------
    banner("STEP 6 / 6: Experiment 3 — SELFIE on EDITOR hidden states over draft span")
    # ---------------------------------------------------------------------
    print("Each row interprets the editor's hidden state at a position inside")
    print("the writer's quoted draft (the <DRAFT>...</DRAFT> span).")
    print()
    print(final["selfie_editor_on_draft"].to_string(index=False))

    # ---------------------------------------------------------------------
    banner("SIDE-BY-SIDE: writer_interp vs editor_interp at matched positions")
    # ---------------------------------------------------------------------
    print("Aligning both SELFIE tables by (layer, rel_pos) where rel_pos is the")
    print("offset within the writer's output span.\n")

    wr = final["writer_result"]
    writer_first = wr.prompt_len
    draft_first = final["editor_draft_start"]

    w_df = final["selfie_writer"].copy()
    w_df["rel_pos"] = w_df["token_idx"] - writer_first
    w_df = w_df.rename(columns={"token": "writer_tok", "interpretation": "writer_interp"})

    e_df = final["selfie_editor_on_draft"].copy()
    e_df["rel_pos"] = e_df["token_idx"] - draft_first
    e_df = e_df.rename(columns={"token": "editor_tok", "interpretation": "editor_interp"})

    merged = w_df.merge(
        e_df[["layer", "rel_pos", "editor_tok", "editor_interp"]],
        on=["layer", "rel_pos"],
        how="inner",
    )
    merged = merged[
        ["layer", "rel_pos", "writer_tok", "editor_tok",
         "writer_interp", "editor_interp"]
    ].sort_values(["layer", "rel_pos"]).reset_index(drop=True)

    print(merged.to_string(index=False))

    # Optional: write everything out as CSVs for later inspection.
    out_dir = os.path.join(_HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    final["selfie_writer"].to_csv(os.path.join(out_dir, "selfie_writer.csv"), index=False)
    final["selfie_editor_on_draft"].to_csv(os.path.join(out_dir, "selfie_editor_on_draft.csv"), index=False)
    merged.to_csv(os.path.join(out_dir, "merged_side_by_side.csv"), index=False)

    with open(os.path.join(out_dir, "texts.txt"), "w") as f:
        f.write("=== writer_output_text ===\n")
        f.write(final["writer_output_text"] + "\n\n")
        f.write("=== editor_verdict (normal) ===\n")
        f.write(final["editor_verdict"] + "\n\n")
        f.write("=== editor_injected_verdict ===\n")
        f.write(final["editor_injected_verdict"] + "\n")

    print()
    print(f"CSVs and raw texts written to: {out_dir}/")
    print("  selfie_writer.csv")
    print("  selfie_editor_on_draft.csv")
    print("  merged_side_by_side.csv")
    print("  texts.txt")

    banner("DONE")


if __name__ == "__main__":
    main()

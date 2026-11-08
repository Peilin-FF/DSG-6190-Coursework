"""Write the LaTeX tables of the report from results/predictions.parquet.

  python scripts/make_tables.py --results results --out report/tables
"""

import argparse
import os

import numpy as np
import pandas as pd

MODELS = [("qwen3-vl-4b", "Qwen3-VL-4B"), ("qwen3-vl-8b", "Qwen3-VL-8B"),
          ("internvl3.5-8b", "InternVL3.5-8B"), ("llava-ov-7b", "LLaVA-OV-7B")]
NAMES = {
    "blind": "Blind (no image)", "direct": "Direct", "cot": "CoT~\\cite{kojima2022zeroshot}",
    "ccot": "CCoT~\\cite{mitra2024ccot}", "dsg_schema": "+ Schema (one query)", "dsg_seq": "+ Grounding (independent)",
    "dsg_hier": "+ Hierarchy", "dsg_full": "DSG~\\cite{hsu2025maze}", "dsg_release": "DSG-release",
    "dsg_shuffle": "Shuffled grounding", "dsg_wrong": "Wrong schema", "dsg_generic": "Generic schema",
    "dsg_q": "DSG-Q (ours)", "dsg_sc3": "DSG-SC, $K{=}3$ (ours)", "dsg_gate": "DSG-Gate (ours)",
}
MAIN_RUNS = [0, 1, 2]


def acc_table(df, dataset, variants, models=MODELS, qtype=None):
    """mean and std over runs (std only when there are several runs) per (variant, model)."""
    d = df[(df.dataset == dataset) & df.variant.isin(variants) & df.run.isin(MAIN_RUNS)]
    if qtype is not None:
        d = d[d.qtype == qtype]
    cells = {}
    for (v, m), g in d.groupby(["variant", "model"]):
        per_run = g.groupby("run")["correct"].mean() * 100
        cells[(v, m)] = (per_run.mean(), per_run.std(ddof=1) if len(per_run) > 1 else None, len(per_run))
    return cells


def fmt(cell, best=False, ref=None, show_std=True):
    if cell is None:
        return "--"
    mean, std, _ = cell
    s = f"{mean:.1f}"
    if best:
        s = f"\\textbf{{{s}}}"
    if show_std and std is not None:
        s += f"\\std{{{std:.1f}}}"
    return s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def main_table(df, out, dataset, rows, label_groups):
    cells = acc_table(df, dataset, [r for r in rows if r != "|"])
    lines = []
    for r in rows:
        if r == "|":
            lines.append("\\midrule")
            continue
        vals = [cells.get((r, m)) for m, _ in MODELS]
        means = [c[0] for c in vals if c is not None]
        avg = f"{np.mean(means):.1f}" if len(means) == len(MODELS) else "--"
        row = [NAMES[r]] + [fmt(c) for c in vals] + [avg]
        lines.append(" & ".join(row) + " \\\\")
    write(out, "\n".join(lines) + "\n")
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/tables")
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    main_table(df, os.path.join(a.out, "vad_main.tex"), "vad",
               ["blind", "direct", "cot", "ccot", "|", "dsg_full", "dsg_release", "|", "dsg_q", "dsg_sc3", "dsg_gate"], None)
    main_table(df, os.path.join(a.out, "vad_ablation.tex"), "vad",
               ["direct", "dsg_schema", "dsg_seq", "dsg_hier", "dsg_full", "|", "dsg_shuffle", "dsg_wrong", "dsg_generic"], None)
    for ds in ("maze", "maze_nolegend", "ai2d"):
        main_table(df, os.path.join(a.out, f"{ds}_main.tex"), ds,
                   ["blind", "direct", "cot", "ccot", "|", "dsg_full", "|", "dsg_q", "dsg_sc3", "dsg_gate"], None)
    print("tables written to", a.out)


if __name__ == "__main__":
    main()

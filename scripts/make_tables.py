"""Write the LaTeX tables of the report from results/predictions.parquet (written by analyze.py).

  python scripts/make_tables.py --results results --out report/tables
"""

import argparse
import os

import numpy as np
import pandas as pd

MODELS = [("qwen3-vl-4b", "Qwen3-VL-4B", "Q-4B"), ("qwen3-vl-8b", "Qwen3-VL-8B", "Q-8B"),
          ("internvl3.5-8b", "InternVL3.5-8B", "IVL-8B"), ("llava-ov-7b", "LLaVA-OV-7B", "LO-7B")]
NAMES = {
    "blind": "Blind (no image)", "direct": "Direct", "cot": "CoT~\\cite{kojima2022zeroshot}",
    "ccot": "CCoT~\\cite{mitra2024ccot}", "dsg_schema": "+ Schema (one query)", "dsg_seq": "+ Grounding",
    "dsg_hier": "+ Hierarchy", "dsg_full": "DSG~\\cite{hsu2025maze}", "dsg_release": "DSG-release",
    "dsg_shuffle": "Shuffled grounding", "dsg_wrong": "Wrong schema", "dsg_generic": "Generic schema",
    "dsg_q": "DSG-Q (ours)", "dsg_x": "DSG-X (ours)", "dsg_sc3": "DSG-SC, $K{=}3$ (ours)", "dsg_gate": "DSG-Gate (ours)",
}
SHORT = dict(NAMES, dsg_full="Full DSG", cot="CoT", ccot="CCoT")
RUNS = [0, 1, 2]


FULL_N = {"vad": 540, "maze": 600, "maze_nolegend": 600, "ai2d": 556}


def cell(df, model, dataset, variant):
    """Mean/std over complete runs only; incomplete runs are left out (shown as -- when none is complete)."""
    d = df[(df.model == model) & (df.dataset == dataset) & (df.variant == variant) & df.run.isin(RUNS)]
    sizes = d.groupby("run")["id"].nunique()
    d = d[d.run.isin(sizes[sizes >= FULL_N[dataset]].index)]
    if d.empty:
        return None
    per = d.groupby("run")["correct"].mean() * 100
    return per.mean(), (per.std(ddof=1) if len(per) > 1 else None), int(d.groupby("run").size().min())


def fmt(c, bold=False, std=True):
    if c is None:
        return "--"
    s = f"{c[0]:.1f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if std and c[1] is not None:
        s += f"\\std{{{c[1]:.1f}}}"
    return s


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def model_table(df, dataset, rows, names, path, std=True, bold_skip=("blind",)):
    cells = {(r, m): cell(df, m, dataset, r) for r in rows if r != "|" for m, _, _ in MODELS}
    best = {}
    for m, _, _ in MODELS:
        vals = [cells[(r, m)][0] for r in rows if r != "|" and r not in bold_skip and cells[(r, m)] is not None]
        best[m] = max(vals) if vals else None
    lines = []
    for r in rows:
        if r == "|":
            lines.append("\\midrule")
            continue
        cs = [cells[(r, m)] for m, _, _ in MODELS]
        means = [c[0] for c in cs if c is not None]
        avg = f"{np.mean(means):.1f}" if len(means) == len(MODELS) else "--"
        lines.append(" & ".join([names[r]] + [fmt(c, bold=(c is not None and best[m] is not None and abs(c[0] - best[m]) < 1e-9
                                                         and r not in bold_skip), std=std)
                                              for c, (m, _, _) in zip(cs, MODELS)] + [avg]) + " \\\\")
    write(path, "\n".join(lines) + "\n")
    return cells


def canonical_table(df, rows, path):
    sets = [("maze", "Maze-Nav"), ("maze_nolegend", "Maze-Nav, no legend"), ("ai2d", "AI2D-Concepts")]
    head = ["\\begin{tabular}{l" + "cccc" * len(sets) + "}", "\\toprule",
            " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{s}}}" for _, s in sets) + " \\\\",
            "".join(f"\\cmidrule(lr){{{2 + 4 * k}-{5 + 4 * k}}}" for k in range(len(sets))),
            "Method & " + " & ".join(short for _ in sets for _, _, short in MODELS) + " \\\\", "\\midrule"]
    cells = {(r, ds, m): cell(df, m, ds, r) for r in rows if r != "|" for ds, _ in sets for m, _, _ in MODELS}
    best = {}
    for ds, _ in sets:
        for m, _, _ in MODELS:
            vals = [cells[(r, ds, m)][0] for r in rows if r not in ("|", "blind") and cells[(r, ds, m)] is not None]
            best[(ds, m)] = max(vals) if vals else None
    body = []
    for r in rows:
        if r == "|":
            body.append("\\midrule")
            continue
        vals = []
        for ds, _ in sets:
            for m, _, _ in MODELS:
                c = cells[(r, ds, m)]
                b = c is not None and r != "blind" and best[(ds, m)] is not None and abs(c[0] - best[(ds, m)]) < 1e-9
                vals.append(fmt(c, bold=b, std=False))
        body.append(" & ".join([SHORT[r]] + vals) + " \\\\")
    write(path, "\n".join(head + body + ["\\bottomrule", "\\end{tabular}"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/tables")
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    model_table(df, "vad", ["blind", "direct", "cot", "ccot", "|", "dsg_full", "dsg_release", "|",
                            "dsg_x", "dsg_q", "dsg_sc3", "dsg_gate"], NAMES, os.path.join(a.out, "vad_main.tex"))
    model_table(df, "vad", ["direct", "dsg_schema", "dsg_seq", "dsg_hier", "dsg_full", "|", "dsg_shuffle",
                            "dsg_wrong", "dsg_generic"], SHORT, os.path.join(a.out, "vad_ablation.tex"), std=False,
                bold_skip=("blind", "dsg_shuffle", "dsg_wrong", "dsg_generic"))
    canonical_table(df, ["blind", "direct", "cot", "ccot", "|", "dsg_full", "|", "dsg_x", "dsg_q", "dsg_sc3", "dsg_gate"],
                    os.path.join(a.out, "canonical.tex"))
    print("tables written to", a.out)


if __name__ == "__main__":
    main()

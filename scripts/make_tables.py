"""Write the LaTeX tables of the report from results/predictions.parquet (written by analyze.py).

  python scripts/make_tables.py --results results --out report/tables
"""

import argparse
import os

import numpy as np
import pandas as pd

MODELS = [("qwen3-vl-4b", "Qwen3-VL-4B", "Q-4B"), ("qwen3-vl-8b", "Qwen3-VL-8B", "Q-8B"),
          ("internvl3.5-8b", "InternVL3.5-8B", "IVL-8B"), ("qwen3.8-27b", "Qwen3.8-27B", "Q3.8-27B")]
NAMES = {  # row names follow Hsu et al. (2025): base VLM, language-only baseline, + schema / + grounding / + hierarchical
    "blind": "Language-only", "direct": "Base VLM", "cot": "CoT~\\cite{kojima2022zeroshot}",
    "ccot": "CCoT~\\cite{mitra2024ccot}", "dsg_schema": "+ schema", "dsg_seq": "+ grounding",
    "dsg_hier": "+ hierarchical", "dsg_full": "DSG~\\cite{hsu2025maze}", "dsg_release": "DSG, released code",
    "dsg_shuffle": "Groundings of another image", "dsg_wrong": "Schema of another concept", "dsg_generic": "Generic schema",
    "dsg_sc3": "DSG, vote over 3 schemas", "dsg_x": "+ exclusive grounding (ours)", "dsg_q": "+ question components (ours)",
}
SHORT = dict(NAMES, dsg_full="DSG", cot="CoT", ccot="CCoT")
# Concept names as written in Hsu et al. (2025), in the order of their four categories.
CONCEPTS = [("tic-tac-toe", "tic-tac-toe"), ("maze", "maze"), ("treasure map", "treasure map"),
            ("solar system", "solar system"), ("atom", "atom"), ("cell", "cell"), ("helping", "helping"),
            ("deceiving", "deceiving"), ("negotiating", "negotiating"),
            ("setting up a table for two", "table setting for two"), ("tidying up a guest room", "guest bedrooms"),
            ("putting up decorations on a door", "door decoration")]
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
    sets = [("maze", "Maze-Nav"), ("maze_nolegend", "Maze-Nav without legend"), ("ai2d", "AI2D")]
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


def concept_table(df, path):
    """Base VLM / DSG accuracy (%) per VAB concept and model; DSG is averaged over the three schema samples."""
    lines = []
    for key, name in CONCEPTS:
        vals = []
        for m, _, _ in MODELS:
            d = df[(df.model == m) & (df.dataset == "vad") & (df.concept == key)]
            base = d[(d.variant == "direct") & (d.run == 0)]["correct"].mean() * 100
            dsg = d[(d.variant == "dsg_full") & d.run.isin(RUNS)].groupby("run")["correct"].mean().mean() * 100
            vals.append(f"{base:.0f} / {dsg:.0f}")
        lines.append(" & ".join([name] + vals) + " \\\\")
    write(path, "\n".join(lines) + "\n")


def schema_table(schema_path, path):
    """Schema sample 0 of every VAB concept, components in dependency order with their parents."""
    import json
    with open(schema_path) as f:
        schemas = json.load(f)["schemas"]
    lines = ["\\begin{tabular}{p{0.24\\linewidth}p{0.70\\linewidth}}", "\\toprule",
             "Concept & Components of schema sample 0 ($s \\leftarrow$ parents)\\\\", "\\midrule"]
    for key, name in CONCEPTS:
        parts = []
        for term, deps in schemas[key][0]["components"]:
            parts.append(f"\\textit{{{term}}}" + (f" $\\leftarrow$ {', '.join(deps)}" if deps else ""))
        lines.append(f"{name} & " + "; ".join(parts) + " \\\\")
    write(path, "\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/tables")
    ap.add_argument("--schemas", default="schemas/qwen3.8-27b_c3.json")
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    model_table(df, "vad", ["blind", "direct", "cot", "ccot", "|", "dsg_release", "dsg_full", "dsg_sc3", "|",
                            "dsg_x", "dsg_q"], NAMES, os.path.join(a.out, "vad_main.tex"))
    model_table(df, "vad", ["direct", "dsg_schema", "dsg_seq", "dsg_hier", "dsg_full", "|", "dsg_generic",
                            "dsg_wrong", "dsg_shuffle"], SHORT, os.path.join(a.out, "vad_ablation.tex"), std=False,
                bold_skip=("blind", "dsg_shuffle", "dsg_wrong", "dsg_generic"))
    canonical_table(df, ["blind", "direct", "cot", "ccot", "|", "dsg_full", "dsg_sc3", "|", "dsg_x", "dsg_q"],
                    os.path.join(a.out, "canonical.tex"))
    concept_table(df, os.path.join(a.out, "vad_concepts.tex"))
    schema_table(a.schemas, os.path.join(a.out, "schemas.tex"))
    print("tables written to", a.out)


if __name__ == "__main__":
    main()

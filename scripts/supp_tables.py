"""Tables of the supplementary material, from results/predictions.parquet and the CSV files of the other scripts.

  python scripts/supp_tables.py --results results --out report/tables > results/supp_numbers.txt

Writes the row files
  supp_paper.tex       VAB accuracy by question type for the base VLM and DSG, next to the values of Hsu et al.
  supp_samples.tex     DSG accuracy for each schema sample next to the base VLM
  supp_maze_qtype.tex  Maze-Nav accuracy for each question type
  supp_turns.tex       predicted turn counts on Maze-Nav (mean, balanced accuracy, Spearman correlation)
  supp_yesno.tex       share of "yes" answers and accuracy on questions whose answer is "no" (VAB)
  supp_shared.tex      share of resolved schemas in which two or more components receive the same grounding
  supp_ai2d.tex        AI2D accuracy per diagram category, base VLM / DSG
and prints the numbers that the text of the supplementary material quotes.
Means over models are means of the unrounded per-model values, and values are rounded with numpy, as in
report_numbers.py.
"""

import argparse
import os

import numpy as np
import pandas as pd

MODELS = [("qwen3-vl-4b", "Qwen3-VL-4B"), ("qwen3-vl-8b", "Qwen3-VL-8B"), ("internvl3.5-8b", "InternVL3.5-8B"),
          ("qwen3.8-27b", "Qwen3.8-27B")]
FULL_N = {"vad": 540, "maze": 600, "maze_nolegend": 600, "ai2d": 556}
ONE_RUN = ("blind", "direct", "cot", "ccot")
SETS = [("vad", "VAB"), ("maze", "Maze-Nav"), ("maze_nolegend", "Maze-Nav without legend"), ("ai2d", "AI2D")]
QTYPES = [None, "counting", "binary", "open"]
# Hsu et al. (ICLR 2025), Table 1 (GPT-4o) and the table of the open-source LLaVA model in their appendix:
# accuracy (%) on VAB, mean over 5 runs; all questions, counting, binary-choice and open-ended questions.
REPORTED = [("GPT-4o", (66.4, 60.4, 65.5, 69.3), (73.0, 70.4, 69.0, 77.6)),
            ("LLaVA", (41.5, 34.4, 55.2, 32.1), (44.4, 34.4, 59.0, 35.4))]
MAZE_METHODS = [("direct", "Base VLM"), ("cot", "CoT"), ("dsg_full", "DSG"), ("dsg_q", "+ question components")]
MAZE_QTYPES = ["right_turns", "total_turns", "relative_position"]


def runs_of(variant):
    return [0] if variant in ONE_RUN else [0, 1, 2]


def per_run(df, model, dataset, variant, qtype=None, category=None):
    """Accuracy (%) of every complete run; incomplete runs are left out, as in the tables of the main text."""
    d = df[(df.model == model) & (df.dataset == dataset) & (df.variant == variant) & df.run.isin(runs_of(variant))]
    sizes = d.groupby("run")["id"].nunique()
    d = d[d.run.isin(sizes[sizes >= FULL_N[dataset]].index)]
    if qtype is not None:
        d = d[d.qtype == qtype]
    if category is not None:
        d = d[d.category == category]
    return d.groupby("run")["correct"].mean() * 100


def acc(df, *args, **kw):
    s = per_run(df, *args, **kw)
    return float(s.mean()) if len(s) else np.nan


def f1(v):
    return "--" if v is None or np.isnan(v) else f"{v:.1f}"


def write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def paper_table(df, path):
    """Rows: model and method (base VLM, DSG); columns: all, counting, binary-choice, open-ended questions."""
    def dsg_cells(d, gain):
        return [f"{f1(d[0])}\\up{{{gain:.1f}}}"] + [f1(v) for v in d[1:]]
    lines = ["\\multicolumn{6}{l}{\\emph{Reported by Hsu \\etal~\\cite{hsu2025maze}, mean over five runs}} \\\\"]
    for name, base, dsg in REPORTED:
        lines.append(" & ".join([name, "Base VLM"] + [f1(v) for v in base]) + " \\\\")
        lines.append(" & ".join(["", "DSG"] + dsg_cells(dsg, dsg[0] - base[0])) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{6}{l}{\\emph{Our reproduction, DSG as the mean over three schema samples}} \\\\"]
    rows = []
    for m, name in MODELS:
        b = [acc(df, m, "vad", "direct", qtype=q) for q in QTYPES]
        d = [acc(df, m, "vad", "dsg_full", qtype=q) for q in QTYPES]
        rows.append((b, d))
        lines.append(" & ".join([name, "Base VLM"] + [f1(v) for v in b]) + " \\\\")
        lines.append(" & ".join(["", "DSG"] + dsg_cells(d, round(d[0], 1) - round(b[0], 1))) + " \\\\")
    lines.append("\\midrule")
    mb = [float(np.mean([r[0][i] for r in rows])) for i in range(4)]
    md = [float(np.mean([r[1][i] for r in rows])) for i in range(4)]
    lines.append(" & ".join(["Mean of the four", "Base VLM"] + [f1(v) for v in mb]) + " \\\\")
    lines.append(" & ".join(["", "DSG"] + dsg_cells(md, round(md[0], 1) - round(mb[0], 1))) + " \\\\")
    write(path, lines)
    print("== VAB by question type (all, counting, binary, open): base VLM / DSG")
    for (m, _), (b, d) in zip(MODELS, rows):
        print(m, [round(v, 1) for v in b], [round(v, 1) for v in d])
    print("mean", [round(v, 1) for v in mb], [round(v, 1) for v in md])


def samples_table(df, path):
    lines = []
    print("\\n== DSG per schema sample")
    for ds, title in SETS:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for m, name in MODELS:
            base = acc(df, m, ds, "direct")
            per = per_run(df, m, ds, "dsg_full")
            cells = [f1(per.get(r, np.nan)) for r in (0, 1, 2)]
            std = f"\\std{{{per.std(ddof=1):.1f}}}" if len(per) > 1 else ""
            lines.append(" & ".join([name, f1(base)] + cells + [f1(per.mean()) + std]) + " \\\\")
            print(ds, m, "base", round(base, 1), "samples", [round(v, 1) for v in per], "std", round(float(per.std(ddof=1)), 1))
        if ds != SETS[-1][0]:
            lines.append("\\midrule")
    write(path, lines)


def maze_qtype_table(df, path):
    """One block per model; rows: methods; columns: right turns, total turns, exit position on the two maze sets."""
    lines = []
    print("\\n== Maze-Nav accuracy by question type (right turns, total turns, exit position)")
    for i, (m, name) in enumerate(MODELS):
        vals = {v: [acc(df, m, ds, v, qtype=q) for ds in ("maze", "maze_nolegend") for q in MAZE_QTYPES]
                for v, _ in MAZE_METHODS}
        best = [max(vals[v][k] for v, _ in MAZE_METHODS) for k in range(6)]
        if i:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{name}}}}} \\\\")
        for v, label in MAZE_METHODS:
            cells = [f"\\textbf{{{f1(x)}}}" if abs(x - best[k]) < 1e-9 else f1(x) for k, x in enumerate(vals[v])]
            lines.append(" & ".join([label] + cells) + " \\\\")
            print(m, v, [round(x, 1) for x in vals[v]])
    write(path, lines)


def turns_table(stats, path):
    s = stats[stats.stat == "maze_turns"].set_index(["model", "variant"])
    lines = []
    for m, name in MODELS:
        cells = []
        for col in ("mean_pred", "balanced_acc", "spearman"):
            cells += [f"{np.round(s.loc[(m, v), col], 2):.2f}" for v in ("direct", "dsg_full")]
        lines.append(" & ".join([name] + cells) + " \\\\")
    write(path, lines)
    print("\n== Maze-Nav turn counting: true mean count", sorted(set(s["mean_true"].round(2))))


def yesno_table(stats, path):
    s = stats[stats.stat == "vab_yes_no"].set_index(["model", "variant"])
    lines = []
    for m, name in MODELS:
        cells = []
        for col in ("yes_share", "acc_on_no"):
            cells += [f"{np.round(s.loc[(m, v), col], 2):.2f}" for v in ("direct", "dsg_full", "dsg_generic")]
        lines.append(" & ".join([name] + cells) + " \\\\")
    write(path, lines)
    print("== VAB yes/no questions per run:", int(s.loc[(MODELS[0][0], "direct"), "n"]))


def shared_table(same, path):
    s = same.set_index(["model", "dataset", "variant"])["same_grounding"] * 100
    lines = []
    for m, name in MODELS:
        cells = [f"{s.loc[(m, ds, v)]:.1f}" for ds in ("vad", "maze", "ai2d") for v in ("dsg_full", "dsg_x")]
        lines.append(" & ".join([name] + cells) + " \\\\")
    write(path, lines)
    same_mazes = all(abs(s.loc[(m, "maze", "dsg_full")] - s.loc[(m, "maze_nolegend", "dsg_full")]) < 1e-9
                     for m, _ in MODELS)
    print("== shared groundings identical on the two maze sets for DSG:", same_mazes)


AI2D_CATEGORIES = {  # AI2D category -> concept name given to the models (scripts/prepare_data.py)
    "foodChainsWebs": "food web", "lifeCycles": "life cycle", "moonPhaseEquinox": "moon phases and seasons",
    "photosynthesisRespiration": "photosynthesis and respiration", "rockCycle": "rock cycle",
    "partsOfTheEarth": "layers of the Earth", "waterCNPCycle": "water and nutrient cycle",
    "rockStrata": "rock strata", "solarSystem": "solar system", "eclipses": "eclipse",
    "faultsEarthquakes": "faults and earthquakes", "volcano": "volcano", "atomStructure": "atom",
}


def ai2d_table(df, path):
    """Base VLM / DSG accuracy (%) per AI2D category; DSG is the mean over the complete schema samples."""
    d = df[df.dataset == "ai2d"]
    sizes = d[(d.variant == "direct") & (d.run == 0) & (d.model == MODELS[0][0])].groupby("category")["id"].nunique()
    lines = []
    print("\n== AI2D per category: questions, base VLM / DSG per model, mean gain")
    for cat in sizes.sort_values(ascending=False, kind="stable").index:
        cells, gains = [], []
        for m, _ in MODELS:
            base = per_run(d, m, "ai2d", "direct", category=cat)
            dsg = per_run(d, m, "ai2d", "dsg_full", category=cat)
            cells.append(f"{base.mean():.0f} / {dsg.mean():.0f}")
            gains.append(dsg.mean() - base.mean())
        lines.append(" & ".join([AI2D_CATEGORIES[cat], str(int(sizes[cat]))] + cells) + " \\\\")
        print(cat, int(sizes[cat]), cells, round(float(np.mean(gains)), 1))
    write(path, lines)


def text_numbers(df):
    print("\n== CoT on the two maze sets: share of outputs cut at the token limit / share without a matching choice")
    for m, _ in MODELS:
        for ds in ("maze", "maze_nolegend"):
            d = df[(df.model == m) & (df.dataset == ds) & (df.variant == "cot") & (df.run == 0)]
            print(m, ds, round(float((d.finish == "length").mean()), 3),
                  round(float(d.match_rule.isin(["nomatch", "empty"]).mean()), 3))
    print("\n== share of outputs without a matching choice: maximum over models and test sets")
    for v in ("blind", "direct", "ccot", "dsg_schema", "dsg_full", "dsg_x", "dsg_q", "cot"):
        rates = []
        for m, _ in MODELS:
            for ds in FULL_N:
                d = df[(df.model == m) & (df.dataset == ds) & (df.variant == v) & df.run.isin(runs_of(v))]
                if len(d):
                    rates.append(float(d.match_rule.isin(["nomatch", "empty"]).mean()))
        print(v, round(max(rates), 3) if rates else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/tables")
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    stats = pd.read_csv(os.path.join(a.results, "answer_stats.csv"))
    same = pd.read_csv(os.path.join(a.results, "same_grounding.csv"))
    paper_table(df, os.path.join(a.out, "supp_paper.tex"))
    samples_table(df, os.path.join(a.out, "supp_samples.tex"))
    maze_qtype_table(df, os.path.join(a.out, "supp_maze_qtype.tex"))
    turns_table(stats, os.path.join(a.out, "supp_turns.tex"))
    yesno_table(stats, os.path.join(a.out, "supp_yesno.tex"))
    shared_table(same, os.path.join(a.out, "supp_shared.tex"))
    ai2d_table(df, os.path.join(a.out, "supp_ai2d.tex"))
    text_numbers(df)


if __name__ == "__main__":
    main()

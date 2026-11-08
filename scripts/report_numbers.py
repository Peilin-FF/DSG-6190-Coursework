"""Every number that the text of the report quotes, computed from the results with the rounding of the tables.

Means over models are means of the per-model accuracies (each the mean over the complete runs), rounded to one
decimal like the table entries, so that a quoted difference equals the difference of two table values.

  python scripts/report_numbers.py --results results --runs runs --data data > results/report_numbers.txt
"""

import argparse
import collections
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest

MODELS = ["qwen3-vl-4b", "qwen3-vl-8b", "internvl3.5-8b", "qwen3.8-27b"]
FULL_N = {"vad": 540, "maze": 600, "maze_nolegend": 600, "ai2d": 556}
ONE_RUN = ("blind", "direct", "cot", "ccot", "dsg_sc3", "dsg_sc5", "dsg_gate")


def acc(df, model, dataset, variant, qtype=None, concept=None):
    runs = [0] if variant in ONE_RUN else [0, 1, 2]
    d = df[(df.model == model) & (df.dataset == dataset) & (df.variant == variant) & df.run.isin(runs)]
    sizes = d.groupby("run")["id"].nunique()
    d = d[d.run.isin(sizes[sizes >= FULL_N[dataset]].index)]
    if qtype is not None:
        d = d[d.qtype == qtype]
    if concept is not None:
        d = d[d.concept == concept]
    return d.groupby("run")["correct"].mean().mean() * 100 if len(d) else np.nan


def row(df, dataset, variant, **kw):
    vals = [acc(df, m, dataset, variant, **kw) for m in MODELS]
    return [round(v, 1) for v in vals] + [round(float(np.mean(vals)), 1)]


def mcnemar(df, model, dataset, a, b, run):
    x = df[(df.model == model) & (df.dataset == dataset)]
    A = x[(x.variant == a) & (x.run == 0)].set_index("id")["correct"]
    B = x[(x.variant == b) & (x.run == run)].set_index("id")["correct"]
    j = A.to_frame("a").join(B.to_frame("b"), how="inner")
    n01, n10 = int(((~j.a) & j.b).sum()), int((j.a & ~j.b).sum())
    return binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", default=os.environ.get("DSG_RUNS", "runs"))
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    print("models:", MODELS, "| entries: per-model accuracies, then the mean over models")
    for ds in FULL_N:
        print(f"\n== {ds}")
        for v in ["blind", "direct", "cot", "ccot", "dsg_schema", "dsg_seq", "dsg_hier", "dsg_full", "dsg_release",
                  "dsg_generic", "dsg_wrong", "dsg_shuffle", "dsg_sc3", "dsg_x", "dsg_q"]:
            r = row(df, ds, v)
            if not all(np.isnan(r[:4])):
                print(f"{v:12s}", r)
    print("\n== VAB question types (mean over models)")
    for qt in ["counting", "binary", "open"]:
        print(qt, {v: row(df, "vad", v, qtype=qt)[-1] for v in ["direct", "cot", "dsg_schema", "dsg_full", "dsg_x", "dsg_q"]})
    print("\n== Maze question types (per model)")
    for ds in ["maze", "maze_nolegend"]:
        for qt in ["right_turns", "total_turns", "relative_position"]:
            print(ds, qt, {v: row(df, ds, v, qtype=qt)[:4] for v in ["direct", "dsg_full", "dsg_x", "dsg_q", "cot"]})
    print("\n== tic-tac-toe (VAB)")
    print({v: row(df, "vad", v, concept="tic-tac-toe") for v in ["direct", "dsg_full", "dsg_x"]})
    print("\n== McNemar p, DSG against the base VLM on VAB (schema samples 0, 1, 2)")
    for m in MODELS:
        print(m, [round(mcnemar(df, m, "vad", "direct", "dsg_full", r), 4) for r in (0, 1, 2)])
    st = pd.read_csv(os.path.join(a.results, "same_grounding.csv"))
    st = st[st.model.isin(MODELS)]
    print("\n== share of resolved schemas with a shared grounding (%)")
    print((st.pivot_table(index=["dataset", "variant"], columns="model", values="same_grounding")[MODELS] * 100).round(1).to_string())
    ans = pd.read_csv(os.path.join(a.results, "answer_stats.csv"))
    print("\n== answer statistics")
    print(ans[ans.model.isin(MODELS)].round(2).to_string(index=False))
    # AI2D groundings that are bare diagram labels (one or two letters, possibly "point j" or "label b")
    lab = re.compile(r"^(?:(?:point|label|letter)\s+)?[a-z]{1,2}\d?$")
    print("\n== AI2D groundings that are bare labels (%)")
    for m in MODELS:
        ans_g = []
        for p in glob.glob(os.path.join(a.runs, "main", m, "ai2d", "dsg_full", "run*.jsonl")):
            seen = set()
            for line in open(p):
                r = json.loads(line)
                if r["image_key"] in seen or not r.get("grounding"):
                    continue
                seen.add(r["image_key"])
                ans_g += [lab.match((g["answer"] or "").strip().lower()) is not None for g in r["grounding"]]
        print(m, round(100 * float(np.mean(ans_g)), 1) if ans_g else None)


if __name__ == "__main__":
    main()

"""Answer statistics used in the analysis of the report, from results/predictions.parquet and the manifests.

  Maze-Nav turn counting (right and total turns): mean predicted count, balanced accuracy over the true counts
  (mean recall per true count) and Spearman correlation between predicted and true counts, base VLM against DSG.
  VAB yes/no questions: share of "yes" answers and accuracy on questions whose answer is "no".

  python scripts/answer_stats.py --results results --data data
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ["qwen3-vl-4b", "qwen3-vl-8b", "internvl3.5-8b", "qwen3.8-27b"]
RUNS = {"direct": [0], "dsg_full": [0, 1, 2], "dsg_generic": [0]}


def manifest(root, name):
    with open(os.path.join(root, "manifests", name)) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def turn_counts(df, maze, model, variant):
    d = df[(df.model == model) & (df.dataset == "maze") & (df.variant == variant) & df.run.isin(RUNS[variant])
           & df.qtype.isin(["right_turns", "total_turns"])].dropna(subset=["pred_idx"])
    t = np.array([int(maze[i]["answer"]) for i in d.id])
    p = np.array([int(maze[i]["choices"][int(k)]) for i, k in zip(d.id, d.pred_idx)])
    bal = np.mean([np.mean(p[t == c] == c) for c in np.unique(t)])
    return {"mean_pred": p.mean(), "mean_true": t.mean(), "balanced_acc": bal, "spearman": spearmanr(p, t).correlation}


def yes_no(df, vad, model, variant):
    d = df[(df.model == model) & (df.dataset == "vad") & (df.qtype == "binary") & (df.variant == variant)
           & df.run.isin(RUNS[variant])].dropna(subset=["pred_idx"])
    d = d[np.array([{c.lower() for c in vad[i]["choices"]} == {"yes", "no"} for i in d.id], dtype=bool)]
    pred = np.array([vad[i]["choices"][int(k)].lower() for i, k in zip(d.id, d.pred_idx)])
    ans = np.array([vad[i]["answer"].lower() for i in d.id])
    return {"yes_share": np.mean(pred == "yes"), "acc_on_no": np.mean(pred[ans == "no"] == "no"), "n": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    maze, vad = manifest(a.data, "maze.jsonl"), manifest(a.data, "vad.jsonl")
    rows = []
    for m in MODELS:
        for v in ("direct", "dsg_full"):
            rows.append({"stat": "maze_turns", "model": m, "variant": v, **turn_counts(df, maze, m, v)})
        for v in ("direct", "dsg_full", "dsg_generic"):
            rows.append({"stat": "vab_yes_no", "model": m, "variant": v, **yes_no(df, vad, m, v)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.results, "answer_stats.csv"), index=False, float_format="%.4f")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

"""Aggregate run files into tidy tables with confidence intervals, paired tests and offline variants.

  python scripts/analyze.py --runs $DSG_RUNS --out results

Offline variants (no extra model calls):
  dsg_sc{K}   majority vote of Full DSG over K schema samples (ties: the answer with the higher mean
              token log-probability)
  dsg_gate    Direct and Full DSG answer; when they disagree, keep the answer whose mean token
              log-probability is higher (one value per schema sample)
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from dsg.stats import cluster_bootstrap_ci, mcnemar  # noqa: E402

KEEP = ["id", "dataset", "method", "tag", "run", "concept", "category", "qtype", "image_key", "order",
        "answer_idx", "pred_idx", "match_rule", "correct", "n_choices", "lp_mean", "lp_sum", "error", "finish"]


def load_runs(root: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(root, "*", "*", "*", "*", "run*.jsonl")):
        suite, model, dataset, variant = path.split(os.sep)[-5:-1]
        if suite.startswith("smoke"):
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = {k: r.get(k) for k in KEEP}
                u = r.get("usage") or {}
                ground = r.get("grounding") or []
                aux = [x for x in (r.get("aux_usage") or []) if x]
                d.update(suite=suite, model=model, variant=variant,
                         prompt_tokens=(u.get("prompt") or 0) + sum((g.get("usage") or {}).get("prompt", 0) for g in ground)
                         + sum(x.get("prompt", 0) for x in aux),
                         completion_tokens=(u.get("completion") or 0)
                         + sum((g.get("usage") or {}).get("completion", 0) for g in ground)
                         + sum(x.get("completion", 0) for x in aux),
                         n_calls=1 + len(ground) + len(aux), n_ground=len(ground))
                rows.append(d)
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(bool)
    df = df.drop_duplicates(subset=["suite", "model", "dataset", "variant", "run", "id"], keep="last")
    return df


def _vote(preds: list, lps: list):
    votes = Counter(p for p in preds if p is not None)
    if not votes:
        return None
    top = max(votes.values())
    tied = [p for p, v in votes.items() if v == top]
    if len(tied) == 1:
        return tied[0]
    best = {}
    for p, lp in zip(preds, lps):
        if p in tied:
            best[p] = max(best.get(p, -1e9), lp if lp is not None else -1e9)
    return max(tied, key=lambda p: best[p])


def add_offline(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    full = df[(df.variant == "dsg_full")]
    for (model, dataset), g in full.groupby(["model", "dataset"]):
        runs = sorted(g.run.unique())
        piv_p = g.pivot_table(index="id", columns="run", values="pred_idx", aggfunc="first", dropna=False)
        piv_l = g.pivot_table(index="id", columns="run", values="lp_mean", aggfunc="first", dropna=False)
        present = g.groupby("id")["run"].apply(set)
        meta = g.drop_duplicates("id").set_index("id")
        for k in (3, 5):
            if len(runs) < k:
                continue
            use = runs[:k]
            for i in present.index:
                if not set(use) <= present[i]:
                    continue
                ps = [None if pd.isna(piv_p.at[i, r]) else int(piv_p.at[i, r]) for r in use]
                ls = [None if pd.isna(piv_l.at[i, r]) else float(piv_l.at[i, r]) for r in use]
                p = _vote(ps, ls)
                m = meta.loc[i]
                out.append(dict(m.to_dict(), id=i, model=model, dataset=dataset, suite="offline",
                                variant=f"dsg_sc{k}", run=0, pred_idx=p, correct=p == m["answer_idx"]))
    direct = df[(df.variant == "direct") & (df.run == 0)].set_index(["model", "dataset", "id"])
    for (model, dataset, run), g in full.groupby(["model", "dataset", "run"]):
        for _, r in g.iterrows():
            key = (model, dataset, r["id"])
            if key not in direct.index:
                continue
            d = direct.loc[key]
            if d["pred_idx"] == r["pred_idx"] or pd.isna(d["lp_mean"]) or pd.isna(r["lp_mean"]):
                p = r["pred_idx"]
            else:
                p = d["pred_idx"] if d["lp_mean"] > r["lp_mean"] else r["pred_idx"]
            out.append(dict(r.to_dict(), suite="offline", variant="dsg_gate", pred_idx=p,
                            correct=(p == r["answer_idx"])))
    return pd.concat([df, pd.DataFrame(out)], ignore_index=True) if out else df


def summarize(df: pd.DataFrame, by: list) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by):
        per_run = g.groupby("run")["correct"].mean()
        clusters = (g["image_key"].astype(str) + "#" + g["run"].astype(str)).tolist()
        lo, hi = cluster_bootstrap_ci(g["correct"].tolist(), clusters)
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), acc=per_run.mean(),
                        std=per_run.std(ddof=1) if len(per_run) > 1 else 0.0, runs=len(per_run),
                        n=int(g.groupby("run").size().max()), ci_lo=lo, ci_hi=hi,
                        nomatch=float(g["pred_idx"].isna().mean()),
                        calls=g["n_calls"].mean() if "n_calls" in g else np.nan,
                        prompt_tokens=g["prompt_tokens"].mean() if "prompt_tokens" in g else np.nan))
    return pd.DataFrame(out)


def paired(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Exact McNemar test of variant b against variant a (run 0 of a against each run of b)."""
    out = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        ga = g[(g.variant == a) & (g.run == 0)].set_index("id")["correct"]
        for r, gb in g[g.variant == b].groupby("run"):
            gb = gb.set_index("id")["correct"]
            ids = ga.index.intersection(gb.index)
            if len(ids) == 0:
                continue
            t = mcnemar(ga.loc[ids].tolist(), gb.loc[ids].tolist())
            out.append(dict(model=model, dataset=dataset, a=a, b=b, run=r, n=len(ids),
                            acc_a=ga.loc[ids].mean(), acc_b=gb.loc[ids].mean(), **t))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.environ.get("DSG_RUNS", "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = add_offline(load_runs(a.runs))
    df.to_parquet(os.path.join(a.out, "predictions.parquet"))
    overall = summarize(df, ["model", "dataset", "variant"])
    overall.to_csv(os.path.join(a.out, "overall.csv"), index=False)
    for col in ("qtype", "category", "concept"):
        summarize(df, ["model", "dataset", "variant", col]).to_csv(os.path.join(a.out, f"by_{col}.csv"), index=False)
    tests = [paired(df, "direct", v) for v in sorted(df.variant.unique()) if v != "direct"]
    tests = [t for t in tests if len(t)]
    if tests:
        pd.concat(tests).to_csv(os.path.join(a.out, "paired_vs_direct.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_columns", 20):
        print(overall.pivot_table(index=["dataset", "variant"], columns="model", values="acc").round(3))


if __name__ == "__main__":
    main()

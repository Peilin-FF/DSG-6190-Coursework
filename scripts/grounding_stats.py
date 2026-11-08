"""Statistics of the resolved schemas, read from the run files (runs/<suite>/<model>/<dataset>/<variant>/run<k>.jsonl).

For every model, test set and schema-based method it reports
  same_grounding: share of image-schema pairs in which two or more components receive the same grounding
                  (after lower-casing and removing articles and punctuation), mean over runs;
  and, for VAB, the accuracy on each concept (mean over runs).

  python scripts/grounding_stats.py --runs runs --out results
"""

import argparse
import collections
import glob
import json
import os
import re

import numpy as np

SUITES = ("main", "improve")  # the hyper-parameter suite repeats DSG for two models only
ARTICLES = re.compile(r"^(the|a|an)\s+")


def norm(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", (s or "").strip().lower())
    return ARTICLES.sub("", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.environ.get("DSG_RUNS", "runs"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    same = collections.defaultdict(list)    # (model, dataset, variant) -> per-run rates
    concept = collections.defaultdict(list)  # (model, variant, concept) -> per-run accuracies
    for suite in SUITES:
        for path in sorted(glob.glob(os.path.join(a.runs, suite, "*", "*", "*", "run*.jsonl"))):
            model, dataset, variant = path.split(os.sep)[-4:-1]
            pairs, correct = {}, collections.defaultdict(list)
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("grounding"):
                        pairs.setdefault(r["image_key"], [norm(g["answer"]) for g in r["grounding"]])
                    if dataset == "vad":
                        correct[r["concept"]].append(bool(r["correct"]))
            if pairs:
                rate = np.mean([len(set(g)) < len(g) for g in pairs.values()])
                same[(model, dataset, variant)].append(rate)
            for c, v in correct.items():
                concept[(model, variant, c)].append(np.mean(v))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "same_grounding.csv"), "w") as f:
        f.write("model,dataset,variant,runs,same_grounding\n")
        for (m, d, v), rates in sorted(same.items()):
            f.write(f"{m},{d},{v},{len(rates)},{np.mean(rates):.4f}\n")
    with open(os.path.join(a.out, "vab_concepts.csv"), "w") as f:
        f.write("model,variant,concept,runs,accuracy\n")
        for (m, v, c), accs in sorted(concept.items()):
            f.write(f"{m},{v},{c},{len(accs)},{np.mean(accs):.4f}\n")
    print("wrote same_grounding.csv and vab_concepts.csv to", a.out)


if __name__ == "__main__":
    main()

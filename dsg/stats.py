"""Statistics: cluster bootstrap confidence intervals and exact McNemar tests."""

from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def cluster_bootstrap_ci(correct: list, clusters: list, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of the mean, resampling whole clusters (questions that share an image)."""
    groups = defaultdict(list)
    for c, k in zip(correct, clusters):
        groups[k].append(float(c))
    sums = np.array([sum(v) for v in groups.values()])
    cnts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def mcnemar(a: list, b: list) -> dict:
    """Exact McNemar test on paired binary outcomes a, b (same questions, same order)."""
    b_only = sum(1 for x, y in zip(a, b) if not x and y)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": float(p)}


def mean_std(values: list) -> tuple:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

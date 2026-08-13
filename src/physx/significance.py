"""significance.py — paired significance tests over the multi-seed matrix.

The matrix trains, for each domain x kind x seed (0..4), a model with an
identical budget (256 samples, 60 epochs, same arch). Because every seed
controls data generation, shuffling, AND weight init, the runs are proper
statistical replicates, so per-domain comparisons between kinds are PAIRED
across seeds.

For each domain we test, on the 5 paired runs:
    phys   vs nophys   (does the physics term matter?)
    phys   vs mlp      (does attention + physics beat the MLP?)
    nophys vs mlp      (does attention alone beat the MLP?)
using the two-sided Wilcoxon signed-rank test (non-parametric, appropriate
for n = 5) on BOTH the answer error and the physics residual, plus Cliff's
delta as a paired effect size. We also pool all domains (5 x 5 = 25 pairs)
for an overall test.

usage: python physx/significance.py [--out paper/fig/significance.json]
"""

import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_DIR = os.path.join(ROOT, "physx", "models")

DOMAINS = ["beam", "cantilever", "projectile", "burgers", "heat2d"]
KINDS = ["phys", "nophys", "mlp"]
SEEDS = [0, 1, 2, 3, 4]
METRICS = ["val_rel_mae", "phys_resid"]

LABELS = {
    "val_rel_mae": "answer error",
    "phys_resid": "physics residual",
    "phys": "physics on",
    "nophys": "physics off",
    "mlp": "MLP",
}


def parse_log(path):
    """Last training line: val_rel_mae X phys_resid Y -> (X, Y) or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            last = None
            for line in f:
                m = re.search(r"val_rel_mae ([0-9.eE+-]+) phys_resid ([0-9.eE+-]+)", line)
                if m:
                    last = (float(m.group(1)), float(m.group(2)))
            return last
    except OSError:
        return None


def load_matrix():
    """results[domain][kind][seed] = {metric: value}."""
    out = {}
    for d in DOMAINS:
        out[d] = {}
        for k in KINDS:
            out[d][k] = {}
            for s in SEEDS:
                log = os.path.join(LOG_DIR, f"matrix_{d}_{k}_s{s}.log")
                parsed = parse_log(log)
                if parsed:
                    out[d][k][s] = {"val_rel_mae": parsed[0], "phys_resid": parsed[1]}
    return out


def cliff_delta(a, b):
    """Paired Cliff's delta: P(a > b) - P(a < b) over the pairs. In [-1, 1];
    > 0 means a is larger (worse for error metrics)."""
    n = len(a)
    if n == 0:
        return 0.0
    gt = sum(1 for x, y in zip(a, b) if x > y)
    lt = sum(1 for x, y in zip(a, b) if x < y)
    return (gt - lt) / n


def wilcoxon_p(a, b):
    """Two-sided Wilcoxon signed-rank test on paired samples."""
    from scipy.stats import wilcoxon
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0, 0
    if len(d) < 5:
        # n = 5: exact sign-permutation null distribution (exhaustive 2^n)
        from itertools import product
        ranks = np.argsort(np.argsort(np.abs(d))) + 1
        W_obs = float(np.sum(ranks * np.sign(d)))
        count = 0
        total = 0
        for signs in product([-1, 1], repeat=len(d)):
            W = float(np.sum(ranks * np.array(signs)))
            total += 1
            if abs(W) >= abs(W_obs):
                count += 1
        return count / total, len(d)
    res = wilcoxon(d, alternative="two-sided")
    return float(res.pvalue), len(d)


def stats_of(xs):
    mu = float(np.mean(xs))
    sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return {"mean": mu, "std": sd, "values": xs, "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "fig", "significance.json"))
    args = ap.parse_args()

    data = load_matrix()

    def complete(d):
        """A domain is complete only when every kind x seed has a final epoch."""
        return all(s in data[d][k] for k in KINDS for s in SEEDS)

    # per-domain paired tests
    tests = {}
    for d in DOMAINS:
        tests[d] = {}
        if not complete(d):
            tests[d]["complete"] = False
            continue
        tests[d]["complete"] = True
        for metric in METRICS:
            tests[d][metric] = {}
            for (a, b) in [("phys", "nophys"), ("phys", "mlp"), ("nophys", "mlp")]:
                va = [data[d][a][s][metric] for s in SEEDS if s in data[d][a] and s in data[d][b]]
                vb = [data[d][b][s][metric] for s in SEEDS if s in data[d][a] and s in data[d][b]]
                p, n_used = wilcoxon_p(va, vb)
                tests[d][metric][f"{a}_vs_{b}"] = {
                    "p": p, "n": n_used, "cliff_delta": cliff_delta(va, vb),
                    "median_ratio": float(np.median(vb) / np.median(va)) if np.median(va) else None,
                    "a_better": bool(np.median(va) < np.median(vb)),
                }

    # pooled across complete domains (5 x 5 = 25 pairs per comparison)
    pooled = {}
    for metric in METRICS:
        pooled[metric] = {}
        for (a, b) in [("phys", "nophys"), ("phys", "mlp"), ("nophys", "mlp")]:
            va, vb = [], []
            for d in DOMAINS:
                if not complete(d):
                    continue
                for s in SEEDS:
                    if s in data[d][a] and s in data[d][b]:
                        va.append(data[d][a][s][metric])
                        vb.append(data[d][b][s][metric])
            if len(va) < 3:
                continue
            p, n_used = wilcoxon_p(va, vb)
            pooled[metric][f"{a}_vs_{b}"] = {
                "p": p, "n": n_used, "cliff_delta": cliff_delta(va, vb),
                "median_ratio": float(np.median(vb) / np.median(va)) if np.median(va) else None,
                "a_better": bool(np.median(va) < np.median(vb)),
            }

    # summary stats (mean +/- std over seeds) for the tables
    summary = {}
    for d in DOMAINS:
        summary[d] = {}
        for k in KINDS:
            summary[d][k] = {m: stats_of([data[d][k][s][m] for s in SEEDS if s in data[d][k]])
                             for m in METRICS}

    out = {"domains": DOMAINS, "kinds": KINDS, "seeds": SEEDS,
           "summary": summary, "per_domain_tests": tests, "pooled_tests": pooled}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    # human-readable table
    def stars(p):
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"{'domain':12s} {'metric':16s} {'phys(mean±sd)':18s} {'nophys':18s} {'mlp':18s} "
          f"{'p(ph-v-np)':10s} {'p(ph-v-mlp)':11s}")
    for d in DOMAINS:
        if not complete(d):
            print(f"{d:12s} (incomplete matrix — rerun significance.py when done)")
            continue
        for metric in METRICS:
            pv = tests[d][metric]["phys_vs_nophys"]["p"]
            pm = tests[d][metric]["phys_vs_mlp"]["p"]
            def fmt(k):
                st = summary[d][k][metric]
                return f"{st['mean']:.3f}±{st['std']:.3f}"
            print(f"{d:12s} {LABELS[metric]:16s} {fmt('phys'):18s} {fmt('nophys'):18s} "
                  f"{fmt('mlp'):18s} {pv:5.3f}{stars(pv):4s} {pm:5.3f}{stars(pm):4s}")
    print("\npooled (25 pairs):")
    for metric in METRICS:
        for (a, b) in [("phys", "nophys"), ("phys", "mlp"), ("nophys", "mlp")]:
            t = pooled[metric][f"{a}_vs_{b}"]
            print(f"  {LABELS[metric]:16s} {LABELS[a]:12s} vs {LABELS[b]:11s} "
                  f"p={t['p']:.4f} {stars(t['p'])}  Cliff={t['cliff_delta']:+.2f}  "
                  f"median_ratio={t['median_ratio']:.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

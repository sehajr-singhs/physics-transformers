"""lca_significance.py — aggregate the Law-Conditioned Attention experiment
and test the conditioning effect.

Pairs: for every (domain, seed) replicate we have the LCA generalist (real
law signature) and the dummy-law ablation (identical architecture, constant
signature). The paired Wilcoxon signed-rank test asks whether the conditioning
stream moves held-out answer error / curve error / physics residual, and
Cliff's delta reports the effect size. Specialists (per-domain single-law
models) are reported per domain for reference.

usage: python physx/lca_significance.py [--models physx/models]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

DOMAINS = ["beam", "cantilever", "projectile", "pendulum", "spring", "rc"]


def cliff_delta(a, b):
    """Cliff's delta for paired samples (a vs b); positive = a smaller."""
    d = np.asarray(a) - np.asarray(b)
    return float(np.mean(np.sign(-d)))


def load_evals(models_dir):
    real, dummy = {}, {}
    for p in sorted(glob.glob(os.path.join(models_dir, "lca_real_s*.eval.json"))):
        with open(p) as f:
            ev = json.load(f)
        real[ev["seed"]] = ev["generalist"]
    for p in sorted(glob.glob(os.path.join(models_dir, "lca_dummy_s*.eval.json"))):
        with open(p) as f:
            ev = json.load(f)
        dummy[ev["seed"]] = ev["generalist"]
    return real, dummy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    real, dummy = load_evals(args.models)
    seeds = sorted(set(real) & set(dummy))
    if not seeds:
        print("no completed seed pairs yet", flush=True)
        return 1

    # pooled pairs (domain, seed)
    pairs = [(d, s) for d in DOMAINS for s in seeds]
    pooled = {m: {"real": [], "dummy": []} for m in ("ans_rel_mae", "curve_err", "phys_resid")}
    per_domain = {d: {m: {"real": [], "dummy": []} for m in pooled} for d in DOMAINS}
    for d, s in pairs:
        r, du = real[s][d], dummy[s][d]
        for m in pooled:
            pooled[m]["real"].append(r[m])
            pooled[m]["dummy"].append(du[m])
            per_domain[d][m]["real"].append(r[m])
            per_domain[d][m]["dummy"].append(du[m])

    out = {"seeds": seeds, "n_pairs": len(pairs), "pooled": {}, "per_domain": {},
           "specialists": {}}
    for m in pooled:
        a, b = np.array(pooled[m]["real"]), np.array(pooled[m]["dummy"])
        try:
            w, p = stats.wilcoxon(a, b)
        except ValueError:
            w, p = float("nan"), float("nan")
        med_ratio = float(np.median(a) / max(np.median(b), 1e-12))
        out["pooled"][m] = {
            "wilcoxon": float(w), "p": float(p),
            "cliffs_delta": cliff_delta(a, b),
            "median_real": float(np.median(a)), "median_dummy": float(np.median(b)),
            "median_ratio": med_ratio,
            "real_mean": float(np.mean(a)), "dummy_mean": float(np.mean(b)),
        }
    for d in DOMAINS:
        out["per_domain"][d] = {}
        for m in pooled:
            a, b = np.array(per_domain[d][m]["real"]), np.array(per_domain[d][m]["dummy"])
            if len(a) >= 3:
                try:
                    w, p = stats.wilcoxon(a, b)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            out["per_domain"][d][m] = {
                "p": float(p),
                "median_real": float(np.median(a)) if len(a) else None,
                "median_dummy": float(np.median(b)) if len(b) else None,
            }
    # specialists (per-domain, mean over whatever seeds reported them)
    specs = {}
    for p in sorted(glob.glob(os.path.join(args.models, "lca_real_s*.eval.json"))):
        with open(p) as f:
            ev = json.load(f)
        for d, r in ev.get("specialists", {}).items():
            if r is None:
                continue
            specs.setdefault(d, []).append(r)
    for d, rows in specs.items():
        out["specialists"][d] = {m: float(np.mean([r[m] for r in rows]))
                                 for m in ("ans_rel_mae", "curve_err", "phys_resid")}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print("=" * 70)
    print(f"LCA significance (paired Wilcoxon, {len(pairs)} pairs across {len(seeds)} seeds)")
    print("=" * 70)
    for m in pooled:
        r = out["pooled"][m]
        print(f"\n{m}:")
        print(f"  real  median {r['median_real']:.4f}   dummy median {r['median_dummy']:.4f}"
              f"   ratio {r['median_ratio']:.2f}x")
        print(f"  Wilcoxon p = {r['p']:.4f}   Cliff's delta = {r['cliffs_delta']:+.3f}")
    print("\nper-domain median answer error (real | dummy | p):")
    for d in DOMAINS:
        r = out["per_domain"][d]["ans_rel_mae"]
        print(f"  {d:11s} {r['median_real']:.3f} | {r['median_dummy']:.3f} | p={r['p']:.3f}")
    print("\nspecialists (single-law, mean):")
    for d in DOMAINS:
        r = out["specialists"].get(d)
        print(f"  {d:11s} ans {r['ans_rel_mae']:.3f} curve {r['curve_err']:.3f}"
              f" phys {r['phys_resid']:.4f}" if r else f"  {d}: n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())

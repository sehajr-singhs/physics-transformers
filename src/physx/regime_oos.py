"""regime_oos.py — out-of-sample test of the regime theory on 10 laws.

The hypothesis was PRE-REGISTERED (models/ext/pre_registration.json) before any
10-law training: LCA benefit is monotone non-decreasing in equation ambiguity,
where ambiguity is computed from the physical-quantity token vocabulary alone.

This script loads the six trained generalists (real/dummy x 3 seeds), computes
the per-law benefit, and tests the pre-registered predictions:

  1. rho(ambiguity, benefit) > 0 over the 9 non-degenerate laws, exact
     permutation p.
  2. Leave-one-law-out: the benefit rank of each held-out law is predicted
     from its ambiguity rank alone; predictive rho over the 9 predictions.
  3. The two laws whose ambiguity RISES when kepler/damped enter the set
     (pendulum 0.5->1.0, spring 0.0->1.0) show positive benefit in the
     10-law model.
  4. Group means are ordered by ambiguity: 1.0 > 0.75 > 0.667 > 0.5.

usage: python physx/regime_oos.py [--out paper/fig/regime_oos.json]
"""

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physx import laws

DOMAINS = laws.SHARED_HEAD_DOMAINS
DEGENERATE = {"rc"}  # parameter-invariant normalized trajectory


def ambiguity_10(domain):
    """Equation ambiguity over the 10-law set (from the vocabulary alone)."""
    mine = set(laws.DOMAIN_QUANTITIES[domain])
    best = 0.0
    for other in DOMAINS:
        if other == domain:
            continue
        theirs = set(laws.DOMAIN_QUANTITIES[other])
        best = max(best, len(mine & theirs) / len(mine))
    return best


def spearman_ranks(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def _rho(rx, ry):
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def spearman_exact(x, y):
    """Spearman rho with an exact permutation p-value (n <= 9).

    Vectorized: for a permutation test the denominator is constant across
    permutations, so only the centered cross-products matter, and all 9!
    permutations can be scored with a single matmul.
    """
    import itertools
    rx = spearman_ranks(x)
    ry = spearman_ranks(y)
    rho_obs = _rho(rx, ry)
    n = len(x)
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = np.sqrt((rx_c ** 2).sum() * (ry_c ** 2).sum())
    s_obs = abs(float((rx_c * ry_c).sum()) / denom)
    perms = np.asarray(list(itertools.permutations(range(n))), dtype=np.float32)
    # correlation of rx_c with each permuted ry == correlation of ry_c with
    # the permuted rx_c (permutation applied to one side)
    s = perms @ rx_c / denom
    count = int(np.sum(np.abs(s) >= s_obs - 1e-12))
    return float(rho_obs), count / len(perms)


def leave_one_out_predictive_rho(amb, ben):
    """For each law i: predict benefit rank from ambiguity rank using the
    monotone fit on the other 8 laws; return rho(predicted, actual)."""
    n = len(amb)
    r_amb = spearman_ranks(amb)
    r_ben = spearman_ranks(ben)
    preds = []
    for i in range(n):
        o = [j for j in range(n) if j != i]
        # fit rho on the training laws; sign tells whether the mapping is
        # monotone increasing or decreasing
        rho_tr = _rho(r_amb[o], r_ben[o])
        sign = 1.0 if rho_tr >= 0 else -1.0
        preds.append(sign * r_amb[i])
    return _rho(np.asarray(preds), r_ben)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/fig/regime_oos.json")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "models", "ext")
    if not os.path.isdir(base):
        # standalone paper repo layout: data lives in results/ at repo root
        alt = os.path.join(os.path.dirname(os.path.dirname(here)), "results")
        if os.path.isdir(alt):
            base = alt
    pre = json.load(open(os.path.join(base, "pre_registration.json")))
    amb = {d: pre["ambiguity_from_vocabulary_only"][d] for d in DOMAINS}

    real = {d: [] for d in DOMAINS}
    dummy = {d: [] for d in DOMAINS}
    for s in (0, 1, 2):
        for law, acc in (("real", real), ("dummy", dummy)):
            path = os.path.join(base, f"lca_{law}_s{s}.eval.json")
            ev = json.load(open(path))
            for d in DOMAINS:
                acc[d].append(ev["generalist"][d]["curve_err"])

    rows = []
    for d in DOMAINS:
        r = float(np.median(real[d]))
        u = float(np.median(dummy[d]))
        rows.append({
            "domain": d,
            "ambiguity": amb[d],
            "curve_real_median": r,
            "curve_dummy_median": u,
            "benefit": 1.0 - r / u,
            "benefit_x": u / r,
        })

    nondeg = [r for r in rows if r["domain"] not in DEGENERATE]
    rho_all, p_all = spearman_exact([r["ambiguity"] for r in rows],
                                    [r["benefit"] for r in rows])
    rho_nd, p_nd = spearman_exact([r["ambiguity"] for r in nondeg],
                                  [r["benefit"] for r in nondeg])
    loo = leave_one_out_predictive_rho([r["ambiguity"] for r in nondeg],
                                       [r["benefit"] for r in nondeg])

    by_amb = {}
    for r in nondeg:
        by_amb.setdefault(r["ambiguity"], []).append(r["benefit"])
    group_means = {round(k, 3): float(np.mean(v)) for k, v in
                   sorted(by_amb.items(), reverse=True)}

    # pre-registered specific predictions
    check_pendulum = next(r for r in rows if r["domain"] == "pendulum")
    check_spring = next(r for r in rows if r["domain"] == "spring")

    out = {
        "pre_registration": "models/ext/pre_registration.json",
        "seeds": [0, 1, 2],
        "rows": rows,
        "spearman_all_10": {"rho": rho_all, "p": p_all},
        "spearman_9_nondegenerate": {"rho": rho_nd, "p": p_nd},
        "leave_one_out_predictive_rho": loo,
        "group_benefit_means_by_ambiguity": group_means,
        "predictions_verified": {
            "pendulum_benefit_positive": float(check_pendulum["benefit"]),
            "spring_benefit_positive": float(check_spring["benefit"]),
            "group_order_1.0_gt_0.75_gt_0.667_gt_0.5": (
                group_means[1.0] >= group_means[0.75] >= group_means[0.667]
                >= group_means[0.5]),
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

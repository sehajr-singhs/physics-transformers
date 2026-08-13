"""agg_matrix.py — aggregate the 75-run baseline matrix (5 domains x 3
architectures x 5 seeds) into the significance summary used by the
loss-channel paper.

Each run writes physx/models/matrix/matrix_{domain}_{kind}_s{seed}.log with a
final `val_rel_mae` / `phys_resid` line, plus a .stats.json with
train_rel_mae. This script parses those committed artifacts and regenerates
paper/fig/significance.json (mean/std over seeds, plus per-domain paired
tests of the physics-loss effect).

usage: python physx/agg_matrix.py [--out paper/fig/significance.json]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models", "matrix")
DOMAINS = ["beam", "cantilever", "projectile", "burgers", "heat2d"]
KINDS = ["phys", "nophys", "mlp"]
SEEDS = list(range(5))

_LAST = re.compile(
    r"val_rel_mae\s+([0-9.]+)\s+phys_resid\s+([0-9.eE+-]+)")


def _files():
    out = []
    for d in DOMAINS:
        for k in KINDS:
            for s in SEEDS:
                out.append(os.path.join(MODELS, f"matrix_{d}_{k}_s{s}.stats.json"))
    return out


FILES = _files()


def _read_run(path):
    """Return dict(val_rel_mae, phys_resid, train_rel_mae) for one run."""
    base = path[: -len(".stats.json")]
    log = base + ".log"
    if not os.path.exists(log):
        # logs were written next to the stats in physx/models/ (one level up)
        log = os.path.join(os.path.dirname(os.path.dirname(base)),
                           os.path.basename(base) + ".log")
    val = phys = None
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LAST.search(line)
                if m:
                    val, phys = float(m.group(1)), float(m.group(2))
    train = None
    if os.path.exists(path):
        with open(path) as f:
            st = json.load(f)
        train = st.get("train_rel_mae")
    return {"val_rel_mae": val, "phys_resid": phys, "train_rel_mae": train}


def _cliff_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n = len(a) * len(b)
    s = 0.0
    for x in a:
        s += np.sum(np.sign(x - b))
    return float(s / n)


def aggregate():
    runs = {d: {k: [] for k in KINDS} for d in DOMAINS}
    for f in FILES:
        parts = os.path.basename(f)[: -len(".stats.json")].split("_")
        domain, kind = parts[1], parts[2]
        runs[domain][kind].append(_read_run(f))

    summary = {}
    for d in DOMAINS:
        summary[d] = {}
        for k in KINDS:
            vals = [r["val_rel_mae"] for r in runs[d][k] if r["val_rel_mae"] is not None]
            res = [r["phys_resid"] for r in runs[d][k] if r["phys_resid"] is not None]
            summary[d][k] = {
                "val_rel_mae": {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                "values": vals, "n": len(vals)},
                "phys_resid": {"mean": float(np.mean(res)), "std": float(np.std(res)),
                               "values": res, "n": len(res)},
            }

    per_domain_tests = {}
    for d in DOMAINS:
        a = [r["val_rel_mae"] for r in runs[d]["phys"]]
        b = [r["val_rel_mae"] for r in runs[d]["nophys"]]
        m = [r["val_rel_mae"] for r in runs[d]["mlp"]]
        per_domain_tests[d] = {"complete": all(v is not None for v in a + b + m),
                               "val_rel_mae": {}}
        for name, x, y in (("phys_vs_nophys", a, b), ("phys_vs_mlp", a, m),
                           ("nophys_vs_mlp", b, m)):
            if None in x or None in y:
                p, cd = float("nan"), float("nan")
            else:
                try:
                    w, p = stats.wilcoxon(x, y)
                except ValueError:
                    p = float("nan")
                cd = _cliff_delta(x, y)
            per_domain_tests[d]["val_rel_mae"][name] = {
                "p": float(p), "n": min(len(x), len(y)),
                "cliff_delta": cd,
                "median_ratio": float(np.median(x) / max(np.median(y), 1e-12)),
                "a_better": float(np.median(x)) < float(np.median(y)),
            }

    out = {"domains": DOMAINS, "kinds": KINDS, "seeds": SEEDS,
           "summary": summary, "per_domain_tests": per_domain_tests}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/fig/significance.json")
    args = ap.parse_args(argv)
    out = aggregate()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for d in DOMAINS:
        line = "  ".join(f"{k}: {out['summary'][d][k]['val_rel_mae']['mean']:.3f}"
                         for k in KINDS)
        print(f"{d:11s} {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

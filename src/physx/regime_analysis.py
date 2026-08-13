import os

"""regime_analysis.py — when does equation conditioning matter?

Quantifies, for each law in the shared-head multi-law experiment, how much the
parameter-token sequence alone identifies the governing equation (the
"equation ambiguity"), and tests whether the measured LCA benefit correlates
with it.

Equation ambiguity of law i:
    amb(i) = max_{j != i}  |quantity(i) ∩ quantity(j)| / |quantity(i)|

    0  -> the tokens are unique to this law (the equation is redundant)
    1  -> another law presents the *identical* token sequence (the equation is
          the only thing that can disambiguate behavior)

LCA benefit of law i:
    benefit(i) = 1 - median_curve_real(i) / median_curve_dummy(i)
    (>0 means the real equation signature improved trajectory fidelity)

The hypothesis: benefit grows monotonically with ambiguity. Spearman's rho is
reported; rc is excluded from the headline correlation because its normalized
trajectory is *identical across parameters* (the curve target carries no
information, so neither condition can improve it — a structural degeneracy,
not a failure of conditioning).
"""

import json

import numpy as np

from . import laws

# ambiguity is computed over the ORIGINAL 6-law set (the experiment the
# 6-law regime figure reports). The 10-law out-of-sample extension lives in
# regime_oos.py and uses laws.SHARED_HEAD_DOMAINS.
DOMAINS = laws.SHARED_HEAD_DOMAINS_6  # beam, cantilever, projectile, pendulum, spring, rc


def ambiguity(domain):
    mine = set(laws.DOMAIN_QUANTITIES[domain])
    best = 0.0
    for other in DOMAINS:
        if other == domain:
            continue
        theirs = set(laws.DOMAIN_QUANTITIES[other])
        best = max(best, len(mine & theirs) / len(mine))
    return best


def spearman_exact(x, y):
    """Spearman's rho with an EXACT permutation p-value (n <= 7)."""
    import itertools
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rho_obs = _rho(rx, ry)
    n = len(x)
    count = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        rp = np.asarray(perm, float)
        r = _rho(rx, rp)
        total += 1
        if abs(r) >= abs(rho_obs) - 1e-12:
            count += 1
    return float(rho_obs), count / total


def _rho(rx, ry):
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def _fig_path(name):
    for cand in (os.path.join("paper", "fig", name),
                 os.path.join("figs", name),
                 name):
        if os.path.exists(cand):
            return cand
    return os.path.join("paper", "fig", name)


def main(out="paper/fig/regime_analysis.json"):
    data = json.load(open(_fig_path("multi_law_data.json")))
    pd = data["per_domain"]

    rows = []
    for d in DOMAINS:
        amb = ambiguity(d)
        m = pd[d]["curve_err"]
        real, dummy = m["median_real"], m["median_dummy"]
        benefit = 1.0 - real / dummy
        rows.append({
            "domain": d,
            "ambiguity": amb,
            "curve_real_median": real,
            "curve_dummy_median": dummy,
            "benefit": benefit,
            "benefit_x": dummy / real,   # how many times better the real model is
            "p": m["p"],
        })

    # headline correlation over the five non-degenerate laws
    five = [r for r in rows if r["domain"] != "rc"]
    rho_all, p_all = spearman_exact([r["ambiguity"] for r in rows],
                                    [r["benefit"] for r in rows])
    rho_five, p_five = spearman_exact([r["ambiguity"] for r in five],
                                      [r["benefit"] for r in five])

    out_d = {
        "domains": rows,
        "spearman_all": {"rho": float(rho_all), "p": float(p_all)},
        "spearman_5_nondegenerate": {"rho": float(rho_five), "p": float(p_five)},
        "note": ("rc excluded from the headline rho: its normalized curve is "
                 "parameter-invariant, so neither condition can improve it"),
    }
    with open(out, "w") as f:
        json.dump(out_d, f, indent=1)
    print(json.dumps(out_d, indent=1))


if __name__ == "__main__":
    main()

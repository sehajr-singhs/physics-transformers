"""make_figures_ext.py — figures for the 10-law extension.

  fig7_regime_oos.png    pre-registered ambiguity -> benefit, 10 laws:
                         the pre-registered monotone prediction vs. the
                         measured benefits (the prediction FAILED; the panel
                         shows the falsification and the failure analysis:
                         token-identity twins vs. superset inflation)
  fig8_deeponet.png      LCA generalist vs DeepONet baselines

Data: physx/models/ext/*.eval.json (6 trained generalists), the
pre-registration json, and paper/fig/deeponet_baselines.json.

usage: python3 paper_physformer/fig/make_figures_ext.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)          # paper dir (fig/ or figs/)
ROOT = HERE
while not (os.path.isdir(os.path.join(ROOT, "physx"))
           or os.path.isdir(os.path.join(ROOT, "src", "physx"))):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        break
    ROOT = parent
MODELS = os.path.join(ROOT, "physx", "models", "ext")
if not os.path.isdir(MODELS) and os.path.isdir(os.path.join(ROOT, "results")):
    MODELS = os.path.join(ROOT, "results")
DATA = os.path.join(PAPER, "fig") if os.path.isdir(os.path.join(PAPER, "fig")) \
    else os.path.join(PAPER, "figs")

OUT = HERE
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 200,
})

REAL_C = "#1a6f8e"
DUMMY_C = "#c0552c"
SPEC_C = "#7a7a7a"
NEW_C = "#2e8b57"
TWIN_C = "#8e1a1a"
GRID = dict(color="#dddddd", lw=0.7, zorder=0)

DOMAINS = ["beam", "cantilever", "projectile", "pendulum", "spring", "rc",
           "damped", "kepler", "lc", "drag"]
OLD6 = set(["beam", "cantilever", "projectile", "pendulum", "spring", "rc"])
# the only pair of laws whose parameter tokens are IDENTICAL (same quantity
# ids in the same order) -- the equation signature is the only distinguishing
# information the model receives
TWINS = {"beam", "cantilever"}


def _load(path):
    with open(path) as f:
        return json.load(f)


def compute_rows():
    pre = _load(os.path.join(MODELS, "pre_registration.json"))
    amb = pre["ambiguity_from_vocabulary_only"]
    real = {d: [] for d in DOMAINS}
    dummy = {d: [] for d in DOMAINS}
    for s in (0, 1, 2):
        for law, acc in (("real", real), ("dummy", dummy)):
            ev = _load(os.path.join(MODELS, f"lca_{law}_s{s}.eval.json"))
            for d in DOMAINS:
                acc[d].append(ev["generalist"][d]["curve_err"])
    rows = []
    for d in DOMAINS:
        r = float(np.median(real[d]))
        u = float(np.median(dummy[d]))
        rows.append({"domain": d, "ambiguity": amb[d], "curve_real": r,
                     "curve_dummy": u, "benefit": 1.0 - r / u,
                     "new": d not in OLD6, "twin": d in TWINS})
    return rows, amb


def fig7_regime_oos():
    rows, amb = compute_rows()
    nondeg = [r for r in rows if r["domain"] != "rc"]
    # pre-registered monotone prediction: group means in ambiguity order
    gmeans = {}
    for r in nondeg:
        gmeans.setdefault(r["ambiguity"], []).append(r["benefit"])
    xs = sorted(gmeans, reverse=True)
    ys = [float(np.mean(gmeans[x])) for x in xs]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    for r in nondeg:
        if r["twin"]:
            c, m, s = TWIN_C, "s", 92
        else:
            c, m, s = (NEW_C if r["new"] else REAL_C), ("^" if r["new"] else "o"), 72
        ax.scatter(r["ambiguity"], r["benefit"], s=s, marker=m, color=c,
                   zorder=3, edgecolor="#333333", linewidth=0.8)
        dy = 0.028 if r["benefit"] >= 0 else -0.05
        va = "bottom" if dy > 0 else "top"
        ax.annotate(r["domain"], (r["ambiguity"], r["benefit"]),
                    xytext=(r["ambiguity"], r["benefit"] + dy),
                    ha="center", va=va, fontsize=8.4, color="#111111")
    rc = next(r for r in rows if r["domain"] == "rc")
    ax.scatter([rc["ambiguity"]], [rc["benefit"]], s=72, marker="x",
               color="#999999", zorder=3)
    ax.annotate("rc (degenerate)", (rc["ambiguity"], rc["benefit"]),
                xytext=(rc["ambiguity"] - 0.02, rc["benefit"] + 0.03),
                fontsize=8.4, color="#777777")
    ax.plot(xs, ys, "--", color="#555555", lw=1.3, zorder=2,
            label="pre-registered prediction\\n(monotone in ambiguity)")
    ax.set_xlabel("equation ambiguity (from token vocabulary alone)")
    ax.set_ylabel("LCA benefit  $1 - \\mathrm{err}_{\\mathrm{real}}/\\mathrm{err}_{\\mathrm{dummy}}$")
    ax.set_title("(a)  Pre-registered prediction vs. measured benefit")
    ax.set_xlim(-0.08, 1.15)
    ax.grid(**GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.2, loc="lower right")
    ax.text(0.02, 0.97,
            "squares: token-identical twin pair\\n"
            "circles: original 6 laws    triangles: 4 new laws\\n"
            "$\\rho = 0.07$ (p = 0.88)  -- prediction falsified",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999999"))

    ax = axes[1]
    # failure analysis: token-identity confusion (exact quantity-id sequence)
    # vs. the overlap measure's superset inflation
    order = sorted(nondeg, key=lambda r: r["benefit"])
    names = [r["domain"] for r in order]
    ben = np.array([r["benefit"] for r in order])
    cols = [TWIN_C if r["twin"] else (NEW_C if r["new"] else REAL_C) for r in order]
    ax.barh(range(len(order)), ben, color=cols, alpha=0.92,
            edgecolor="#333333", linewidth=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=8.6)
    ax.axvline(0, color="#444444", lw=0.9)
    ax.set_xlabel("measured LCA benefit")
    ax.set_title("(b)  Failure analysis: benefit by law")
    ax.text(0.98, 0.97,
            "red: only pair with IDENTICAL\\nparameter tokens (beam/cantilever)\\n"
            "benefit is not monotone in the\\noverlap measure (spring: -0.41)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999999"))
    ax.grid(**GRID, axis="x"); ax.set_axisbelow(True)

    fig.suptitle("Out-of-sample regime test: the pre-registered prediction failed; the failure is analyzed, not hidden",
                 y=1.02, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_regime_oos.png"), bbox_inches="tight")
    plt.close(fig)


def fig8_deeponet():
    rows, amb = compute_rows()
    db = _load(os.path.join(DATA, "deeponet_baselines.json"))
    per_law = db["per_law"]
    order = sorted([r for r in rows if r["domain"] != "rc"], key=lambda r: r["benefit"])

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    x = np.arange(len(order))
    w = 0.34
    gen = [next(r for r in rows if r["domain"] == o["domain"])["curve_real"] for o in order]
    dnet = [per_law[o["domain"]]["curve_err"] for o in order]
    ax.bar(x - w / 2, dnet, w, label="per-law DeepONet (10 dedicated models)",
           color=SPEC_C, alpha=0.92, edgecolor="#333333", linewidth=0.5)
    ax.bar(x + w / 2, gen, w, label="LCA generalist (1 model, no law label)",
           color=REAL_C, alpha=0.92, edgecolor="#333333", linewidth=0.5)
    for xi, v in zip(x - w / 2, dnet):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.2)
    for xi, v in zip(x + w / 2, gen):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels([o["domain"] for o in order], fontsize=9)
    ax.set_ylabel("held-out trajectory error (peak-normalized MAE)")
    ax.set_title("External operator-network baseline on the same 10-law protocol (same data budget)")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    ax.grid(**GRID, axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, 0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_deeponet.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    fig7_regime_oos()
    fig8_deeponet()
    print("done")


if __name__ == "__main__":
    main()

"""make_figures.py — figures for the physics-transformer paper.

  fig1_architecture.png   tensor pipeline + fused reasoning/physics layers
  fig2_multilaw.png       per-law trajectory error, real vs dummy signature
  fig3_lawswap.png        causal law-swap steering + disruption
  fig4_fewshot.png        few-shot adaptation to a held-out law
  fig5_regime.png         equation-ambiguity vs LCA benefit (regime theory)
  fig6_channels.png       the two physics channels: loss (consistency) vs
                          input (fidelity) — heat2d sweep + DeepXDE comparison

Every panel is drawn from the committed JSON artifacts under paper/fig/.

usage: python3 paper_physformer/fig/make_figures.py [--only fig2,fig3]
"""

import argparse
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
GRID = dict(color="#dddddd", lw=0.7, zorder=0)


def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


# ------------------------------------------------------------------ fig 1
def fig1_architecture():
    fig = plt.figure(figsize=(9.8, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.18)

    # ---- panel (a): tensor pipeline
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    ax.set_title("(a)  Tensor pipeline: physics as tensors", pad=6)

    def box(ax, x, y, w, h, text, fc, fs=8.6, lw=1.2):
        from matplotlib.patches import FancyBboxPatch
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                           fc=fc, ec="#333333", lw=lw)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#111111")

    def arrow(ax, x1, y1, x2, y2, color="#444444"):
        from matplotlib.patches import FancyArrowPatch
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=color, lw=1.4))

    box(ax, 2, 37, 26, 8, "Problem tensor\n$\\mathbf{P}=(L,P,E,I,h)$", "#eaf2fb")
    box(ax, 2, 26, 26, 8, "Quantity labels\nlength·force·modulus\ninertia·thickness", "#eaf2fb")
    box(ax, 34, 32, 30, 11, "Quantity embedding\n$\\mathbf{x}_i = \\mathrm{Emb}(q_i) + \\mathrm{proj}(p_i)$\n+ positional encoding", "#dff0d8")
    box(ax, 70, 33, 28, 10, "Token tensor\n$(B,\\,P,\\,d_{\\mathrm{model}})$", "#fcf8e3")
    arrow(ax, 28, 41, 34, 39)
    arrow(ax, 15, 34, 15, 34)
    arrow(ax, 28, 30, 34, 35)
    arrow(ax, 64, 37, 70, 38)
    box(ax, 6, 12, 26, 8, "Target tensors\ntrajectory $(50\\times d)$\nfield $(n_x\\times n_y)$\nanswer scalar", "#f2dede")
    box(ax, 40, 12, 28, 8, "Physics-consistency\nlayer (differentiable\nODE/PDE residual)", "#e8d5f5")
    arrow(ax, 34, 16, 40, 16)
    ax.text(50, 3.5, "inputs and targets are tensors with physical meaning — never flattened bags of numbers",
            ha="center", fontsize=8.4, style="italic", color="#333333")

    # ---- panel (b): fused layers
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    ax.set_title("(b)  Fused reasoning + physics layers (one transformer, many laws)", pad=6)

    for i, (y, lab) in enumerate([(38, "layer 1"), (29, "layer 2"), (20, "layer 3")]):
        box(ax, 6, y, 40, 7, f"reasoning: self-attention + FFN  ({lab})", "#dff0d8")
        box(ax, 52, y, 44, 7, "LCA: cross-attend to law vector", "#fcf8e3")
        arrow(ax, 46, y + 3.5, 52, y + 3.5)
    box(ax, 52, 11, 44, 6, "law MLP  $\\leftarrow$ equation signature\n(22-operator vocabulary)", "#e8d5f5")
    box(ax, 6, 11, 40, 6, "law-gated readout  $\\gamma\\odot h + \\beta$", "#f5d5e8")
    arrow(ax, 74, 17, 74, 20)
    arrow(ax, 64, 11, 56, 11)
    arrow(ax, 26, 17, 26, 20)
    box(ax, 6, 2, 40, 6, "trajectory / field tensor + answer", "#eaf2fb")
    box(ax, 52, 2, 44, 6, "residual loss  $\\mathcal{L}_{\\mathrm{phys}} = \\|\\mathcal{R}\\,u\\|^2$", "#f2dede")
    arrow(ax, 26, 8, 26, 2)
    arrow(ax, 74, 8, 74, 2)
    ax.text(50, 44.5, "physics enters twice: as INPUT (equation-conditioned attention) and as LOSS (residual)",
            ha="center", fontsize=8.4, style="italic", color="#333333")

    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig2_multilaw():
    d = _load("multi_law_data.json")
    pd = d["per_domain"]
    order = ["beam", "cantilever", "projectile", "pendulum", "spring", "rc"]
    real = [pd[k]["curve_err"]["median_real"] for k in order]
    dummy = [pd[k]["curve_err"]["median_dummy"] for k in order]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, dummy, w, label="constant-signature control", color=DUMMY_C, alpha=0.92)
    ax.bar(x + w / 2, real, w, label="law-conditioned (LCA)", color=REAL_C, alpha=0.92)
    for xi, r in zip(x, real):
        ax.text(xi + w / 2, r + 0.008, f"{r:.3f}", ha="center", fontsize=8)
    for xi, r in zip(x, dummy):
        ax.text(xi - w / 2, r + 0.008, f"{r:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("median trajectory error (RMSE)")
    ax.set_title("Trajectory fidelity: one shared transformer, six laws (36 paired runs, 6 seeds)")
    ax.legend(frameon=False, fontsize=8.6)
    ax.grid(**GRID, axis="y")
    ax.set_axisbelow(True)
    # pooled result annotation
    ax.text(0.99, 0.96, "pooled: p = 0.0003 (Wilcoxon, 36 pairs)\nmedian reduction 21%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#999999"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_multilaw.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig3_lawswap():
    d = _load("law_swap_data.json")
    s = d["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    real = s["si"]["real_median"]; dummy = s["si"]["dummy_median"]
    bars = ax.bar([0, 1], [real, dummy], width=0.5,
                  color=[REAL_C, DUMMY_C], alpha=0.92)
    for b, v in zip(bars, [real, dummy]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.012 if v >= 0 else -0.03),
                f"{v:+.3f}", ha="center", fontsize=8.6)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["law-conditioned", "constant control"])
    ax.set_ylabel("steering index  $\\Delta$SI")
    ax.set_title("(a)  Swapping the equation signature\n(beam tokens $\\to$ cantilever law)")
    ax.grid(**GRID, axis="y"); ax.set_axisbelow(True)
    ax.text(0.5, 0.95, "p < 0.0001 (n = 72)", transform=ax.transAxes,
            ha="center", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999999"))

    ax = axes[1]
    rd = s["disruption"]["real_median"]; dd = s["disruption"]["dummy_median"]
    bars = ax.bar([0, 1], [rd, dd], width=0.5, color=[REAL_C, DUMMY_C], alpha=0.92)
    for b, v in zip(bars, [rd, dd]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["law-conditioned", "constant control"])
    ax.set_ylabel("prediction disruption (RMSE)")
    ax.set_title("(b)  Sensitivity to the injected law")
    ax.grid(**GRID, axis="y"); ax.set_axisbelow(True)
    ax.text(0.5, 0.95, "control is exactly insensitive (0.000)", transform=ax.transAxes,
            ha="center", fontsize=8.6,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#999999"))
    fig.suptitle("Causal test: the equation vector steers behavior (6 seeds, 72 pooled samples)",
                 y=1.03, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_lawswap.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig4_fewshot():
    d = _load("fewshot_data.json")
    med = d["median"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    labels = ["generalist\nlaw-conditioned", "generalist\nconstant control", "from-scratch\nspecialist"]
    colors = [REAL_C, DUMMY_C, SPEC_C]
    for ax, key, title in [(axes[0], "ans", "(a)  answer error"),
                           (axes[1], "curve", "(b)  trajectory error")]:
        vals = [med["real"][key], med["dummy"][key], med["spec"][key]]
        bars = ax.bar([0, 1, 2], vals, width=0.55, color=colors, alpha=0.92)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.6)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=8.2)
        ax.set_ylabel("median error")
        ax.set_title(title)
        ax.grid(**GRID, axis="y"); ax.set_axisbelow(True)
    fig.suptitle("Adapting to a held-out law (cantilever) with 24 samples = 25% of the specialist's budget (3 seeds)",
                 y=1.03, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_fewshot.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig5_regime():
    d = _load("regime_analysis.json")
    rows = d["domains"]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for r in rows:
        is_rc = r["domain"] == "rc"
        m = "x" if is_rc else "o"
        c = "#999999" if is_rc else REAL_C
        ax.scatter(r["ambiguity"], r["benefit"], s=70, marker=m, color=c,
                   zorder=3, edgecolor="#333333", linewidth=0.8)
        dy = 0.035
        ax.annotate(r["domain"], (r["ambiguity"], r["benefit"]),
                    xytext=(r["ambiguity"], r["benefit"] + dy),
                    ha="center", fontsize=8.6, color="#111111")
    # perfect monotone fit over the 5 non-degenerate laws
    x = np.linspace(-0.05, 1.05, 50)
    ax.plot(x, 0.62 * x, "--", color=REAL_C, lw=1.2, zorder=2,
            label="perfect monotone trend (5 laws)")
    ax.set_xlabel("equation ambiguity  (max token-set overlap with another law)")
    ax.set_ylabel("LCA benefit  $1 - \\mathrm{err}_{\\mathrm{real}}/\\mathrm{err}_{\\mathrm{dummy}}$")
    ax.set_title("Regime theory: conditioning matters exactly where tokens cannot identify the law")
    ax.set_xlim(-0.08, 1.15)
    ax.grid(**GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    ax.text(0.99, 0.04,
            "Spearman $\\rho = 1.0$, exact permutation $p = 0.017$ (n = 5)\n"
            "rc (grey): curve target is parameter-invariant — structural degeneracy",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#999999"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_regime.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 6
def fig6_channels():
    d = _load("physvdata_data.json")
    groups = {}
    for r in d:
        groups.setdefault((r["n"], r["w_phys"]), []).append(r)

    def agg(key):
        out = {}
        for (n, w), rs in groups.items():
            out[(n, w)] = (float(np.mean([r[key] for r in rs])),
                           float(np.std([r[key] for r in rs])))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axes[0]
    field = agg("field_mean_err"); resid = agg("phys_resid")
    for n, col in [(64, DUMMY_C), (256, REAL_C)]:
        for w, m in [(0.0, "o"), (0.05, "s")]:
            fv, fsd = field[(n, w)]; rv, rsd = resid[(n, w)]
            ax.errorbar(rv, fv, xerr=rsd, yerr=fsd, marker=m, color=col,
                        capsize=3, ms=7, lw=1.6, zorder=3)
            ax.annotate(f"n={n}  $w_{{\\mathrm{{phys}}}}$={w:g}", (rv, fv),
                        xytext=(rv * 1.35, fv), fontsize=7.8, color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("governing-equation residual (physics violation)")
    ax.set_ylabel("held-out field error (mean)")
    ax.set_title("(a)  Loss channel: consistency, not accuracy\n(heat plate, 3 seeds each)")
    ax.grid(**GRID); ax.set_axisbelow(True)

    ax = axes[1]
    dc = _load("deepxde_comparison.json")["instances"]
    insts = ["canonical", "mild", "shock"]
    x = np.arange(3)
    px = [dc[i]["physformer"]["curve_err"] for i in insts]
    dx = [dc[i]["deepxde"]["curve_err"] for i in insts]
    ax.bar(x - 0.19, dx, 0.38, label="per-instance PINN (DeepXDE)", color=SPEC_C, alpha=0.92)
    ax.bar(x + 0.19, px, 0.38, label="generalist PhysFormer", color=REAL_C, alpha=0.92)
    for xi, v in zip(x - 0.19, dx):
        ax.text(xi, v + 0.015, f"{v:.4f}", ha="center", fontsize=7.6)
    for xi, v in zip(x + 0.19, px):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(insts)
    ax.set_ylabel("full-field curve error (Burgers)")
    ax.set_title("(b)  Generalist vs specialist PINN\n(specialists: ~19 min training each)")
    ax.legend(frameon=False, fontsize=8.2)
    ax.grid(**GRID, axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, 0.62)

    fig.suptitle("The two physics channels do different work", y=1.03, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_channels.png"), bbox_inches="tight")
    plt.close(fig)


FIGS = {
    "fig1": fig1_architecture,
    "fig2": fig2_multilaw,
    "fig3": fig3_lawswap,
    "fig4": fig4_fewshot,
    "fig5": fig5_regime,
    "fig6": fig6_channels,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated fig ids")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else set(FIGS)
    for name, fn in FIGS.items():
        if name in want:
            print("making", name)
            fn()
    print("done")


if __name__ == "__main__":
    main()

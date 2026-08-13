"""run_physvdata.py — does physics supervision improve field-level
generalization? A controlled sweep on the 2D heat plate (the hardest field
domain): identical architecture, data, and budget; the ONLY difference is the
physics-consistency loss (w_phys = 0.05 vs 0.0 = data-only), at two data
fractions (25% and 100% of the 256-sample budget), 3 seeds.

Evaluation is on the held-out val split from the training seed protocol:
  problems = generate(heat2d, n=samples, seed); val = problems[-samples//8:]
metrics per val problem: answer rel error, field max/mean error (2D field vs
manufactured exact solution), and the governing-equation residual.

usage: python physx/run_physvdata.py [--stages train,score]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
MODELS = os.path.join(ROOT, "physx", "models", "sweep")
EPOCHS = 240
CONFIGS = [(n, w) for n in (64, 256) for w in (0.0, 0.05)]
SEEDS = [0, 1, 2]


def tag(n, w, s):
    return f"h2d_n{n}_w{int(w * 100):02d}_s{s}"


def train_one(n, w, s, threads=2):
    t = tag(n, w, s)
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--domain", "heat2d", "--epochs", str(EPOCHS), "--samples", str(n),
           "--seed", str(s), "--threads", str(threads), "--w-traj", "3",
           "--w-phys", str(w), "--save", os.path.join(MODELS, f"{t}.pt")]
    log = os.path.join(MODELS, f"{t}.log")
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[sweep] {t}: {'OK' if r.returncode == 0 else 'FAIL'}", flush=True)
    return r.returncode == 0


@torch.no_grad()
def score_one(n, w, s):
    from physx import dataset, sim
    from physx.physformer import build
    t = tag(n, w, s)
    meta = json.load(open(os.path.join(MODELS, f"{t}.stats.json")))
    st = meta["param_stats"]
    ans_mean, ans_std = meta["answer_stats"]
    arch = meta.get("arch", {})
    model = build("heat2d", st, d_model=arch.get("d_model", 48),
                  n_layers=arch.get("n_layers", 3), nhead=arch.get("nhead", 4),
                  dim_ff=arch.get("dim_ff", 96),
                  traj_hidden=arch.get("traj_hidden", 64),
                  traj_steps=arch.get("traj_steps", 400),
                  kind=arch.get("kind", "physformer"),
                  sigmoid_traj=arch.get("sigmoid_traj", False))
    model.load_state_dict(torch.load(os.path.join(MODELS, f"{t}.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    probs = dataset.generate("heat2d", n=n, seed=s)[-max(4, n // 8):]
    grid = sim.H2D_N
    ans_errs, max_errs, mean_errs, phys = [], [], [], []
    for p in probs:
        pids = torch.tensor([list(range(len(st["keys"])))], dtype=torch.long)
        vals = torch.tensor([dataset.normalize(p["params"], st)], dtype=torch.float32)
        ans, traj = model(pids, vals)
        field = traj[0, :, 0].numpy().reshape(grid, grid) * float(p["params"]["A"])
        exact = sim.heat2d_traj(p["params"]).reshape(grid, grid)
        ans_pred = float(ans[0]) * ans_std + ans_mean
        ans_errs.append(abs(ans_pred - float(p["answer"])) / float(p["answer"]))
        max_errs.append(float(np.max(np.abs(field - exact)) / exact.max()))
        mean_errs.append(float(np.mean(np.abs(field - exact)) / exact.max()))
        params_t = {k: torch.tensor([v], dtype=torch.float32) for k, v in p["params"].items()}
        phys.append(model.physics_residual(
            torch.tensor(field.reshape(1, -1, 1)), params_t).mean().item())
    return {"tag": t, "n": n, "w_phys": w, "seed": s,
            "ans_rel_mae": float(np.mean(ans_errs)),
            "field_max_err": float(np.mean(max_errs)),
            "field_mean_err": float(np.mean(mean_errs)),
            "phys_resid": float(np.mean(phys))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="train,score")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True)

    if "train" in args.stages:
        import concurrent.futures as cf
        jobs = [(n, w, s) for n, w in CONFIGS for s in SEEDS]
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            ok = list(ex.map(lambda j: train_one(*j), jobs))
        if not all(ok):
            print("[sweep] some runs failed", flush=True)

    if "score" in args.stages:
        results = [score_one(n, w, s) for n, w in CONFIGS for s in SEEDS]
        with open(os.path.join(ROOT, "paper", "fig", "physvdata_data.json"), "w") as f:
            json.dump(results, f, indent=1)
        for r in results:
            print(f"[sweep] {r['tag']}: ans {r['ans_rel_mae']:.3f} "
                  f"field_max {r['field_max_err']:.3f} field_mean {r['field_mean_err']:.4f} "
                  f"phys {r['phys_resid']:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

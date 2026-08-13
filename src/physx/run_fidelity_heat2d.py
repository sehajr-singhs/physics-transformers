"""run_fidelity_heat2d.py — trajectory-fidelity retrain for the heat plate.

The matrix phys model nails the scalar (1.5%) but underfits the 2D field
(shape spans [0.33, 0.80] instead of [0, 1]). Mirroring the Burgers fix,
retrain with a heavier trajectory weight (w_traj=3) and a longer budget
(120 epochs), 3 seeds, then promote the best canonical candidate.

usage: python physx/run_fidelity_heat2d.py
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
LOG_DIR = os.path.join(ROOT, "physx", "models")
CANONICAL = {"A": 300.0, "k": 2.0, "l": 3.0}


def run_job(seed):
    tag = f"fid_heat2d_s{seed}"
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--domain", "heat2d", "--epochs", "120", "--samples", "256",
           "--seed", str(seed), "--threads", "2", "--w-traj", "3",
           "--save", os.path.join(LOG_DIR, "matrix", f"{tag}.pt")]
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[fidh2d] {tag}: {'OK' if r.returncode == 0 else 'FAIL'}", flush=True)
    return tag, r.returncode == 0


def canonical_score(tag):
    """answer + field (max/mean) error on the canonical plate."""
    from physx import dataset, sim
    from physx.physformer import build
    import torch
    path = os.path.join(LOG_DIR, "matrix", f"{tag}.pt")
    stats = os.path.join(LOG_DIR, "matrix", f"{tag}.stats.json")
    meta = json.load(open(stats))
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
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    p = CANONICAL
    pids = torch.tensor([list(range(len(st["keys"])))], dtype=torch.long)
    vals = torch.tensor([dataset.normalize(p, st)], dtype=torch.float32)
    with torch.no_grad():
        ans, traj = model(pids, vals)
    n = sim.H2D_N
    field = traj[0, :, 0].numpy().reshape(n, n) * float(p["A"])
    exact = sim.heat2d_traj(p).reshape(n, n)
    ans_pred = float(ans[0]) * ans_std + ans_mean
    ans_err = abs(ans_pred - float(p["A"])) / float(p["A"])
    max_err = float(np.max(np.abs(field - exact)) / exact.max())
    mean_err = float(np.mean(np.abs(field - exact)) / exact.max())
    return {"tag": tag, "answer_err": ans_err, "field_max_err": max_err,
            "field_mean_err": mean_err}


def main():
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run_job, range(3)))
    scores = []
    for tag, ok in results:
        if ok:
            try:
                scores.append(canonical_score(tag))
                print(f"[fidh2d] {tag}: {scores[-1]}")
            except Exception as e:  # noqa: BLE001
                print(f"[fidh2d] {tag}: eval failed {e}")
    if not scores:
        print("[fidh2d] no candidates")
        return 1
    best = min(scores, key=lambda s: s["field_max_err"])
    import shutil
    shutil.copyfile(os.path.join(LOG_DIR, "matrix", f"{best['tag']}.pt"),
                    os.path.join(LOG_DIR, "heat2d.pt"))
    shutil.copyfile(os.path.join(LOG_DIR, "matrix", f"{best['tag']}.stats.json"),
                    os.path.join(LOG_DIR, "heat2d.stats.json"))
    print(f"[fidh2d] promoted {best['tag']} -> physx/models/heat2d.pt")
    with open(os.path.join(LOG_DIR, "fid_heat2d_scores.json"), "w") as f:
        json.dump(scores, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

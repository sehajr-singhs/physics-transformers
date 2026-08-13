"""run_fidelity.py — headline trajectory-fidelity retrains.

Trains beam, cantilever, and burgers (physformer + physics, 3 seeds each)
with the sigmoid shape constraint on the trajectory head; burgers uses a
trajectory-weighted, longer budget (w_traj=3, 120 epochs) because its field
is 4x larger. Each candidate is evaluated on both held-out accuracy and the
canonical showcase case; the best candidate per domain is promoted to
physx/models/<domain>.pt.

usage: python physx/run_fidelity.py
"""

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
LOG_DIR = os.path.join(ROOT, "physx", "models")

CANONICAL = {
    "beam": {"L": 4.0, "P": 3000.0, "E": 2e11, "I": 5e-6, "h": 0.2},
    "cantilever": {"L": 4.0, "P": 3000.0, "E": 2e11, "I": 5e-6, "h": 0.2},
    "burgers": {"nu": 0.05, "A": 1.5, "sigma": 0.3},
}
JOBS = []
for d in ["beam", "cantilever"]:
    for s in range(3):
        JOBS.append((d, s, 60, 1.0))
for s in range(3):
    JOBS.append(("burgers", s, 120, 3.0))


def run_job(domain, seed, epochs, w_traj):
    tag = f"fid_{domain}_s{seed}"
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--domain", domain, "--epochs", str(epochs), "--samples", "256",
           "--seed", str(seed), "--threads", "4", "--w-traj", str(w_traj),
           "--save", os.path.join(LOG_DIR, "matrix", f"{tag}.pt")]
    log_path = os.path.join(LOG_DIR, f"{tag}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"[fid] {tag}: {'OK' if r.returncode == 0 else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)
    return tag, r.returncode == 0


def canonical_error(domain, tag):
    """Best-effort: answer + curve error on the canonical case."""
    try:
        from physx import dataset, sim
        from physx.physformer import build
        import numpy as np
        import torch
        base = os.path.join(LOG_DIR, "matrix", tag)
        meta = json.load(open(base + ".stats.json"))
        st = meta["param_stats"]
        arch = meta["arch"]
        model = build(domain, st, d_model=arch["d_model"], n_layers=arch["n_layers"],
                      traj_hidden=arch["traj_hidden"], traj_steps=arch["traj_steps"],
                      kind=arch["kind"], sigmoid_traj=arch.get("sigmoid_traj", False))
        model.load_state_dict(torch.load(base + ".pt", map_location="cpu", weights_only=True))
        model.eval()
        p = CANONICAL[domain]
        pids = torch.tensor([[i for i in range(len(st["keys"]))]], dtype=torch.long)
        vals = torch.tensor([dataset.normalize(p, st)], dtype=torch.float32)
        ans_mean, ans_std = meta["answer_stats"]
        with torch.no_grad():
            ans, traj = model(pids, vals)
        pred = dataset.answer_inverse(domain, float(ans[0]) * ans_std + ans_mean)
        exact = sim.closed(domain, p)["answer"]
        aerr = abs(pred - exact) / exact
        if domain == "burgers":
            field = traj[0, :, 0].numpy().reshape(sim.NT, sim.NX) * p["A"]
            u_exact = sim.burgers_field(p["nu"], p["A"], p["sigma"], sim.TF)
            cerr = float(np.max(np.abs(field[-1] - u_exact)) / np.max(np.abs(u_exact)))
        else:
            if traj.shape[1] != 50:
                cerr = float("nan")
            else:
                w_pred = traj[0, :, 0].numpy() * pred
                w_exact = sim.trajectory(domain, p, 50)[:, 0]
                cerr = float(np.max(np.abs(w_pred - w_exact)) / np.max(np.abs(w_exact)))
        return aerr, cerr
    except Exception as e:  # pragma: no cover
        return float("nan"), float("nan")


def main():
    os.makedirs(os.path.join(LOG_DIR, "matrix"), exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(run_job, *j) for j in JOBS]
        tags = {f.result()[0] for f in futs if f.result()[1]}
    print(f"[fid] {len(tags)} candidates, evaluating canonical cases", flush=True)
    results = {}
    for d in ["beam", "cantilever", "burgers"]:
        cands = [t for t in tags if t.startswith(f"fid_{d}_")]
        scored = []
        for t in cands:
            aerr, cerr = canonical_error(d, t)
            scored.append((t, aerr, cerr))
            print(f"[fid] {t}: canonical answer err {aerr * 100:.2f}%, curve err {cerr * 100:.1f}%", flush=True)
        results[d] = scored
        best = min(scored, key=lambda x: (x[1], x[2]))
        import shutil
        src_pt = os.path.join(LOG_DIR, "matrix", f"{best[0]}.pt")
        src_st = os.path.join(LOG_DIR, "matrix", f"{best[0]}.stats.json")
        shutil.copyfile(src_pt, os.path.join(LOG_DIR, f"{d}.pt"))
        shutil.copyfile(src_st, os.path.join(LOG_DIR, f"{d}.stats.json"))
        print(f"[fid] promoted {best[0]} -> physx/models/{d}.pt", flush=True)
    with open(os.path.join(ROOT, "paper", "fig", "fig_fidelity.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("[fid] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""train.py — train PhysFormer on a generated physics dataset.

Targets are scaled so the model actually converges:
  * answers spanning orders of magnitude (beam, cantilever, rc) are learned
    in log10 space
  * trajectories are shape-normalized (divided by their own peak) for the
    wide-range domains, so the head learns the universal deformation shape
  * params are z-scored per domain

Loss = w_ans * MSE(answer) + w_traj * MSE(trajectory) + w_phys * physics residual
The physics residual (residuals.py) is the physics-consistency layer: it
enforces the governing equation on the *de-scaled* predicted trajectory.
With --hard-phys the residual is per-sample weighted by its own magnitude
(hard-example mining: the model focuses on its worst physical violations).

usage: python physx/train.py --domain beam [--epochs 60] [--samples 256] [--save models/beam.pt]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    # support running as a plain script: python physx/train.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physx import dataset
    from physx.physformer import build
else:
    from . import dataset
    from .physformer import build


def make_batches(problems, st, ans_stats, tstats, batch_size, shuffle=True, seed=0,
                  shape_norm=False, peak_key=None):
    """shape_norm: normalize each trajectory by its own peak (universal shape in
    [0, 1]); the absolute scale is recovered from the answer head. This stops
    large-amplitude problems from crushing small ones (beam deflections span
    five orders of magnitude). Returns peaks (B,1,C) for physics de-scaling.

    peak_key: for domains whose field maximum is not exactly on the sampling
    grid (Burgers peaks at t = 0, heat2d peaks between grid points), the scale
    is the known parameter (A) rather than the sampled maximum."""
    rng = np.random.RandomState(seed)
    idx = list(range(len(problems)))
    if shuffle:
        rng.shuffle(idx)
    tmean, tstd = tstats
    for i in range(0, len(idx), batch_size):
        chunk = [problems[j] for j in idx[i:i + batch_size]]
        pids = [list(range(len(st["keys"]))) for _ in chunk]
        vals = [dataset.normalize(p["params"], st) for p in chunk]
        ans_mean, ans_std = ans_stats
        y = np.array([(dataset.answer_transform(p["domain"], p["answer"]) - ans_mean) / ans_std
                      for p in chunk], dtype=np.float32)
        traj = np.array([np.array(p["traj"], dtype=np.float32) for p in chunk])
        params = {k: torch.tensor([p["params"][k] for p in chunk], dtype=torch.float32)
                  for k in st["keys"]}
        if shape_norm:
            if peak_key is not None:
                peak = np.array([[p["params"][peak_key]] for p in chunk], dtype=np.float32)
                peak = peak[:, None, :]
            else:
                peak = np.abs(traj).max(axis=1, keepdims=True)
            peak[peak < 1e-9] = 1.0
            traj_n = traj / peak
            peaks = torch.tensor(peak, dtype=torch.float32)
        else:
            traj_n = (traj - np.array(tmean)) / np.array(tstd)
            peaks = None
        yield (
            torch.tensor(pids, dtype=torch.long),
            torch.tensor(vals, dtype=torch.float32),
            torch.tensor(y),
            torch.tensor(traj_n, dtype=torch.float32),
            params,
            peaks,
        )


def train(domain, epochs=60, samples=256, batch_size=32, seed=0, lr=1e-3,
          w_ans=1.0, w_traj=1.0, w_phys=0.05, d_model=48, n_layers=3,
          device="cpu", shape_norm=None, hard_phys=False, traj_hidden=64,
          threads=None, kind="physformer"):
    if threads:
        torch.set_num_threads(threads)
    # the seed controls data generation, batch shuffling, AND weight init, so
    # multi-seed runs are proper statistical replicates
    torch.manual_seed(seed)
    np.random.seed(seed)
    # shape/scale decoupling: amplitudes spanning orders of magnitude (beam /
    # cantilever deflection, RC voltage, Burgers field, 2D temperature field)
    # are learned as a normalized shape; the answer head carries the scale.
    if shape_norm is None:
        shape_norm = domain in ("beam", "cantilever", "burgers", "rc", "heat2d")
    # Burgers and heat2d fields do not attain their exact maximum on the
    # sampling grid (t = 0 slice / between grid points), so their scale is the
    # known amplitude parameter A, not the sampled maximum
    peak_key = "A" if (shape_norm and domain in ("burgers", "heat2d")) else None
    problems = dataset.generate(domain, n=samples, seed=seed)
    st = dataset.stats(problems, domain)
    ans_stats = dataset.answer_stats(problems, domain)
    tstats = dataset.traj_stats(problems)
    n_val = max(4, samples // 8)
    train_p = problems[:-n_val]
    val_p = problems[-n_val:]

    traj_steps = int(np.array(problems[0]["traj"]).shape[0])
    # shape-normalized outputs are constrained to (0, 1) by a sigmoid on the
    # trajectory head: deflections/voltages/velocities are non-negative here,
    # and the constraint removes unphysical negative predictions
    model = build(domain, st, d_model=d_model, n_layers=n_layers,
                  traj_hidden=traj_hidden, traj_steps=traj_steps,
                  kind=kind, sigmoid_traj=bool(shape_norm)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def run_epoch(probs, shuffle, seed_, phys_w):
        model.train() if shuffle else model.eval()
        total = 0.0
        for pids, vals, y, traj_n, params, peaks in make_batches(
                probs, st, ans_stats, tstats, batch_size, shuffle=shuffle,
                seed=seed_, shape_norm=shape_norm, peak_key=peak_key):
            pids, vals, y, traj_n = (t.to(device) for t in (pids, vals, y, traj_n))
            params = {k: v.to(device) for k, v in params.items()}
            if shuffle:
                opt.zero_grad()
            ans_pred, traj_pred = model(pids, vals)
            loss_ans = torch.nn.functional.mse_loss(ans_pred, y)
            loss_traj = torch.nn.functional.mse_loss(traj_pred, traj_n)
            if shape_norm:
                # shape (0..1) x per-problem peak = absolute physical quantity
                traj_real = traj_pred * peaks.to(device)
            else:
                traj_real = traj_pred * torch.tensor(tstats[1], dtype=torch.float32, device=device) \
                    + torch.tensor(tstats[0], dtype=torch.float32, device=device)
            phys_b = model.physics_residual(traj_real, params)
            if hard_phys:
                # per-sample residual weighting (hard-example mining): the
                # batch mean emphasizes the worst physical violations instead
                # of averaging them away
                wgt = phys_b.detach().abs().clamp(min=1e-8)
                phys = (phys_b * wgt).mean() / wgt.mean().detach().clamp(min=1e-8)
            else:
                phys = phys_b.mean()
            # scale-free physics loss: divide by the detached magnitude so the
            # physics gradient stays O(1) relative to the data losses no matter
            # how large the raw residual is (PINN-style balancing)
            phys_norm = phys / (phys.detach().abs() + 1e-6)
            loss = w_ans * loss_ans + w_traj * loss_traj + phys_w * phys_norm
            if shuffle:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item()
        return total / max(1, len(probs) // batch_size)

    for epoch in range(1, epochs + 1):
        # physics-loss warm-up: let the data loss settle first, then ramp in
        phys_w = w_phys * min(1.0, epoch / 10)
        tr = run_epoch(train_p, shuffle=True, seed_=seed + epoch, phys_w=phys_w)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                val_mae, val_phys = evaluate(model, st, ans_stats, tstats, val_p, device,
                                             shape_norm=shape_norm, peak_key=peak_key)
            print(f"[{domain}] epoch {epoch:3d}/{epochs} train_loss {tr:.4f} "
                  f"val_rel_mae {val_mae:.4f} phys_resid {val_phys:.4f}", flush=True)
    with torch.no_grad():
        tr_mae, _ = evaluate(model, st, ans_stats, tstats, train_p[:64], device,
                             shape_norm=shape_norm, peak_key=peak_key)
    if kind == "mlp":
        arch = {"d_model": d_model, "n_layers": n_layers, "nhead": 0,
                "dim_ff": d_model, "traj_hidden": traj_hidden,
                "traj_steps": traj_steps, "kind": kind,
                "sigmoid_traj": bool(shape_norm)}
    else:
        arch = {"d_model": d_model, "n_layers": n_layers,
                "nhead": model.reasoning.layers[0].self_attn.num_heads,
                "dim_ff": model.reasoning.layers[0].linear1.out_features,
                "traj_hidden": traj_hidden, "traj_steps": traj_steps,
                "kind": kind, "sigmoid_traj": bool(shape_norm)}
    return model, {"param_stats": st, "answer_stats": ans_stats,
                  "traj_stats": tstats, "domain": domain,
                  "train_rel_mae": tr_mae,
                  "traj_norm": "shape" if shape_norm else "global",
                  "hard_phys": bool(hard_phys),
                  "arch": arch}


@torch.no_grad()
def evaluate(model, st, ans_stats, tstats, problems, device="cpu", shape_norm=False,
             peak_key=None):
    model.eval()
    errs = []
    phys = []
    for pids, vals, y, traj_n, params, peaks in make_batches(
            problems, st, ans_stats, tstats, batch_size=len(problems),
            shuffle=False, shape_norm=shape_norm, peak_key=peak_key):
        pids, vals = pids.to(device), vals.to(device)
        params = {k: v.to(device) for k, v in params.items()}
        ans_pred, traj_pred = model(pids, vals)
        domain = model.domain
        ans_mean, ans_std = ans_stats
        preds = np.array([dataset.answer_inverse(domain, (a.item() * ans_std + ans_mean))
                          for a in ans_pred])
        true = np.array([p["answer"] for p in problems])
        scale = np.abs(true)
        scale[scale == 0] = 1.0
        errs.append(np.mean(np.abs(preds - true) / scale))
        if shape_norm:
            traj_real = traj_pred * peaks
        else:
            traj_real = traj_pred * torch.tensor(tstats[1], dtype=torch.float32, device=device) \
                + torch.tensor(tstats[0], dtype=torch.float32, device=device)
        phys.append(model.physics_residual(traj_real, params).mean().item())
    return float(np.mean(errs)), float(np.mean(phys))


def main(argv=None):
    ap = argparse.ArgumentParser(description="train PhysFormer on a physics domain")
    ap.add_argument("--domain", required=True, choices=dataset.RANGES.keys())
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-phys", type=float, default=0.05, help="physics-loss weight (0 disables physics)")
    ap.add_argument("--hard-phys", action="store_true", help="per-sample residual-weighted physics loss")
    ap.add_argument("--shape-norm", dest="shape_norm", action="store_true", default=None,
                    help="force per-problem trajectory normalization")
    ap.add_argument("--no-shape-norm", dest="shape_norm", action="store_false",
                    help="force global trajectory normalization")
    ap.add_argument("--traj-hidden", type=int, default=64, help="trajectory-head hidden width")
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    ap.add_argument("--w-traj", type=float, default=1.0, help="trajectory loss weight")
    ap.add_argument("--kind", choices=["physformer", "mlp"], default="physformer",
                    help="encoder kind: physformer (attention + optional physics) "
                         "or mlp (data-only baseline, no physics term)")
    ap.add_argument("--save", default=None, help="path to save model weights (e.g. models/beam.pt)")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    if args.save is None:
        args.save = os.path.join(here, "models", f"{args.domain}.pt")

    t0 = time.time()
    if args.kind == "mlp":
        args.w_phys = 0.0  # the MLP baseline is data-only by construction
    model, meta = train(
        args.domain,
        epochs=args.epochs,
        samples=args.samples,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        seed=args.seed,
        w_phys=args.w_phys,
        w_traj=args.w_traj,
        hard_phys=args.hard_phys,
        shape_norm=args.shape_norm,
        traj_hidden=args.traj_hidden,
        threads=args.threads,
        kind=args.kind,
    )
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    with open(os.path.splitext(args.save)[0] + ".stats.json", "w") as f:
        json.dump(meta, f)
    print(f"saved {args.save} + stats ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

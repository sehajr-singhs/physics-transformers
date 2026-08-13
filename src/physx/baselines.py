"""baselines.py — DeepONet (Lu et al., 2021) external baselines for the
multi-law curve-prediction protocol.

The LCA generalist is compared against operator-network baselines that do NOT
receive the governing equation:

  1. per-law DeepONet      : one model per law, trained only on that law's
                             samples (the standard way operator networks are
                             applied to parametric problems).
  2. pooled DeepONet + law : one model over all laws whose branch receives the
                             law as a ONE-HOT LABEL plus the parameters. This
                             is the strongest supervised baseline: it gets the
                             law identity directly, which the LCA generalist
                             never does (for beam vs cantilever the parameter
                             tokens are literally identical — only the
                             equation signature differs).

All models are evaluated with the same metric as the generalist
(curve_err = mean |pred - true| / per-sample peak, per channel), on the same
held-out problems.

usage:
  python physx/baselines.py [--threads 4] [--out paper/fig/deeponet_baselines.json]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physx import dataset, laws
else:
    from . import dataset, laws

DOMAINS = laws.SHARED_HEAD_DOMAINS
ONE_CHANNEL = {"beam", "cantilever", "rc"}
TRAJ_STEPS = 50
P_BASIS = 64


class DeepONet(nn.Module):
    """branch(params) and trunk(s) networks; u(s, c) = sum_k branch_kc trunk_k(s)."""

    def __init__(self, n_branch, n_channels=2, p_basis=P_BASIS, hidden=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_branch, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p_basis * n_channels),
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p_basis),
        )
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.n_channels = n_channels
        self.p_basis = p_basis

    def forward(self, branch_in, s):
        """branch_in (B, n_branch); s (n_q, 1) -> (B, n_q, n_channels)."""
        bk = self.branch(branch_in).view(branch_in.shape[0], self.p_basis, self.n_channels)
        tk = self.trunk(s)  # (n_q, p_basis)
        out = torch.einsum("bkc,qk->bqc", bk, tk) + self.bias
        return out


def curve_err(pred, true, n_channels=2):
    """Same metric as the generalist: per-sample peak-normalized MAE."""
    if true.shape[-1] == 1 and n_channels == 2:
        pred = pred[..., :1]
    denom = np.abs(true).max(axis=1, keepdims=True)
    denom[denom < 1e-9] = 1.0
    return float(np.mean(np.mean(np.abs(pred - true) / denom, axis=1)))


def make_batches(domain, probs, st, batch_size, seed, shuffle=True, n_channels=2):
    rng = np.random.RandomState(seed)
    idx = list(range(len(probs)))
    if shuffle:
        rng.shuffle(idx)
    keys = st["keys"]
    for i in range(0, len(idx), batch_size):
        chunk = [probs[j] for j in idx[i:i + batch_size]]
        vals = [dataset.normalize(p["params"], st) for p in chunk]
        traj = np.array([np.array(p["traj"], dtype=np.float32) for p in chunk])
        peak = np.abs(traj).max(axis=1, keepdims=True)
        peak[peak < 1e-9] = 1.0
        traj_n = traj / peak
        yield (torch.tensor(vals, dtype=torch.float32),
               torch.tensor(traj_n, dtype=torch.float32))


def make_batches_channels(domain, probs, st, batch_size, seed, shuffle=True,
                          n_channels=2):
    """Same as make_batches but pads 1-channel targets to n_channels by
    duplicating the column (fixes silent broadcasting in the MSE loss)."""
    rng = np.random.RandomState(seed)
    idx = list(range(len(probs)))
    if shuffle:
        rng.shuffle(idx)
    keys = st["keys"]
    for i in range(0, len(idx), batch_size):
        chunk = [probs[j] for j in idx[i:i + batch_size]]
        vals = [dataset.normalize(p["params"], st) for p in chunk]
        traj = np.array([np.array(p["traj"], dtype=np.float32) for p in chunk])
        peak = np.abs(traj).max(axis=1, keepdims=True)
        peak[peak < 1e-9] = 1.0
        traj_n = traj / peak
        if traj_n.shape[-1] == 1 and n_channels == 2:
            traj_n = np.concatenate([traj_n, traj_n], axis=-1)
        yield (torch.tensor(vals, dtype=torch.float32),
               torch.tensor(traj_n, dtype=torch.float32))


def run_deeponet(domain, train_p, val_p, st, epochs=250, seed=0, batch_size=32,
                 threads=4, law_onehot=None):
    """Train one DeepONet. law_onehot: (10,) vector to append to branch input
    (pooled model with a supervised law label), or None for per-law."""
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_param = len(st["keys"])
    n_branch = n_param + (len(DOMAINS) if law_onehot is not None else 0)
    n_channels = 2 if law_onehot is not None or domain not in ONE_CHANNEL else 1
    model = DeepONet(n_branch, n_channels=n_channels)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    s = torch.linspace(0.0, 1.0, TRAJ_STEPS).unsqueeze(-1)
    law_t = (torch.tensor(law_onehot, dtype=torch.float32).unsqueeze(0)
             if law_onehot is not None else None)
    bf = make_batches_channels if law_onehot is not None else make_batches
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for vals, traj_n in bf(domain, train_p, st, batch_size, seed + epoch,
                              n_channels=n_channels):
            opt.zero_grad()
            bi = vals if law_t is None else torch.cat([vals, law_t.expand(vals.shape[0], -1)], dim=-1)
            pred = model(bi, s)
            loss = nn.functional.mse_loss(pred, traj_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
    model.eval()
    with torch.no_grad():
        errs = []
        for vals, traj_n in bf(domain, val_p, st, 64, seed=0, shuffle=False,
                               n_channels=n_channels):
            bi = vals if law_t is None else torch.cat([vals, law_t.expand(vals.shape[0], -1)], dim=-1)
            pred = model(bi, s).numpy()
            true = traj_n.numpy()
            errs.append(curve_err(pred, true, n_channels))
    return model, float(np.mean(errs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out", default="paper/fig/deeponet_baselines.json")
    ap.add_argument("--pooled-only", action="store_true",
                    help="skip per-law training; only run the pooled+lawid model")
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    problems_path = os.path.join(here, "models", "ext", "problems_s0.json")
    with open(problems_path) as f:
        cache = json.load(f)
    problems = cache["problems"]
    per_domain = cache["per_domain"]
    n_val = max(4, per_domain // 8)
    train_p = {d: problems[d][:-n_val] for d in DOMAINS}
    val_p = {d: problems[d][-n_val:] for d in DOMAINS}
    metas = {d: dataset.stats(problems[d], d) for d in DOMAINS}

    t0 = time.time()
    results = {"domains": DOMAINS, "per_domain": per_domain, "n_val": n_val,
               "epochs": args.epochs, "per_law": {}, "pooled_lawid": None}
    if not args.pooled_only:
        for d in DOMAINS:
            _, err = run_deeponet(d, train_p[d], val_p[d], metas[d],
                                  epochs=args.epochs, threads=args.threads)
            results["per_law"][d] = {"curve_err": round(err, 5)}
            print(f"[per-law {d}] curve_err {err:.4f} ({time.time() - t0:.0f}s)", flush=True)
            with open(args.out, "w") as f:  # progressive save: never lose work
                json.dump(results, f, indent=1)

    # pooled DeepONet with the law as a one-hot label
    st_pool = {d: metas[d] for d in DOMAINS}
    all_train, all_val = [], []
    for d in DOMAINS:
        for p in train_p[d]:
            all_train.append((d, p))
        for p in val_p[d]:
            all_val.append((d, p))
    rng = np.random.RandomState(0)
    rng.shuffle(all_train)
    # train a single branch with per-domain normalization: pad values, append
    # the one-hot law id
    law_ids = {d: i for i, d in enumerate(DOMAINS)}
    n_param_max = max(len(st_pool[d]["keys"]) for d in DOMAINS)

    def build_branch(domain, p):
        vals = dataset.normalize(p["params"], metas[domain])
        # pad parameter values to a fixed width so every law maps to the same
        # branch input dimensionality (zero-padding = absent parameter)
        vals = vals + [0.0] * (n_param_max - len(vals))
        onehot = [0.0] * len(DOMAINS)
        onehot[law_ids[domain]] = 1.0
        return vals + onehot

    torch.manual_seed(0)
    np.random.seed(0)
    if args.threads:
        torch.set_num_threads(args.threads)
    n_branch = n_param_max + len(DOMAINS)
    model = DeepONet(n_branch, n_channels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    s = torch.linspace(0.0, 1.0, TRAJ_STEPS).unsqueeze(-1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng2 = np.random.RandomState(epoch)
        rng2.shuffle(all_train)
        total = 0.0
        for i in range(0, len(all_train), 32):
            chunk = all_train[i:i + 32]
            bi = torch.tensor([build_branch(d, p) for d, p in chunk], dtype=torch.float32)
            trajs = [np.array(p["traj"], dtype=np.float32) for _, p in chunk]
            # pad 1-channel domains (beam, cantilever, rc) to 2 channels
            trajs = [t if t.shape[-1] == 2 else np.concatenate([t, t], axis=-1)
                     for t in trajs]
            traj = np.stack(trajs, axis=0)
            peak = np.abs(traj).max(axis=1, keepdims=True)
            peak[peak < 1e-9] = 1.0
            traj_n = torch.tensor(traj / peak, dtype=torch.float32)
            opt.zero_grad()
            pred = model(bi, s)
            loss = nn.functional.mse_loss(pred, traj_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
    model.eval()
    pooled_errs = {}
    with torch.no_grad():
        for d in DOMAINS:
            errs = []
            for p in val_p[d]:
                bi = torch.tensor([build_branch(d, p)], dtype=torch.float32)
                pred = model(bi, s).numpy()[0]
                true = np.array(p["traj"], dtype=np.float32)
                # peak-normalize the target exactly as in training so the
                # metric compares like for like
                peak = np.abs(true).max(axis=0, keepdims=True)
                peak[peak < 1e-9] = 1.0
                true_n = true / peak
                errs.append(curve_err(pred, true_n, model.n_channels))
            pooled_errs[d] = float(np.mean(errs))
    results["pooled_lawid"] = pooled_errs
    print(f"[pooled+lawid] median curve_err {np.median(list(pooled_errs.values())):.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

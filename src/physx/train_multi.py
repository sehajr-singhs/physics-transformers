"""train_multi.py — train the multi-law generalist (PhysFormerLCA).

One transformer, shared body AND shared output heads, across several physical
laws. Each sample carries the governing equation itself as a binary operator
signature (laws.py); Law-Conditioned Attention injects that signature into
every layer. Input tokens are labeled by physical quantity (length, force,
modulus, ...) from a shared vocabulary, so beam and cantilever present the
IDENTICAL token sequence — the operator signature is the only signal about
which law is in force. The physics-consistency loss is still computed per
sample from its own governing equation.

Normalization is per domain (param z-score, answer z-score, per-channel peak
trajectory normalization) so a single head can serve all laws; the signature
lets the model invert the right normalization per sample.

usage:
  python physx/train_multi.py --law real --seed 0 --save models/lca_real_s0.pt
  python physx/train_multi.py --law dummy --seed 0 --save models/lca_dummy_s0.pt
  python physx/train_multi.py --eval-only --load models/lca_real_s0.pt \
      --out paper/fig/multi_law_data.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physx import dataset, laws, residuals
    from physx.physformer import PhysFormerLCA, build
    from physx.train import evaluate as _evaluate_single
else:
    from . import dataset, laws, residuals
    from .physformer import PhysFormerLCA, build
    from .train import evaluate as _evaluate_single

DOMAINS_6 = laws.SHARED_HEAD_DOMAINS_6   # the original 6 laws (13-token vocab)
DOMAINS_EXT = laws.SHARED_HEAD_DOMAINS    # the 10-law extension (15-token vocab)
TRAJ_STEPS = 50
TRAJ_DIM = 2
ONE_CHANNEL = {"beam", "cantilever", "rc"}  # domains predicting a single column

# input tokens are labeled by PHYSICAL QUANTITY (shared vocabulary), so beam
# and cantilever present the identical token sequence; the only per-sample
# signal about which law applies is the operator signature (laws.py)
# The original 6-law models were trained with the first 13 quantity tokens;
# the extended vocabulary (damping, inductance) is used only by --ext runs.
N_PARAMS_6 = 13
N_PARAMS_EXT = len(laws.QUANTITY_VOCAB)

# unified parameter embedding for the few-shot / law-swap setups: each domain's
# parameters occupy a contiguous offset block in one table
TOTAL_PARAMS = sum(len(dataset.RANGES[d]) for d in DOMAINS_6)


def domain_meta(domain, problems):
    """Per-domain normalization stats."""
    st = dataset.stats(problems, domain)
    ans_stats = dataset.answer_stats(problems, domain)
    return st, ans_stats


def make_domain_batches(domain, problems, st, ans_stats, batch_size, seed, shuffle=True):
    """Yield (pids, vals, y, traj_n, peaks, params, law_sig) for one domain."""
    rng = np.random.RandomState(seed)
    idx = list(range(len(problems)))
    if shuffle:
        rng.shuffle(idx)
    qids = laws.DOMAIN_QUANTITIES[domain]
    keys = st["keys"]
    sig = torch.tensor(laws.signature(domain), dtype=torch.float32)
    for i in range(0, len(idx), batch_size):
        chunk = [problems[j] for j in idx[i:i + batch_size]]
        pids = [qids for _ in chunk]
        vals = [dataset.normalize(p["params"], st) for p in chunk]
        y = np.array([(dataset.answer_transform(domain, p["answer"]) - ans_stats[0])
                      / ans_stats[1] for p in chunk], dtype=np.float32)
        traj = np.array([np.array(p["traj"], dtype=np.float32) for p in chunk])
        # per-channel peak normalization (channels live in [-1, 1] after this)
        peak = np.abs(traj).max(axis=1, keepdims=True)          # (B, 1, C)
        peak[peak < 1e-9] = 1.0
        traj_n = traj / peak
        params = {k: torch.tensor([p["params"][k] for p in chunk], dtype=torch.float32)
                  for k in keys}
        yield (
            torch.tensor(pids, dtype=torch.long),
            torch.tensor(vals, dtype=torch.float32),
            torch.tensor(y),
            torch.tensor(traj_n, dtype=torch.float32),
            torch.tensor(peak, dtype=torch.float32),
            params,
            sig.expand(len(chunk), -1),
        )


def train_multi(domains=DOMAINS_6, n_params=N_PARAMS_6, law="real", epochs=120, per_domain=56, batch_size=32,
                seed=0, lr=1e-3, w_ans=1.0, w_traj=1.0, w_phys=0.05,
                d_model=48, n_layers=3, traj_hidden=64, threads=None, problems=None):
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_val = max(4, per_domain // 8)
    n_tr = per_domain - n_val
    if problems is None:
        problems = {d: dataset.generate(d, n=per_domain, seed=seed + 100 * i)
                    for i, d in enumerate(domains)}
    probs = problems
    metas = {d: domain_meta(d, probs[d]) for d in domains}
    train_p = {d: probs[d][:-n_val] for d in domains}
    val_p = {d: probs[d][-n_val:] for d in domains}

    model = PhysFormerLCA(
        domains, laws.VOCAB_SIZE, law_mode=law, n_params=n_params,
        d_model=d_model, n_layers=n_layers, traj_hidden=traj_hidden,
        traj_steps=TRAJ_STEPS, traj_dim=TRAJ_DIM,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def run_epoch(shuffle, seed_, phys_w):
        model.train() if shuffle else model.eval()
        total = 0.0
        n_steps = 0
        for d in domains:
            st, ans_stats = metas[d]
            for pids, vals, y, traj_n, peak, params, sig in make_domain_batches(
                    d, train_p[d] if shuffle else val_p[d], st, ans_stats,
                    batch_size, seed_, shuffle=shuffle):
                if shuffle:
                    opt.zero_grad()
                law_sig = sig if law == "real" else None
                ans_pred, traj_pred = model(pids, vals, law_sig)
                loss_ans = torch.nn.functional.mse_loss(ans_pred, y)
                if d in ONE_CHANNEL:
                    loss_traj = torch.nn.functional.mse_loss(traj_pred[..., 0], traj_n[..., 0])
                else:
                    loss_traj = torch.nn.functional.mse_loss(traj_pred, traj_n)
                traj_real = traj_pred * peak
                phys_b = model.physics_residual(d, traj_real, params)
                phys = phys_b.mean()
                phys_norm = phys / (phys.detach().abs() + 1e-6)
                loss = w_ans * loss_ans + w_traj * loss_traj + phys_w * phys_norm
                if shuffle:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total += loss.item()
                n_steps += 1
        return total / max(1, n_steps)

    for epoch in range(1, epochs + 1):
        phys_w = w_phys * min(1.0, epoch / 10)
        tr = run_epoch(True, seed + epoch, phys_w)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                res = eval_multi(model, metas, val_p, domains, law=law)
            avg = np.mean([r["ans_rel_mae"] for r in res.values()])
            print(f"[multi:{law}] epoch {epoch:3d}/{epochs} train_loss {tr:.4f} "
                  f"val_avg_ans {avg:.4f}", flush=True)
    return model, metas


@torch.no_grad()
def eval_multi(model, metas, problems, domains=DOMAINS_6, law="real", batch_size=64):
    """Per-domain held-out evaluation: answer rel error, curve error, residual."""
    model.eval()
    out = {}
    for d in domains:
        st, ans_stats = metas[d]
        probs = problems[d]
        errs, curves, phys = [], [], []
        for pids, vals, y, traj_n, peak, params, sig in make_domain_batches(
                d, probs, st, ans_stats, batch_size, seed=0, shuffle=False):
            law_sig = sig if law == "real" else None
            ans_pred, traj_pred = model(pids, vals, law_sig)
            preds = np.array([dataset.answer_inverse(d, (a.item() * ans_stats[1] + ans_stats[0]))
                              for a in ans_pred])
            true = np.array([p["answer"] for p in probs])
            scale = np.abs(true)
            scale[scale == 0] = 1.0
            errs.append(np.mean(np.abs(preds - true) / scale))
            traj_real = (traj_pred * peak).numpy()
            t_true = np.array([np.array(p["traj"], dtype=np.float32) for p in probs])
            if d in ONE_CHANNEL:
                td, tt = traj_real[..., 0], t_true[..., 0]
            else:
                td, tt = traj_real, t_true
            denom = np.abs(tt).max(axis=1, keepdims=True)
            denom[denom < 1e-9] = 1.0
            curves.append(np.mean(np.mean(np.abs(td - tt) / denom, axis=1)))
            params_t = {k: v for k, v in params.items()}
            traj_t = torch.tensor(traj_real)
            phys.append(model.physics_residual(d, traj_t, params_t).mean().item())
        out[d] = {
            "ans_rel_mae": float(np.mean(errs)),
            "curve_err": float(np.mean(curves)),
            "phys_resid": float(np.mean(phys)),
        }
    return out


def evaluate_specialists(domains, problems, models_dir):
    """Evaluate the per-domain single-law specialists on the same problems."""
    out = {}
    for d in domains:
        stats_path = os.path.join(models_dir, f"{d}.stats.json")
        pt_path = os.path.join(models_dir, f"{d}.pt")
        if not (os.path.exists(stats_path) and os.path.exists(pt_path)):
            out[d] = None
            continue
        with open(stats_path) as f:
            meta = json.load(f)
        st = meta["param_stats"]
        ans_stats = meta["answer_stats"]
        tstats = meta["traj_stats"]
        shape_norm = meta.get("traj_norm", "global") == "shape"
        model = build(d, st, traj_steps=int(np.array(problems[d][0]["traj"]).shape[0]),
                      traj_dim=2 if d in ("projectile", "pendulum", "spring") else 1,
                      traj_hidden=meta.get("arch", {}).get("traj_hidden", 64),
                      kind=meta.get("arch", {}).get("kind", "physformer"),
                      sigmoid_traj=bool(shape_norm))
        model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        model.eval()
        mae, phys = _evaluate_single(model, st, ans_stats, tstats, problems[d],
                                     shape_norm=shape_norm)
        # curve error on the same metric as the generalist
        curves = []
        with torch.no_grad():
            pids = torch.tensor([[i for i in range(len(st["keys"]))]] * len(problems[d]))
            vals = torch.tensor([dataset.normalize(p["params"], st) for p in problems[d]])
            ans_pred, traj_pred = model(pids, vals)
            tt = np.array([np.array(p["traj"], dtype=np.float32) for p in problems[d]])
            if shape_norm:
                peak = np.abs(tt).max(axis=1, keepdims=True)
                peak[peak < 1e-9] = 1.0
                traj_real = (traj_pred * torch.tensor(peak)).numpy()
            else:
                traj_real = (traj_pred * torch.tensor(tstats[1]) +
                             torch.tensor(tstats[0])).numpy()
            if d in ONE_CHANNEL:
                td, tt2 = traj_real[..., 0], tt[..., 0]
            else:
                td, tt2 = traj_real, tt
            denom = np.abs(tt2).max(axis=1, keepdims=True)
            denom[denom < 1e-9] = 1.0
            curves.append(np.mean(np.mean(np.abs(td - tt2) / denom, axis=1)))
        out[d] = {"ans_rel_mae": mae, "curve_err": float(np.mean(curves)),
                  "phys_resid": phys}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", choices=["real", "dummy"], default="real")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--per-domain", type=int, default=56)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-phys", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ext", action="store_true",
                    help="10-law extension: extended domain set + 15-token quantity vocab")
    ap.add_argument("--problems-json", default=None,
                    help="load pre-generated problems {domain: [problem]} (skips generation)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--load", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "models")
    os.makedirs(models_dir, exist_ok=True)

    domains = DOMAINS_EXT if args.ext else DOMAINS_6
    n_params = N_PARAMS_EXT if args.ext else N_PARAMS_6

    if args.eval_only:
        metas = None
        # regenerate the val problems exactly as during training
        per_domain = 56
        n_val = max(4, per_domain // 8)
        probs = {d: dataset.generate(d, n=per_domain, seed=0 + 100 * i)
                 for i, d in enumerate(domains)}
        val_p = {d: probs[d][-n_val:] for d in domains}
        metas = {d: domain_meta(d, probs[d]) for d in domains}
        model = PhysFormerLCA(domains, laws.VOCAB_SIZE, law_mode=args.law,
                              n_params=n_params)
        model.load_state_dict(torch.load(args.load, map_location="cpu"))
        gen = eval_multi(model, metas, val_p, domains, law=args.law)
        specs = evaluate_specialists(domains, val_p, models_dir)
        out = {"law": args.law, "domains": domains,
               "generalist": gen, "specialists": specs,
               "val_seed": 0, "per_domain": per_domain}
        if args.out:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
        print(json.dumps(out, indent=1))
        return 0

    t0 = time.time()
    probs = None
    if args.problems_json:
        with open(args.problems_json) as f:
            probs = json.load(f)
        if "problems" in probs and isinstance(probs["problems"], dict):
            probs = probs["problems"]
        for d in domains:
            if d not in probs:
                raise SystemExit(f"problems json missing domain {d}")
    model, metas = train_multi(domains=domains, n_params=n_params, law=args.law,
                               epochs=args.epochs, per_domain=args.per_domain,
                               batch_size=args.batch_size, seed=args.seed,
                               w_phys=args.w_phys, threads=args.threads, problems=probs)
    if args.save is None:
        sub = "ext/" if args.ext else ""
        args.save = os.path.join(models_dir, sub + f"lca_{args.law}_s{args.seed}.pt")
    torch.save(model.state_dict(), args.save)
    with open(os.path.splitext(args.save)[0] + ".stats.json", "w") as f:
        json.dump({"law": args.law, "domains": domains, "epochs": args.epochs,
                   "per_domain": args.per_domain, "seed": args.seed,
                   "trained_seconds": round(time.time() - t0, 1)}, f)
    # per-seed held-out eval (same data/protocol as eval-only)
    n_val = max(4, args.per_domain // 8)
    if probs is None:
        probs = {d: dataset.generate(d, n=args.per_domain, seed=args.seed + 100 * i)
                 for i, d in enumerate(domains)}
    val_p = {d: probs[d][-n_val:] for d in domains}
    metas = {d: domain_meta(d, probs[d]) for d in domains}
    with torch.no_grad():
        gen = eval_multi(model, metas, val_p, domains, law=args.law)
    specs = evaluate_specialists(domains, val_p, models_dir)
    ev = {"law": args.law, "seed": args.seed, "generalist": gen, "specialists": specs}
    with open(os.path.splitext(args.save)[0] + ".eval.json", "w") as f:
        json.dump(ev, f, indent=1)
    print(f"eval: {json.dumps(gen)} ({time.time() - t0:.1f}s)", flush=True)
    print(f"saved {args.save} + stats + eval ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""train_fewshot.py — data-efficient adaptation to a new governing equation.

The law-swap experiment shows the operator signature is causally active at
inference but not sufficient for full zero-shot re-targeting: swapping the
signature on a trained problem moves the prediction toward the injected law
without landing exactly on it. The complement is adaptation WITH data: does
Law-Conditioned Attention let the generalist learn a NEW law (never seen in
pretraining) from a fraction of the specialist's data?

Protocol (all budgets identical across conditions):
  pretrain   : one generalist on 5 laws (beam, projectile, pendulum, spring,
               rc), 96 samples/law, 120 epochs, real or dummy signature.
  finetune   : cantilever is held out of pretraining entirely. The pretrained
               generalist is fine-tuned on 24 cantilever samples (25% of the
               96-sample specialist budget), 40 epochs, w_phys = 0.05.
  baselines  : (a) the dummy-signature ablation under the identical protocol;
               (b) a from-scratch single-law specialist trained on the same
               24 samples for the same 40 epochs.
  evaluation : the same 12 held-out cantilever problems (seed s+100 protocol)
               for all conditions; answer rel error, curve error, residual.

usage: python physx/train_fewshot.py --law real --seed 0 [--stage pre|ft|spec]
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
    from physx.train_multi import (TOTAL_PARAMS, TRAJ_STEPS, ONE_CHANNEL,
                                   domain_meta, make_domain_batches, train_multi,
                                   eval_multi)
else:
    from . import dataset, laws, residuals
    from .physformer import PhysFormerLCA, build
    from .train_multi import (TOTAL_PARAMS, TRAJ_STEPS, ONE_CHANNEL,
                              domain_meta, make_domain_batches, train_multi,
                              eval_multi)

FIVE = ["beam", "projectile", "pendulum", "spring", "rc"]  # cantilever held out
HELD_OUT = "cantilever"
PRE_EPOCHS = 120
FT_EPOCHS = 40
PER_DOMAIN = 96
FT_N = 24            # fine-tune samples (25% of the specialist budget)
FT_SEED_OFFSET = 300  # fine-tune data stream
EVAL_SEED_OFFSET = 100  # held-out evaluation stream (standard protocol)
W_PHYS = 0.05


def finetune(model, law, ft_probs, st, ans_stats, epochs=FT_EPOCHS, seed=0,
             w_phys=W_PHYS, threads=2):
    torch.set_num_threads(threads)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, epochs + 1):
        phys_w = w_phys * min(1.0, epoch / 10)
        model.train()
        for pids, vals, y, traj_n, peak, params, sig in make_domain_batches(
                HELD_OUT, ft_probs, st, ans_stats, batch_size=24, seed=seed + epoch,
                shuffle=True):
            opt.zero_grad()
            law_sig = sig if law == "real" else None
            ans_pred, traj_pred = model(pids, vals, law_sig)
            loss_ans = torch.nn.functional.mse_loss(ans_pred, y)
            loss_traj = torch.nn.functional.mse_loss(traj_pred[..., 0], traj_n[..., 0])
            traj_real = traj_pred * peak
            phys = model.physics_residual(HELD_OUT, traj_real, params).mean()
            phys_norm = phys / (phys.detach().abs() + 1e-6)
            loss = loss_ans + loss_traj + phys_w * phys_norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def eval_held_out(model, law, eval_probs, st, ans_stats):
    """answer rel error + curve error on the fixed held-out cantilever set."""
    model.eval()
    probs = eval_probs
    errs, curves = [], []
    for pids, vals, y, traj_n, peak, params, sig in make_domain_batches(
            HELD_OUT, probs, st, ans_stats, batch_size=64, seed=0, shuffle=False):
        is_lca = hasattr(model, "lca") and model.lca is not None
        law_sig = sig if (law == "real" and is_lca) else None
        if is_lca:
            ans_pred, traj_pred = model(pids, vals, law_sig)
        else:
            ans_pred, traj_pred = model(pids, vals)
        preds = np.array([dataset.answer_inverse(HELD_OUT, (a.item() * ans_stats[1]
                                                           + ans_stats[0]))
                          for a in ans_pred])
        true = np.array([p["answer"] for p in probs])
        scale = np.abs(true)
        scale[scale == 0] = 1.0
        errs.append(np.mean(np.abs(preds - true) / scale))
        traj_real = (traj_pred * peak).numpy()[..., 0]
        t_true = np.array([np.array(p["traj"], dtype=np.float32)[:, 0] for p in probs])
        denom = np.abs(t_true).max(axis=1, keepdims=True)
        denom[denom < 1e-9] = 1.0
        curves.append(np.mean(np.mean(np.abs(traj_real - t_true) / denom, axis=1)))
    return {"ans_rel_mae": float(np.mean(errs)), "curve_err": float(np.mean(curves))}


def train_specialist(seed, ft_probs, threads=2):
    """From-scratch single-law specialist, same 24 samples / 40 epochs."""
    from physx import train as train_mod
    model, meta = train_mod.train(
        HELD_OUT, epochs=FT_EPOCHS, samples=FT_N + 4, batch_size=24, seed=seed,
        w_phys=W_PHYS, w_traj=1.0, threads=threads,
    )
    return model, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--law", choices=["real", "dummy"], default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=["pre", "ft", "spec", "all"], default="all")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "models", "fewshot")
    os.makedirs(models_dir, exist_ok=True)
    s = args.seed

    # held-out evaluation problems (fixed protocol, never trained on)
    eval_probs = dataset.generate(HELD_OUT, n=PER_DOMAIN, seed=s + EVAL_SEED_OFFSET)[-PER_DOMAIN // 8:]
    ft_probs = dataset.generate(HELD_OUT, n=PER_DOMAIN, seed=s + FT_SEED_OFFSET)[:FT_N]

    if args.stage in ("all", "pre"):
        t0 = time.time()
        model, _ = train_multi(domains=FIVE, law=args.law, epochs=PRE_EPOCHS,
                               per_domain=PER_DOMAIN, batch_size=32, seed=s,
                               w_phys=W_PHYS, threads=args.threads)
        torch.save(model.state_dict(), os.path.join(models_dir, f"{args.law}_s{s}_pre.pt"))
        print(f"[fewshot] {args.law} s{s} pretrain done ({time.time() - t0:.0f}s)", flush=True)
        if args.stage == "pre":
            return 0
        del model
        torch.set_num_threads(args.threads)

    if args.stage in ("all", "ft"):
        t0 = time.time()
        model = PhysFormerLCA(FIVE + [HELD_OUT], laws.VOCAB_SIZE, law_mode=args.law,
                              n_params=TOTAL_PARAMS, traj_hidden=64)
        model.load_state_dict(torch.load(os.path.join(models_dir, f"{args.law}_s{s}_pre.pt"),
                                         map_location="cpu", weights_only=True))
        st, ans_stats = domain_meta(HELD_OUT, ft_probs)
        model = finetune(model, args.law, ft_probs, st, ans_stats, seed=s + FT_SEED_OFFSET,
                         threads=args.threads)
        torch.save(model.state_dict(), os.path.join(models_dir, f"{args.law}_s{s}_ft.pt"))
        ev = eval_held_out(model, args.law, eval_probs, st, ans_stats)
        with open(os.path.join(models_dir, f"{args.law}_s{s}_ft.eval.json"), "w") as f:
            json.dump({"law": args.law, "seed": s, **ev}, f, indent=1)
        print(f"[fewshot] {args.law} s{s} finetune {ev} ({time.time() - t0:.0f}s)", flush=True)

    if args.stage in ("all", "spec"):
        t0 = time.time()
        model, meta = train_specialist(s, ft_probs, threads=args.threads)
        st = meta["param_stats"]
        ans_stats = meta["answer_stats"]
        ev = eval_held_out(model, "real", eval_probs, st, ans_stats)
        with open(os.path.join(models_dir, f"spec_s{s}.eval.json"), "w") as f:
            json.dump({"law": "spec", "seed": s, **ev}, f, indent=1)
        print(f"[fewshot] spec s{s} {ev} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

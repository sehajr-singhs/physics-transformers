"""run_transfer_ablations.py — decompose what transfers in few-shot law
acquisition.

Ablations of the adapted generalist (cantilever, 24 samples / 40 epochs),
each removing one hypothesized carrier of transfer:

  novocab : re-initialize the quantity-embedding table (no shared vocabulary)
  nophys  : fine-tune with the physics residual weight = 0 (no residual)
  frozen  : freeze the encoder + law-conditioned attention weights during
            fine-tuning (no attention-weight transfer)

The full adapted generalist (real) and the constant-signature control and the
from-scratch specialist come from the committed few-shot eval files. Medians
over the three pretrained seeds are reported and written to
paper_fewshot/fig/transfer_ablations.json.
"""

import argparse
import json
import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physx import dataset, laws
    from physx.physformer import PhysFormerLCA
    from physx.train_fewshot import (FIVE, HELD_OUT, PER_DOMAIN, FT_N,
                                     FT_SEED_OFFSET, EVAL_SEED_OFFSET, W_PHYS,
                                     finetune, eval_held_out)
    from physx.train_multi import TOTAL_PARAMS, domain_meta
else:
    from . import dataset, laws
    from .physformer import PhysFormerLCA
    from .train_fewshot import (FIVE, HELD_OUT, PER_DOMAIN, FT_N,
                                FT_SEED_OFFSET, EVAL_SEED_OFFSET, W_PHYS,
                                finetune, eval_held_out)
    from .train_multi import TOTAL_PARAMS, domain_meta

SEEDS = [0, 1, 2]
ABLATIONS = ["novocab", "nophys", "frozen"]


def run_one(seed, ablate, threads=2):
    torch.set_num_threads(threads)
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "models", "fewshot")
    eval_probs = dataset.generate(HELD_OUT, n=PER_DOMAIN,
                                  seed=seed + EVAL_SEED_OFFSET)[-PER_DOMAIN // 8:]
    ft_probs = dataset.generate(HELD_OUT, n=PER_DOMAIN,
                                seed=seed + FT_SEED_OFFSET)[:FT_N]

    model = PhysFormerLCA(FIVE + [HELD_OUT], laws.VOCAB_SIZE, law_mode="real",
                          n_params=TOTAL_PARAMS, traj_hidden=64)
    model.load_state_dict(torch.load(
        os.path.join(models_dir, f"real_s{seed}_pre.pt"),
        map_location="cpu", weights_only=True))

    if ablate == "novocab":
        # remove the shared quantity vocabulary: fresh embedding table
        model.param_emb = torch.nn.Embedding(model.n_params, model.d_model)
        model.param_emb.weight.data.normal_(0, 0.02)
    elif ablate == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
        for m in model.lca:
            for p in m.parameters():
                p.requires_grad = False

    st, ans_stats = domain_meta(HELD_OUT, ft_probs)
    w_phys = 0.0 if ablate == "nophys" else W_PHYS
    model = finetune(model, "real", ft_probs, st, ans_stats,
                     seed=seed + FT_SEED_OFFSET, w_phys=w_phys, threads=threads)
    ev = eval_held_out(model, "real", eval_probs, st, ans_stats)
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    results = {a: {} for a in ABLATIONS}
    for a in ABLATIONS:
        for s in seeds:
            ev = run_one(s, a, threads=args.threads)
            results[a][str(s)] = ev
            print(f"[transfer-abl] {a} s{s}: {ev}", flush=True)

    import numpy as np
    med = {a: {k: float(np.median([results[a][str(s)][k] for s in seeds]))
               for k in ("ans_rel_mae", "curve_err")} for a in ABLATIONS}
    out = {"ablations": results, "median": med}
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "paper_fewshot", "fig", "transfer_ablations.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(med, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

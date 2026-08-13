"""law_swap.py — causal steering of behavior by the governing equation.

The beam and the cantilever present the IDENTICAL token sequence to the
multi-law generalist (same physical quantities: length, force, modulus,
inertia, thickness — see laws.DOMAIN_QUANTITIES) and share identical
parameter ranges, so the only information distinguishing the two laws is the
operator signature injected by Law-Conditioned Attention.

This experiment swaps the signature AT INFERENCE TIME: a beam problem is fed
to the network with the cantilever's law signature (and vice versa). Because
tokens and values are untouched, any change in the output is *caused* by the
equation vector alone. We measure whether the model's prediction moves toward
the analytic solution of the law whose signature it was given.

  steering index  SI = (d_beam - d_cant) / (d_beam + d_cant)

where d_beam/d_cant are the normalized trajectory errors of the swapped
prediction against the beam and cantilever ground truths. SI > 0 means the
swapped prediction is closer to the *target* law's solution (the equation
steers behavior); SI < 0 means it stays with the source law. The dummy-law
ablation (constant signature) must show no steering at all.

usage: python physx/law_swap.py [--models physx/models]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physx import dataset, laws, sim
    from physx.physformer import PhysFormerLCA
    from physx.train_multi import TOTAL_PARAMS, domain_meta
else:
    from . import dataset, laws, sim
    from .physformer import PhysFormerLCA
    from .train_multi import TOTAL_PARAMS, domain_meta

STEPS = 50
PAIR = ("beam", "cantilever")
PER_DOMAIN = 96
N_VAL = PER_DOMAIN // 8


def load_model(path, law):
    model = PhysFormerLCA(laws.SHARED_HEAD_DOMAINS, laws.VOCAB_SIZE, law_mode=law,
                          n_params=TOTAL_PARAMS, traj_hidden=64)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def curve_err(pred, truth):
    """Mean abs deviation of a (S,1) trajectory, normalized by truth peak."""
    denom = np.abs(truth).max()
    denom = denom if denom > 1e-12 else 1.0
    return float(np.mean(np.abs(pred - truth) / denom))


def steer_batch(model, law, src, tgt, probs_src, meta_src, meta_tgt):
    """Swap the signature on every held-out sample of the source law.

    Returns per-sample dicts {si, ans_si, e_native, e_swap_src, e_swap_tgt}.
    """
    st_src, ans_src = meta_src
    st_tgt, ans_tgt = meta_tgt
    qids = laws.DOMAIN_QUANTITIES[src]
    sig_src = torch.tensor(laws.signature(src), dtype=torch.float32)
    sig_tgt = torch.tensor(laws.signature(tgt), dtype=torch.float32)
    rows = []
    with torch.no_grad():
        for p in probs_src:
            params = p["params"]
            pids = torch.tensor([qids], dtype=torch.long)
            vals = torch.tensor([dataset.normalize(params, st_src)], dtype=torch.float32)
            peak = float(np.abs(np.array(p["traj"])).max()) or 1.0
            # native signature
            a_n, t_n = model(pids, vals, sig_src.unsqueeze(0))
            # swapped signature (same tokens, same values)
            a_s, t_s = model(pids, vals, sig_tgt.unsqueeze(0))
            pred_native = t_n[0, :, 0].numpy() * peak
            pred_swap = t_s[0, :, 0].numpy() * peak
            truth_src = np.array(p["traj"], dtype=np.float32)[:, 0]
            truth_tgt = sim.trajectory(tgt, params, STEPS)[:, 0]
            e_native = curve_err(pred_native, truth_src)
            e_swap_src = curve_err(pred_swap, truth_src)
            e_swap_tgt = curve_err(pred_swap, truth_tgt)
            si = (e_swap_src - e_swap_tgt) / (e_swap_src + e_swap_tgt + 1e-12)
            # answer steering (log10 answers for beam/cantilever); the swapped
            # prediction is read out with the SOURCE domain's answer stats, so
            # a model that ignores the signature stays at the source answer
            ans_mean_s, ans_std_s = ans_src
            a_swap = float(dataset.answer_inverse(src, a_s[0].item() * ans_std_s + ans_mean_s))
            ans_true_src = float(p["answer"])
            ans_true_tgt = float(sim.closed(tgt, params)["answer"])
            ae_src = abs(a_swap - ans_true_src) / ans_true_src
            ae_tgt = abs(a_swap - ans_true_tgt) / ans_true_tgt
            ans_si = (ae_src - ae_tgt) / (ae_src + ae_tgt + 1e-12)
            rows.append({"si": si, "ans_si": ans_si, "e_native": e_native,
                         "e_swap_src": e_swap_src, "e_swap_tgt": e_swap_tgt,
                         "disruption": e_swap_src - e_native})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "paper", "fig", "law_swap_data.json"))
    args = ap.parse_args()

    # regenerated held-out problems, exactly as during training (seed 0/100)
    probs = {d: dataset.generate(d, n=PER_DOMAIN, seed=0 + 100 * i)
             for i, d in enumerate(PAIR)}
    metas = {d: domain_meta(d, probs[d]) for d in PAIR}
    val = {d: probs[d][-N_VAL:] for d in PAIR}

    out = {"summary": {}}
    rows_all = {"real": {"b2c": [], "c2b": []}, "dummy": {"b2c": [], "c2b": []}}
    for law in ("real", "dummy"):
        seeds = sorted({int(p.split("_s")[1].split(".")[0])
                        for p in glob.glob(os.path.join(args.models, f"lca_{law}_s*.pt"))})
        per_seed = {}
        for s in seeds:
            path = os.path.join(args.models, f"lca_{law}_s{s}.pt")
            model = load_model(path, law)
            # beam tokens + cantilever signature  ->  toward cantilever
            b2c = steer_batch(model, law, "beam", "cantilever", val["beam"],
                              metas["beam"], metas["cantilever"])
            # cantilever tokens + beam signature  ->  toward beam
            c2b = steer_batch(model, law, "cantilever", "beam", val["cantilever"],
                              metas["cantilever"], metas["beam"])
            rows_all[law]["b2c"] += b2c
            rows_all[law]["c2b"] += c2b
            all_rows = b2c + c2b
            per_seed[s] = {
                "si_median": float(np.median([r["si"] for r in all_rows])),
                "si_mean": float(np.mean([r["si"] for r in all_rows])),
                "si_pos_frac": float(np.mean([r["si"] > 0 for r in all_rows])),
                "ans_si_median": float(np.median([r["ans_si"] for r in all_rows])),
                "beam_to_cant": {
                    "si_median": float(np.median([r["si"] for r in b2c])),
                    "e_native_median": float(np.median([r["e_native"] for r in b2c])),
                    "e_swap_beam_median": float(np.median([r["e_swap_src"] for r in b2c])),
                    "e_swap_cant_median": float(np.median([r["e_swap_tgt"] for r in b2c])),
                },
                "cant_to_beam": {
                    "si_median": float(np.median([r["si"] for r in c2b])),
                    "e_native_median": float(np.median([r["e_native"] for r in c2b])),
                    "e_swap_cant_median": float(np.median([r["e_swap_src"] for r in c2b])),
                    "e_swap_beam_median": float(np.median([r["e_swap_tgt"] for r in c2b])),
                },
            }
        out[law] = {"seeds": seeds, "per_seed": per_seed}

    # pooled per-sample causal metrics, beam->cantilever direction (the clean
    # direction: cantilever truth is the larger deflection, no scale trap)
    def pooled(law, key):
        return np.array([r[key] for r in rows_all[law]["b2c"]])

    for key, label in (("si", "steering index (SI)"),
                       ("disruption", "signature-induced disruption")):
        a, b = pooled("real", key), pooled("dummy", key)
        try:
            w, p = stats.wilcoxon(a, b)
        except ValueError:
            w, p = float("nan"), float("nan")
        out["summary"][key] = {
            "real_median": float(np.median(a)), "dummy_median": float(np.median(b)),
            "real_mean": float(np.mean(a)), "dummy_mean": float(np.mean(b)),
            "wilcoxon_p": float(p), "n": int(len(a)),
        }

    # paired statistics over seeds: real vs dummy steering
    real_si = [out["real"]["per_seed"][s]["si_median"] for s in out["real"]["seeds"]]
    dummy_si = [out["dummy"]["per_seed"][s]["si_median"] for s in out["dummy"]["seeds"]]
    try:
        w, p = stats.wilcoxon(real_si, dummy_si)
    except ValueError:
        w, p = float("nan"), float("nan")
    out["summary"]["per_seed_si"] = {
        "real": real_si, "dummy": dummy_si,
        "real_median": float(np.median(real_si)),
        "dummy_median": float(np.median(dummy_si)),
        "wilcoxon_p": float(p),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 74)
    print("Causal law-swap steering (beam <-> cantilever at inference)")
    print("=" * 74)
    su = out["summary"]
    print("\npooled per-sample, beam tokens -> cantilever signature (n=%d):" % su["si"]["n"])
    for key, lab in (("si", "steering index SI"), ("disruption", "disruption")):
        r = su[key]
        print(f"  {lab:16s} real med {r['real_median']:+.3f} | dummy med {r['dummy_median']:+.3f}"
              f"   Wilcoxon p = {r['wilcoxon_p']:.4f}")
    ps = su["per_seed_si"]
    print(f"\nper-seed median SI  real {[f'{x:+.3f}' for x in ps['real']]}")
    print(f"                    dummy {[f'{x:+.3f}' for x in ps['dummy']]}"
          f"   p = {ps['wilcoxon_p']:.4f}")
    s = out["real"]["per_seed"][0]
    print("\nreal s0 detail:")
    print(f"  beam tokens + cantilever sig : native err {s['beam_to_cant']['e_native_median']:.3f}"
          f" -> swapped vs beam {s['beam_to_cant']['e_swap_beam_median']:.3f}"
          f" | vs cantilever {s['beam_to_cant']['e_swap_cant_median']:.3f}"
          f" (SI {s['beam_to_cant']['si_median']:+.3f})")
    print(f"  cant tokens + beam sig        : native err {s['cant_to_beam']['e_native_median']:.3f}"
          f" -> swapped vs cant {s['cant_to_beam']['e_swap_cant_median']:.3f}"
          f" | vs beam {s['cant_to_beam']['e_swap_beam_median']:.3f}"
          f" (SI {s['cant_to_beam']['si_median']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

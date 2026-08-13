"""solve.py — the engineering front-end of physx.

Computes a closed-form engineering answer, verifies it with an independent
numeric simulation, and (when a trained PhysFormer exists) predicts it with
the physics-adjusted transformer.

usage:
  python physx/solve.py --domain beam --params '{"L":4,"P":3000,"E":2e11,"I":5e-6,"h":0.2}'
  python physx/solve.py --json '{"domain":"beam","params":{...}}'
  python physx/solve.py --domains
"""

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    # support running as a plain script: python physx/solve.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from physx import sim, dataset
else:
    from . import sim, dataset

HERE = os.path.dirname(os.path.abspath(__file__))

# defaults for missing params (mid-range of the dataset)
DEFAULTS = {
    d: {k: (lo + hi) / 2 for k, (lo, hi) in dataset.RANGES[d].items()}
    for d in dataset.RANGES
}

QUESTIONS = {
    "projectile": "Projectile launched at {v0:.1f} m/s at {angle:.1f} deg. Find the horizontal range.",
    "pendulum": "Pendulum of length {L:.2f} m released from {theta0:.1f} deg. Find the period.",
    "spring": "Spring-mass with k = {k:.1f} N/m, m = {m:.2f} kg. Find the natural frequency.",
    "beam": "Simply supported beam, span {L:.2f} m, center point load {P:.0f} N, "
            "E = {E:.3g} Pa, I = {I:.3g} m^4. Find the maximum deflection.",
    "cantilever": "Cantilever beam, length {L:.2f} m, tip point load {P:.0f} N, "
                  "E = {E:.3g} Pa, I = {I:.3g} m^4. Find the maximum deflection.",
    "burgers": "Viscous Burgers flow, viscosity nu = {nu:.3g}, initial amplitude A = {A:.2f}, "
               "initial width sigma = {sigma:.2f}. Find the peak velocity at t = 0.4 s.",
    "rc": "RC circuit with R = {R:.3g} ohm, C = {C:.3g} F, V0 = {V0:.1f} V. Find the time constant.",
    "heat2d": "Square plate (1 m x 1 m), edges held at {A:.0f}/2 K, interior heat "
               "source with modes k = {k:.0f}, l = {l:.0f}. Find the peak temperature.",
}


def solve(domain, params):
    p = dict(DEFAULTS.get(domain, {}))
    p.update({k: float(v) for k, v in (params or {}).items()})
    cl = sim.closed(domain, p)
    vr = sim.verify(domain, p)
    question = QUESTIONS[domain].format(**p)
    return {
        "domain": domain,
        "question": question,
        "params": p,
        "answer": cl["answer"],
        "unit": cl["unit"],
        "verified": bool(vr.get("ok")),
        "residual": vr.get("residual"),
        "verify_note": {k: v for k, v in vr.items() if k not in ("ok", "residual")},
        "extras": {k: v for k, v in cl.items() if k not in ("answer", "unit")},
        "model_prediction": None,
        "model_used": None,
    }


def predict(domain, params, meta, model_path):
    """Run the trained PhysFormer on one problem (best-effort)."""
    try:
        import torch
        if __package__ in (None, ""):
            from physx.physformer import build
        else:
            from .physformer import build
    except Exception as e:  # pragma: no cover
        return None, f"torch unavailable: {e}"

    st = meta["param_stats"]
    ans_mean, ans_std = meta["answer_stats"]
    tmean, tstd = meta.get("traj_stats", ([0.0, 0.0], [1.0, 1.0]))
    traj_norm = meta.get("traj_norm", "global")

    p = dict(DEFAULTS.get(domain, {}))
    p.update({k: float(v) for k, v in (params or {}).items()})
    arch = meta.get("arch", {})
    model = build(domain, st,
                  d_model=arch.get("d_model", 48),
                  n_layers=arch.get("n_layers", 3),
                  nhead=arch.get("nhead", 4),
                  dim_ff=arch.get("dim_ff", 96),
                  traj_hidden=arch.get("traj_hidden", 64),
                  traj_steps=arch.get("traj_steps", 50),
                  kind=arch.get("kind", "physformer"),
                  sigmoid_traj=arch.get("sigmoid_traj", False))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        return None, f"model/stats mismatch (retrain): {str(e)[:160]}"
    model.eval()
    with torch.no_grad():
        pids = torch.tensor([list(range(len(st["keys"])))], dtype=torch.long)
        vals = torch.tensor([dataset.normalize(p, st)], dtype=torch.float32)
        ans, traj = model(pids, vals)
        y = float(ans[0]) * ans_std + ans_mean
        pred = dataset.answer_inverse(domain, y)
        if traj_norm == "shape":
            # trajectory head predicts the normalized shape; the scale is the
            # trajectory's own peak. For beam/cantilever that peak IS the
            # answer (max deflection); for RC it is V0 (the answer is tau);
            # for Burgers the field peaks at A at t=0, and for heat2d the peak
            # temperature is A (not exactly attained on the sampling grid), so
            # the scale is the initial amplitude / peak parameter.
            scale = float(p["V0"]) if domain == "rc" else (
                float(p["A"]) if domain in ("burgers", "heat2d") else pred)
            traj_real = traj * scale
        else:
            traj_real = traj * torch.tensor(tstd, dtype=torch.float32) \
                + torch.tensor(tmean, dtype=torch.float32)
        params_t = {k: torch.tensor([p[k]], dtype=torch.float32) for k in st["keys"]}
        resid = float(model.physics_residual(traj_real, params_t).mean())
    return pred, resid


def main(argv=None):
    ap = argparse.ArgumentParser(description="physx engineering solver")
    ap.add_argument("--domain", default=None, choices=list(dataset.RANGES))
    ap.add_argument("--params", default=None, help='JSON object of parameters')
    ap.add_argument("--json", dest="json_in", default=None, help='full JSON {"domain":..., "params":{...}}')
    ap.add_argument("--model", default=None, help="path to a trained PhysFormer .pt (auto-detected if in models/)")
    ap.add_argument("--domains", action="store_true", help="list supported domains")
    args = ap.parse_args(argv)

    if args.domains:
        print(json.dumps({"domains": list(dataset.RANGES), "params": dataset.RANGES}))
        return 0

    if args.json_in:
        blob = json.loads(args.json_in)
        domain, params = blob["domain"], blob.get("params", {})
    else:
        domain = args.domain
        params = json.loads(args.params) if args.params else {}

    if domain not in dataset.RANGES:
        print(json.dumps({"error": f"unknown domain {domain!r}; use --domains"}))
        return 2

    out = solve(domain, params)

    model_path = args.model
    if not model_path:
        cand = os.path.join(HERE, "models", f"{domain}.pt")
        if os.path.exists(cand):
            model_path = cand
    if model_path and os.path.exists(model_path):
        stats_file = os.path.splitext(model_path)[0] + ".stats.json"
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                meta = json.load(f)
            pred, resid = predict(domain, params, meta, model_path)
            out["model_prediction"] = pred
            out["model_residual"] = resid
            out["model_used"] = os.path.basename(model_path)

    print(json.dumps(out))
    return 0 if out["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())

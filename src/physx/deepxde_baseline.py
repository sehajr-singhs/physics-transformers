"""deepxde_baseline.py — external PINN baseline (DeepXDE) for the Burgers domain.

Standard practice in the PINN literature trains ONE network per problem
instance (a fresh FNN, autodiff PDE loss). That is the *most favourable*
regime for a PINN, and it is exactly what we give DeepXDE here. PhysFormer,
by contrast, is a single network trained once on 256 problem instances that
then generalizes to unseen (nu, A, sigma). Comparing the two on the same
instances is therefore a conservative test of PhysFormer.

Protocol (identical for both models, on the model grid x in [-1,1], t in [0,TF]):
  - answer error:  |peak(u_pred) - peak(u_exact)| / peak(u_exact) at t = TF
  - curve error:   max|u_pred - u_exact| / max|u_exact| over the full (x,t) field
  - physics residual: the same finite-difference residual that trains
    PhysFormer's physics layer (u_t + u u_x - nu u_xx, forward-time / central-
    space, normalized by the dominant-term magnitude), evaluated on the
    predicted field. For DeepXDE the field is sampled at the model grid points.

DeepXDE PINN details: FNN 2-50x4-1 tanh, 2000 collocation points, 200 initial,
80 boundary points, Adam 12k steps + L-BFGS 2k steps (a standard PINN
budget for viscous Burgers; identical for every instance). Boundary data are
the exact Cole-Hopf values at x = +/-1 — the same ground-truth source as
PhysFormer's training trajectories.

usage:
  DDE_BACKEND=pytorch PYTHONPATH=vendor/deepxde python physx/deepxde_baseline.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("DDE_BACKEND", "pytorch")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "deepxde"))
import deepxde as dde  # noqa: E402
import torch  # noqa: E402

from physx import sim  # noqa: E402

TF = sim.TF
NX, NT = sim.NX, sim.NT

# representative instances: (nu, A, sigma)
INSTANCES = [
    ("canonical", {"nu": 0.05, "A": 1.5, "sigma": 0.3}),
    ("mild", {"nu": 0.10, "A": 0.8, "sigma": 0.45}),
    ("shock", {"nu": 0.02, "A": 1.8, "sigma": 0.25}),
]


def fd_burgers_residual(field, nu):
    """Same finite-difference residual as residuals._burgers, in numpy."""
    u = field.reshape(NT, NX)
    dx = (sim.XR - sim.XL) / (NX - 1)
    dt = TF / (NT - 1)
    ut = (u[1:, 1:-1] - u[:-1, 1:-1]) / dt
    ux = (u[1:, 2:] - u[1:, :-2]) / (2 * dx)
    uxx = (u[1:, 2:] - 2 * u[1:, 1:-1] + u[1:, :-2]) / dx ** 2
    uu = u[1:, 1:-1]
    res = ut + uu * ux - nu * uxx
    scale = (np.abs(uu * ux) + np.abs(nu * uxx) + 1e-8).max(axis=-1)
    return float(((res / scale[:, None]) ** 2).mean())


def evaluate(field, params):
    """field: (NX*NT,) flattened on the model grid (t-major, x-fastest, the
    same order as burgers_traj). Each time slice is compared against the exact
    solution at that time; curve_err is the maximum pointwise error over the
    full (x, t) solution relative to the final-time peak."""
    nu = params["nu"]
    ts = np.linspace(0.0, TF, NT)
    exact_all = np.stack([sim.burgers_field(nu, params["A"], params["sigma"], t)
                          for t in ts])  # (NT, NX)
    field_f = field.reshape(NT, NX)
    peak_exact = float(np.max(np.abs(exact_all[-1])))
    peak_pred = float(np.max(np.abs(field_f[-1])))
    answer_err = abs(peak_pred - peak_exact) / peak_exact
    curve_err = float(np.max(np.abs(field_f - exact_all)) /
                      (peak_exact + 1e-12))
    resid = fd_burgers_residual(field, nu)
    return {"answer_err": answer_err, "curve_err": curve_err,
            "phys_resid": resid, "peak_pred": peak_pred, "peak_exact": peak_exact}


def train_deepxde(params, seed=0):
    """Train one per-instance PINN. Returns the field on the model grid."""
    torch.set_num_threads(4)
    nu, A, sigma = params["nu"], params["A"], params["sigma"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    geom = dde.geometry.Interval(-1.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, TF)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def u0(x):
        return A * np.exp(-x[:, 0:1] ** 2 / (2.0 * sigma ** 2))

    # exact time-dependent boundary values at x = +/-1 (Cole-Hopf);
    # _cole_hopf takes a scalar t, so evaluate pointwise
    ts = np.linspace(0.0, TF, 200)
    ul = np.array([sim._cole_hopf(nu, A, sigma, np.array([-1.0]), float(t))[0] for t in ts])
    ur = np.array([sim._cole_hopf(nu, A, sigma, np.array([1.0]), float(t))[0] for t in ts])

    def bc_func(x):
        side = np.sign(x[:, 0:1]).ravel()
        t = x[:, 1]
        vals = np.where(side < 0, np.interp(t, ts, ul), np.interp(t, ts, ur))
        return vals.reshape(-1, 1)

    def pde(x, u):
        du_x = dde.grad.jacobian(u, x, i=0, j=0)
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t + u * du_x - nu * du_xx

    data = dde.data.TimePDE(
        geomtime, pde,
        [dde.IC(geomtime, u0, lambda _, on_initial: on_initial),
         dde.DirichletBC(geomtime, bc_func, lambda _, on_boundary: on_boundary)],
        num_domain=2000, num_boundary=80, num_initial=200,
    )
    net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, loss_weights=[1.0, 5.0, 5.0])
    t0 = time.time()
    model.train(iterations=12000, display_every=6000)
    dde.optimizers.config.set_LBFGS_options(maxiter=2000)
    model.compile("L-BFGS")
    model.train()
    wall = time.time() - t0

    # evaluate on the model grid
    xs = np.linspace(-1.0, 1.0, NX)
    tsg = np.linspace(0.0, TF, NT)
    X = np.stack([np.tile(xs, NT), np.repeat(tsg, NX)], axis=-1)
    u_pred = model.predict(X).ravel()
    return u_pred, wall


def physformer_field(params, meta, model_path):
    """PhysFormer prediction of the flattened field + residual (identical path
    to solve.predict, but returns the field itself)."""
    from physx.physformer import build
    from physx import dataset as ds
    st = meta["param_stats"]
    ans_mean, ans_std = meta["answer_stats"]
    arch = meta.get("arch", {})
    model = build("burgers", st,
                  d_model=arch.get("d_model", 48),
                  n_layers=arch.get("n_layers", 3),
                  nhead=arch.get("nhead", 4),
                  dim_ff=arch.get("dim_ff", 96),
                  traj_hidden=arch.get("traj_hidden", 64),
                  traj_steps=arch.get("traj_steps", 200),
                  kind=arch.get("kind", "physformer"),
                  sigmoid_traj=arch.get("sigmoid_traj", False))
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    p = dict(params)
    with torch.no_grad():
        pids = torch.tensor([list(range(len(st["keys"])))], dtype=torch.long)
        vals = torch.tensor([ds.normalize(p, st)], dtype=torch.float32)
        ans, traj = model(pids, vals)
        y = float(ans[0]) * ans_std + ans_mean
        pred = ds.answer_inverse("burgers", y)
        scale = float(p["A"])
        traj_real = traj * scale
        params_t = {k: torch.tensor([p[k]], dtype=torch.float32) for k in st["keys"]}
        resid = float(model.physics_residual(traj_real, params_t).mean())
    field = traj_real[0, :, 0].numpy()
    return field, pred, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=None, choices=[n for n, _ in INSTANCES])
    args = ap.parse_args()

    model_path = os.path.join(ROOT, "physx", "models", "burgers.pt")
    stats_path = os.path.join(ROOT, "physx", "models", "burgers.stats.json")
    meta = json.load(open(stats_path)) if os.path.exists(stats_path) else None

    out_path = os.path.join(ROOT, "paper", "fig", "deepxde_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "paper", "fig", "deepxde_fields"), exist_ok=True)
    # each instance writes its own fragment; merge at the end to avoid
    # last-write-wins clobbering when instances run in parallel
    results = {"instances": {}, "model": "deepxde 1.15.0 (pytorch backend)",
               "physformer_model": os.path.basename(model_path)}

    selected = INSTANCES if not args.instance else \
        [(args.instance, dict(p)) for n, p in INSTANCES if n == args.instance]
    for name, params in selected:
        row = {"params": params}
        # DeepXDE per-instance PINN
        u_pred, wall = train_deepxde(params)
        row["deepxde"] = evaluate(u_pred, params)
        row["deepxde"]["train_wall_s"] = wall
        np.save(os.path.join(ROOT, "paper", "fig", "deepxde_fields",
                             f"deepxde_{name}.npy"), u_pred)
        # PhysFormer (single model, all instances)
        if meta:
            field, pred, resid = physformer_field(params, meta, model_path)
            ev = evaluate(field, params)
            ev["pred"] = pred
            ev["model_resid"] = resid
            row["physformer"] = ev
            np.save(os.path.join(ROOT, "paper", "fig", "deepxde_fields",
                                 f"physformer_{name}.npy"), field)
        results["instances"][name] = row
        print(f"[{name}] deepxde: answer={row['deepxde']['answer_err']*100:.2f}% "
              f"curve={row['deepxde']['curve_err']*100:.2f}% "
              f"resid={row['deepxde']['phys_resid']:.4f} ({wall:.0f}s)")
        if meta:
            p = row["physformer"]
            print(f"[{name}] physformer: answer={p['answer_err']*100:.2f}% "
                  f"curve={p['curve_err']*100:.2f}% resid={p['phys_resid']:.4f}")
        with open(os.path.join(ROOT, "paper", "fig", "deepxde_fields",
                               f"{name}.json"), "w") as f:
            json.dump(row, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

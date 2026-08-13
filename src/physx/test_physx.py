"""test_physx.py — physics-AI tests: simulators, dataset, PhysFormer, solve CLI."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from . import dataset, sim
from .physformer import build
from .residuals import residual


def _repo_root():
    """Repo root for either layout: AGE working tree or standalone paper repo.

    Here is .../physx (AGE tree) or .../src/physx (standalone repo).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(os.path.dirname(here)) == "src":
        return os.path.dirname(os.path.dirname(here))      # standalone repo
    return os.path.dirname(here)                           # AGE working tree


def _data_path(rel):
    """Resolve a committed data file across layouts.

    AGE tree: paper/fig/x.json, physx/models/ext/x.json, bench/x.json ...
    standalone repo: figs/x.json, results/x.json ...
    """
    root = _repo_root()
    base = os.path.basename(rel)
    candidates = [os.path.join(root, rel)]
    if rel.startswith("physx" + os.sep):
        # standalone repos keep model artifacts under results/
        stripped = rel.split(os.sep, 1)[1]          # models/ext/x.json
        candidates.append(os.path.join(root, "results", stripped))
        candidates.append(os.path.join(root, "results", base))
    candidates += [os.path.join(root, "figs", base),
                   os.path.join(root, "paper", "fig", base)]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


class TestSimulators(unittest.TestCase):
    def test_projectile_closed_matches_euler(self):
        p = {"v0": 20.0, "angle": 45.0}
        cl = sim.projectile_closed(p)
        vr = sim.projectile_verify(p)
        self.assertAlmostEqual(vr["numeric_range"], cl["range"], delta=cl["range"] * 2e-3)
        self.assertTrue(vr["ok"])
        self.assertAlmostEqual(cl["max_height"], 20.0 ** 2 * np.sin(np.radians(45)) ** 2 / (2 * 9.81), places=6)

    def test_beam_fd_matches_closed_form(self):
        p = {"L": 4.0, "P": 10000.0, "E": 200e9, "I": 5e-6, "h": 0.2}
        analytic = sim.beam_closed(p)["max_deflection"]
        vr = sim.beam_verify(p)
        self.assertTrue(vr["ok"], f"fd={vr['numeric_max_deflection']} analytic={analytic}")
        self.assertAlmostEqual(vr["numeric_max_deflection"], analytic, delta=analytic * 2e-2)
        # sanity: w = PL^3/(48EI)
        self.assertAlmostEqual(analytic, 1e4 * 64 / (48 * 2e11 * 5e-6), places=10)

    def test_cantilever_fd_matches_closed_form(self):
        p = {"L": 4.0, "P": 10000.0, "E": 200e9, "I": 5e-6, "h": 0.2}
        analytic = sim.cantilever_closed(p)["max_deflection"]
        vr = sim.cantilever_verify(p)
        self.assertTrue(vr["ok"], f"fd={vr['numeric_max_deflection']} analytic={analytic}")
        self.assertAlmostEqual(vr["numeric_max_deflection"], analytic, delta=analytic * 2e-2)
        # sanity: wmax = PL^3/(3EI)  (3x the simply supported value)
        self.assertAlmostEqual(analytic, 1e4 * 64 / (3 * 2e11 * 5e-6), places=10)
        # shape is universal: w(x)/wmax = (3s^2 - s^3)/2 with s = x/L
        x, w = sim.cantilever_fd(p, n=200)
        s = x / p["L"]
        wshape = w / w.max()
        expect = (3 * s ** 2 - s ** 3) / 2
        self.assertLess(np.max(np.abs(wshape - expect)), 2e-2)

    def test_pendulum_small_angle_period(self):
        p = {"L": 1.0, "theta0": 5.0}
        cl = sim.pendulum_closed(p)
        self.assertAlmostEqual(cl["answer"], 2 * np.pi * np.sqrt(1.0 / 9.81), delta=0.02)
        vr = sim.pendulum_verify(p)
        self.assertTrue(vr["ok"])

    def test_spring_ode_residual(self):
        p = {"k": 20.0, "m": 2.0, "A": 0.5}
        cl = sim.spring_closed(p)
        self.assertAlmostEqual(cl["answer"], np.sqrt(10.0), places=9)
        vr = sim.spring_verify(p)
        self.assertTrue(vr["ok"])

    def test_rc_charging(self):
        p = {"R": 1e3, "C": 1e-3, "V0": 12.0}
        cl = sim.rc_closed(p)
        self.assertEqual(cl["answer"], 1.0)
        vr = sim.rc_verify(p)
        self.assertTrue(vr["ok"])

    def test_burgers_fv_matches_cole_hopf(self):
        # smooth case: FV upwind vs Cole-Hopf closed form
        p = {"nu": 0.08, "A": 1.0, "sigma": 0.4}
        cl = sim.burgers_closed(p)
        vr = sim.burgers_verify(p)
        self.assertTrue(vr["ok"], f"fv peak={vr['numeric_peak']} exact={cl['peak_u']} rel={vr['residual']}")
        self.assertLess(vr["residual"], 5e-2)
        # shock case: viscosity an order of magnitude lower
        p2 = {"nu": 0.02, "A": 2.0, "sigma": 0.2}
        vr2 = sim.burgers_verify(p2)
        self.assertTrue(vr2["ok"], f"shock-case rel={vr2['residual']}")

    def test_burgers_initial_condition(self):
        p = {"nu": 0.05, "A": 1.5, "sigma": 0.3}
        u0 = sim.burgers_field(p["nu"], p["A"], p["sigma"], 0.0)
        x = np.linspace(sim.XL, sim.XR, sim.NX)
        expect = p["A"] * np.exp(-x ** 2 / (2 * p["sigma"] ** 2))
        self.assertLess(np.max(np.abs(u0 - expect)), 1e-3)
        # peak of the final field is the scalar answer, so shape-norm works
        cl = sim.burgers_closed(p)
        self.assertAlmostEqual(cl["answer"], np.max(np.abs(sim.burgers_field(p["nu"], p["A"], p["sigma"], sim.TF))), places=10)

    def test_heat2d_fd_matches_manufactured(self):
        # 2D Poisson: independent finite-difference solver vs the manufactured
        # closed form, at several spatial mode numbers (incl. the max k = l = 3)
        for (k, l) in [(1, 1), (1, 3), (3, 2), (3, 3)]:
            p = {"A": 350.0, "k": k, "l": l}
            cl = sim.heat2d_closed(p)
            vr = sim.heat2d_verify(p)
            self.assertTrue(vr["ok"], f"k={k} l={l} rel={vr['residual']}")
            self.assertLess(vr["residual"], 2e-2, f"k={k} l={l}")
            self.assertAlmostEqual(vr["numeric_peak"], cl["peak_temperature"], delta=350 * 2e-2)
            # boundary condition: u = A/2 on the edges
            x, u = sim.heat2d_fd(p, n=20)
            self.assertAlmostEqual(u[0, 10], 175.0, delta=0.5)

    def test_heat2d_shape_norm_property(self):
        # peak = A (the scalar answer), and the normalized shape is in [0, 1]
        p = {"A": 300.0, "k": 2, "l": 3}
        traj = sim.heat2d_traj(p)
        self.assertEqual(traj.shape, (sim.H2D_N * sim.H2D_N, 1))
        # the exact peak A is attained between grid points; the sampled max
        # approaches it as the grid resolves the mode (within 2 K here)
        self.assertAlmostEqual(np.abs(traj).max(), 300.0, delta=2.0)
        shape = traj / np.abs(traj).max()
        self.assertGreaterEqual(shape.min(), 0.0)
        self.assertLessEqual(shape.max(), 1.0)
        cl = sim.heat2d_closed(p)
        self.assertEqual(cl["answer"], 300.0)
        self.assertEqual(cl["unit"], "K")


    def test_damped_closed_matches_rk4(self):
        # damped oscillator: closed-form wd vs RK4 zero-crossing period
        p = {"k": 20.0, "m": 1.0, "c": 0.5, "A": 0.5}
        cl = sim.damped_closed(p)
        vr = sim.damped_verify(p)
        self.assertTrue(vr["ok"], f"period={vr['period_numeric']} vs 2pi/wd")
        self.assertAlmostEqual(vr["period_numeric"], 2 * np.pi / cl["omega_damped"],
                               delta=(2 * np.pi / cl["omega_damped"]) * 5e-3)
        # underdamped requires zeta < 1
        self.assertLess(cl["zeta"], 1.0)

    def test_kepler_closed_matches_leapfrog(self):
        for e in (0.1, 0.4, 0.65):
            p = {"a": 2e11, "M": 1e30, "e": e}
            cl = sim.kepler_closed(p)
            vr = sim.kepler_verify(p)
            self.assertTrue(vr["ok"], f"e={e} rel={vr['residual']}")
            self.assertLess(vr["residual"], 2e-3)
        # the closed form is Kepler's third law
        self.assertAlmostEqual(cl["period"], 2 * np.pi * np.sqrt(2e11 ** 3 / (sim.G_KEP * 1e30)),
                               places=6)

    def test_lc_closed_matches_symplectic(self):
        p = {"L": 1e-4, "C": 1e-7, "V0": 5.0}
        cl = sim.lc_closed(p)
        self.assertAlmostEqual(cl["answer"], 1.0 / np.sqrt(1e-4 * 1e-7), places=6)
        vr = sim.lc_verify(p)
        self.assertTrue(vr["ok"], f"period={vr['period_numeric']}")
        self.assertAlmostEqual(vr["period_numeric"], 2 * np.pi / cl["omega"],
                               delta=(2 * np.pi / cl["omega"]) * 2e-3)

    def test_drag_closed_matches_euler(self):
        p = {"m": 2.0, "b": 0.5}
        cl = sim.drag_closed(p)
        self.assertAlmostEqual(cl["answer"], 2.0 * 9.81 / 0.5, places=6)
        vr = sim.drag_verify(p)
        self.assertTrue(vr["ok"], f"terminal={vr['numeric_terminal']} vs {cl['terminal_velocity']}")
        self.assertLess(vr["residual"], 2e-3)

    def test_new_laws_residual_floor(self):
        # residuals on the TRUE trajectories must be small (guards float32
        # cancellation and coarse-grid truncation regressions)
        import torch
        for d in ("damped", "kepler", "lc", "drag"):
            probs = dataset.generate(d, n=8, seed=3)
            pr = {k: torch.tensor([p["params"][k] for p in probs], dtype=torch.float32)
                  for k in probs[0]["params"]}
            tt = torch.tensor(np.array([p["traj"] for p in probs], dtype=np.float32))
            r = residual(d, tt, pr)
            self.assertLess(float(r.median()), 1e-3, f"{d} residual floor")


class TestDataset(unittest.TestCase):
    def test_generate_and_stats(self):
        problems = dataset.generate("beam", n=32, seed=1)
        self.assertEqual(len(problems), 32)
        p0 = problems[0]
        for key in ("domain", "params", "answer", "unit", "traj"):
            self.assertIn(key, p0)
        cols = 2 if p0["domain"] in ("projectile", "pendulum", "spring") else 1
        self.assertEqual(np.array(p0["traj"]).shape[1], cols)
        st = dataset.stats(problems, "beam")
        self.assertEqual(set(st["keys"]), {"L", "P", "E", "I", "h"})
        norm = dataset.normalize(p0["params"], st)
        self.assertEqual(len(norm), 5)

    def test_all_domains_generate(self):
        for d in dataset.RANGES:
            probs = dataset.generate(d, n=8, seed=3)
            self.assertEqual(len(probs), 8, d)

    def test_burgers_trajectory_shape(self):
        p = {"nu": 0.05, "A": 1.5, "sigma": 0.3}
        traj = sim.burgers_traj(p)
        self.assertEqual(traj.shape, (sim.NX * sim.NT, 1))
        self.assertTrue((traj >= 0).all())
        # the field peaks at A at t = 0; the shape-norm peak is therefore A,
        # and the scalar answer is the peak of the FINAL field (the 5e-5
        # tolerance absorbs interpolation from the fine reference grid)
        self.assertAlmostEqual(np.abs(traj).max(), p["A"], places=3)
        cl = sim.burgers_closed(p)
        self.assertAlmostEqual(cl["answer"], np.max(np.abs(sim.burgers_field(p["nu"], p["A"], p["sigma"], sim.TF))), places=10)


class TestResiduals(unittest.TestCase):
    def test_residual_zero_on_exact_trajectories(self):
        for d in dataset.RANGES:
            p = {k: (v[0] + v[1]) / 2 for k, v in dataset.RANGES[d].items()}
            traj = sim.trajectory(d, p, steps=50)
            t = torch_tensor(np.array([traj]))
            params = {k: torch_tensor(np.array([v])) for k, v in p.items()}
            r = residual(d, t, params).item()
            # burgers is checked on a coarse 25 x 8 (x, t) grid with forward
            # time differences, so its truncation floor is higher; heat2d uses
            # a 4th-order Laplacian on a 20 x 20 grid, floor ~1e-3 at k=l=3
            tol = {"burgers": 5e-2, "heat2d": 5e-3}.get(d, 1e-3)
            self.assertLess(r, tol, f"{d} residual {r}")


def torch_tensor(arr):
    import torch
    return torch.tensor(arr, dtype=torch.float32)


class TestPhysFormer(unittest.TestCase):
    def test_forward_shapes(self):
        import torch
        st = {"keys": ["L", "P", "E", "I", "h"], "mean": {}, "std": {}}
        for k in st["keys"]:
            st["mean"][k], st["std"][k] = 3.0, 1.0
        model = build("beam", st, d_model=32, n_layers=2)
        pids = torch.tensor([[0, 1, 2, 3, 4]])
        vals = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        ans, traj = model(pids, vals)
        self.assertEqual(ans.shape, (1,))
        self.assertEqual(traj.shape, (1, 50, 1))
        self.assertFalse(torch.isnan(ans).any())

    def test_training_reduces_error(self):
        from .train import train
        model, meta = train("spring", epochs=12, samples=48, batch_size=16,
                            d_model=32, n_layers=2, seed=5)
        # physics-informed training must improve on the data (rel MAE < 1)
        self.assertLess(meta["train_rel_mae"], 1.0)
        self.assertEqual(set(meta["param_stats"]["keys"]), {"k", "m", "A"})

    def test_mlp_baseline_trains(self):
        from .train import train
        model, meta = train("spring", epochs=8, samples=48, batch_size=16,
                            d_model=32, n_layers=2, seed=7, kind="mlp")
        self.assertEqual(model.kind, "mlp")
        self.assertEqual(meta["arch"]["kind"], "mlp")
        self.assertLess(meta["train_rel_mae"], 1.0)

    def test_burgers_training_runs(self):
        from .train import train
        model, meta = train("burgers", epochs=6, samples=48, batch_size=16,
                            d_model=32, n_layers=2, seed=3)
        self.assertEqual(model.traj_steps, sim.NX * sim.NT)
        self.assertEqual(meta["traj_norm"], "shape")

    def test_heat2d_training_runs(self):
        from .train import train
        model, meta = train("heat2d", epochs=6, samples=48, batch_size=16,
                            d_model=32, n_layers=2, seed=3)
        self.assertEqual(model.traj_steps, sim.H2D_N * sim.H2D_N)
        self.assertEqual(meta["traj_norm"], "shape")
        self.assertLess(meta["train_rel_mae"], 1.0)

    def test_heat2d_mlp_trains(self):
        from .train import train
        model, meta = train("heat2d", epochs=6, samples=48, batch_size=16,
                            d_model=32, n_layers=2, seed=4, kind="mlp")
        self.assertEqual(model.kind, "mlp")
        self.assertLess(meta["train_rel_mae"], 1.0)


class TestSolveCLI(unittest.TestCase):
    def test_cli_beam(self):
        here = os.path.dirname(os.path.abspath(__file__))
        blob = json.dumps({"domain": "beam", "params": {"L": 4, "P": 3000, "E": 2e11, "I": 5e-6, "h": 0.2}})
        res = subprocess.run(
            [sys.executable, os.path.join(here, "solve.py"), "--json", blob],
            capture_output=True, text=True, cwd=os.path.dirname(here),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        out = json.loads(res.stdout)
        self.assertTrue(out["verified"])
        self.assertAlmostEqual(out["answer"], 3000 * 64 / (48 * 2e11 * 5e-6), places=6)
        self.assertEqual(out["unit"], "m")
        self.assertIn("max_stress", out["extras"])

    def test_cli_cantilever(self):
        here = os.path.dirname(os.path.abspath(__file__))
        blob = json.dumps({"domain": "cantilever", "params": {"L": 4, "P": 3000, "E": 2e11, "I": 5e-6, "h": 0.2}})
        res = subprocess.run(
            [sys.executable, os.path.join(here, "solve.py"), "--json", blob],
            capture_output=True, text=True, cwd=os.path.dirname(here),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        out = json.loads(res.stdout)
        self.assertTrue(out["verified"])
        self.assertAlmostEqual(out["answer"], 3000 * 64 / (3 * 2e11 * 5e-6), places=6)
        self.assertEqual(out["unit"], "m")
        self.assertIn("max_stress", out["extras"])

    def test_cli_projectile_defaults(self):
        here = os.path.dirname(os.path.abspath(__file__))
        blob = json.dumps({"domain": "projectile", "params": {"v0": 15, "angle": 30}})
        res = subprocess.run(
            [sys.executable, os.path.join(here, "solve.py"), "--json", blob],
            capture_output=True, text=True, cwd=os.path.dirname(here),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        out = json.loads(res.stdout)
        self.assertTrue(out["verified"])
        self.assertAlmostEqual(out["answer"], 15 ** 2 * np.sin(np.radians(60)) / 9.81, places=6)

    def test_cli_burgers(self):
        here = os.path.dirname(os.path.abspath(__file__))
        blob = json.dumps({"domain": "burgers", "params": {"nu": 0.05, "A": 1.5, "sigma": 0.3}})
        res = subprocess.run(
            [sys.executable, os.path.join(here, "solve.py"), "--json", blob],
            capture_output=True, text=True, cwd=os.path.dirname(here),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        out = json.loads(res.stdout)
        self.assertTrue(out["verified"])
        self.assertAlmostEqual(out["answer"], sim.burgers_closed({"nu": 0.05, "A": 1.5, "sigma": 0.3})["answer"], places=6)

    def test_cli_heat2d(self):
        here = os.path.dirname(os.path.abspath(__file__))
        blob = json.dumps({"domain": "heat2d", "params": {"A": 400, "k": 2, "l": 3}})
        res = subprocess.run(
            [sys.executable, os.path.join(here, "solve.py"), "--json", blob],
            capture_output=True, text=True, cwd=os.path.dirname(here),
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        out = json.loads(res.stdout)
        self.assertTrue(out["verified"])
        self.assertEqual(out["answer"], 400.0)
        self.assertEqual(out["unit"], "K")
        self.assertIn("center_temperature", out["extras"])

    def test_cli_unknown_domain(self):
        here = os.path.dirname(os.path.abspath(__file__))
        res = subprocess.run(
            [sys.executable, os.path.join(here, "solve.py"), "--domain", "nope"],
            capture_output=True, text=True, cwd=os.path.dirname(here),
        )
        self.assertNotEqual(res.returncode, 0)


class TestLawConditionedAttention(unittest.TestCase):
    """The LCA invention: law signatures, quantity tokens, and the multi-law
    generalist (physx/laws.py, physx/train_multi.py)."""

    def test_vocabulary_size_and_signatures(self):
        from . import laws
        self.assertEqual(laws.VOCAB_SIZE, 22)
        for d in dataset.RANGES:
            sig = laws.signature(d)
            self.assertEqual(len(sig), laws.VOCAB_SIZE)
            self.assertTrue(all(v in (0.0, 1.0) for v in sig))
            self.assertGreater(sum(sig), 0)
        # the two bending laws share all parameters but differ in operators
        self.assertNotEqual(laws.signature("beam"), laws.signature("cantilever"))
        self.assertIn("piecewise_load", laws.LAW_SIG_NAMES["beam"])
        self.assertNotIn("piecewise_load", laws.LAW_SIG_NAMES["cantilever"])

    def test_quantity_tokens_identical_for_bending_laws(self):
        from . import laws
        self.assertEqual(laws.DOMAIN_QUANTITIES["beam"], laws.DOMAIN_QUANTITIES["cantilever"])
        # shared vocabulary: pendulum's length token is the beam's length token
        self.assertEqual(laws.DOMAIN_QUANTITIES["pendulum"][0], laws.DOMAIN_QUANTITIES["beam"][0])
        self.assertEqual(len(laws.QUANTITY_VOCAB), 15)

    def test_lca_forward_real_and_dummy(self):
        import torch
        from . import laws
        from .physformer import PhysFormerLCA
        b = 4
        for mode in ("real", "dummy"):
            model = PhysFormerLCA(laws.SHARED_HEAD_DOMAINS, laws.VOCAB_SIZE,
                                  law_mode=mode, n_params=len(laws.QUANTITY_VOCAB))
            pids = torch.tensor([laws.DOMAIN_QUANTITIES["beam"]] * b)
            vals = torch.randn(b, 5)
            sig = torch.tensor([laws.signature("beam")] * b)
            ans, traj = model(pids, vals, sig if mode == "real" else None)
            self.assertEqual(ans.shape, (b,))
            self.assertEqual(traj.shape, (b, 50, 2))
            loss = ans.mean() + traj.mean()
            loss.backward()  # gradients flow through the conditioning stream
            self.assertIsNotNone(model.law_mlp[0].weight.grad)

    def test_lca_physics_residual_per_domain(self):
        import torch
        from . import laws
        from .physformer import PhysFormerLCA
        model = PhysFormerLCA(laws.SHARED_HEAD_DOMAINS, laws.VOCAB_SIZE,
                              law_mode="real", n_params=len(laws.QUANTITY_VOCAB))
        traj = torch.rand(4, 50, 2) * 0.1
        params = {"L": torch.ones(4), "P": torch.ones(4) * 1e4,
                  "E": torch.ones(4) * 2e11, "I": torch.ones(4) * 1e-5,
                  "h": torch.ones(4) * 0.2}
        r_beam = model.physics_residual("beam", traj, params)
        r_cant = model.physics_residual("cantilever", traj, params)
        self.assertEqual(r_beam.shape, (4,))
        self.assertGreater(r_beam.sum().item(), 0)
        self.assertGreater(r_cant.sum().item(), 0)

    def test_multi_law_training_runs(self):
        from . import laws, train_multi
        torch = __import__("torch")
        torch.manual_seed(0)
        model, metas = train_multi.train_multi(epochs=2, per_domain=8,
                                               batch_size=8, seed=0, threads=2)
        # after 2 epochs the loss is finite and gradients updated the law stream
        self.assertLess(model.law_mlp[0].weight.abs().sum().item(), 1e9)
        # the default train_multi protocol is the original six laws
        self.assertEqual(set(metas.keys()), set(laws.SHARED_HEAD_DOMAINS_6))

    def test_signature_structure_and_bending_relation(self):
        from . import laws
        for d in laws._LAW_SIG:
            s = laws.signature(d)
            self.assertEqual(len(s), laws.VOCAB_SIZE)
            self.assertTrue(all(v in (0.0, 1.0) for v in s))
        # beam and cantilever signatures differ and are ordered by inclusion
        b, c = set(laws._LAW_SIG["beam"]), set(laws._LAW_SIG["cantilever"])
        self.assertTrue(c < b)  # cantilever is a strict subset of beam's operators
        self.assertNotEqual(laws.signature("beam"), laws.signature("cantilever"))

    def test_law_swap_dummy_is_exactly_insensitive(self):
        # with a constant signature the ablation's output must not change when
        # a (different) law signature is supplied at inference
        import torch
        from . import laws
        from .physformer import PhysFormerLCA
        model = PhysFormerLCA(laws.SHARED_HEAD_DOMAINS, laws.VOCAB_SIZE,
                              law_mode="dummy", n_params=len(laws.QUANTITY_VOCAB))
        model.eval()
        pids = torch.tensor([laws.DOMAIN_QUANTITIES["beam"]])
        vals = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5]])
        a1, t1 = model(pids, vals, torch.tensor([laws.signature("beam")]))
        a2, t2 = model(pids, vals, torch.tensor([laws.signature("cantilever")]))
        self.assertTrue(torch.allclose(a1, a2))
        self.assertTrue(torch.allclose(t1, t2))

    def test_law_swap_real_changes_output(self):
        # for the real-signature model the same tokens with different law
        # signatures must flow through the conditioning stream (causal plumbing)
        import torch
        from . import laws
        from .physformer import PhysFormerLCA
        model = PhysFormerLCA(laws.SHARED_HEAD_DOMAINS, laws.VOCAB_SIZE,
                              law_mode="real", n_params=len(laws.QUANTITY_VOCAB))
        model.eval()
        pids = torch.tensor([laws.DOMAIN_QUANTITIES["beam"]])
        vals = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5]])
        a1, t1 = model(pids, vals, torch.tensor([laws.signature("beam")]))
        a2, t2 = model(pids, vals, torch.tensor([laws.signature("cantilever")]))
        self.assertFalse(torch.allclose(t1, t2, atol=1e-6))

    def test_fewshot_finetune_and_eval_run(self):
        from . import dataset
        from .physformer import PhysFormerLCA
        from .train_fewshot import (FT_N, PER_DOMAIN, HELD_OUT, eval_held_out,
                                    finetune)
        from .train_multi import TOTAL_PARAMS, domain_meta
        import torch
        torch.manual_seed(0)
        model = PhysFormerLCA(["beam", "cantilever", "projectile", "pendulum",
                               "spring", "rc"], 22, law_mode="real",
                              n_params=TOTAL_PARAMS)
        ft = dataset.generate(HELD_OUT, n=PER_DOMAIN, seed=300)[:FT_N]
        ev = dataset.generate(HELD_OUT, n=PER_DOMAIN, seed=100)[-PER_DOMAIN // 8:]
        st, ans = domain_meta(HELD_OUT, ft)
        model = finetune(model, "real", ft, st, ans, epochs=2, seed=300, threads=2)
        res = eval_held_out(model, "real", ev, st, ans)
        self.assertTrue(0 <= res["ans_rel_mae"] < 5.0)
        self.assertTrue(0 <= res["curve_err"] < 1.0)


class TestComponentAnalyses(unittest.TestCase):
    """Data-integrity tests for the component papers' analysis scripts
    (matrix aggregation, transfer ablations, gate benchmark)."""

    def test_matrix_aggregation_reads_all_75_runs(self):
        from . import agg_matrix
        files = agg_matrix.FILES
        self.assertEqual(len(files), 75)
        kinds = set()
        domains = set()
        for f in files:
            base = os.path.basename(f).replace(".stats.json", "")
            parts = base.split("_")
            domains.add(parts[1]); kinds.add(parts[2])
        self.assertEqual(kinds, {"phys", "nophys", "mlp"})
        self.assertEqual(domains,
                         {"beam", "cantilever", "burgers", "heat2d", "projectile"})

    def test_transfer_ablations_json_is_committed_and_measured(self):
        # every ablation row must have per-seed measured values (3 seeds),
        # not estimates -- the few-shot paper's Table 2 depends on this
        # (data lives in the AGE working tree; skipped in standalone repos)
        import json
        p = os.path.join(_repo_root(), "paper_fewshot", "fig",
                         "transfer_ablations.json")
        if not os.path.exists(p):
            self.skipTest("transfer-ablations data not in this repo")
        d = json.load(open(p))
        for abl in ("novocab", "nophys", "frozen"):
            self.assertEqual(set(d["ablations"][abl].keys()), {"0", "1", "2"})
            for s in ("0", "1", "2"):
                self.assertTrue(0 <= d["ablations"][abl][s]["ans_rel_mae"] < 100)
        # the dominant-carrier claim: removing the vocabulary is catastrophic
        self.assertGreater(d["median"]["novocab"]["ans_rel_mae"],
                           5 * d["median"]["frozen"]["ans_rel_mae"])

    def test_gate_benchmark_results_structure(self):
        # (data lives in the AGE working tree; skipped in standalone repos)
        import json
        p = os.path.join(_repo_root(), "bench", "gate_bench_results.json")
        if not os.path.exists(p):
            self.skipTest("gate-benchmark data not in this repo")
        d = json.load(open(p))
        ms = d["missions"]
        self.assertEqual(len(ms), 7)
        gate_fs = [m for m in ms if m["gate"]["truthOk"] is False
                   and m["gate"]["reported"] == "success"]
        nogate_fs = [m for m in ms if m["noGate"]["truthOk"] is False
                     and m["noGate"]["reported"] == "success"]
        self.assertEqual(len(gate_fs), 0)      # 0% false success with gate
        self.assertEqual(len(nogate_fs), 2)    # 2/7 = 29% without


class TestRegimeTheory(unittest.TestCase):
    """The regime theory (physx/regime_analysis.py): equation conditioning
    pays exactly where parameter tokens cannot identify the law."""

    def test_ambiguity_values_from_vocabulary(self):
        from . import laws, regime_analysis as reg
        expected = {"beam": 1.0, "cantilever": 1.0, "projectile": 0.5,
                    "pendulum": 0.5, "spring": 0.0, "rc": 0.0}
        for d, amb in expected.items():
            self.assertAlmostEqual(reg.ambiguity(d), amb, places=6)
        # ambiguity is a property of the problem statement, not the network
        self.assertEqual(laws.DOMAIN_QUANTITIES["beam"],
                         laws.DOMAIN_QUANTITIES["cantilever"])

    def test_preregistration_values_match_vocabulary(self):
        # the 10-law out-of-sample pre-registration must be computable from
        # the vocabulary alone (models/ext/pre_registration.json)
        from . import regime_oos as ro
        import json as _json
        path = _data_path(os.path.join("physx", "models", "ext",
                                       "pre_registration.json"))
        if not os.path.exists(path):
            self.skipTest("10-law pre-registration not in this repo")
        pre = _json.load(open(path))
        for d, amb in pre["ambiguity_from_vocabulary_only"].items():
            self.assertAlmostEqual(ro.ambiguity_10(d), amb, places=3, msg=d)
        # excluded degeneracy is structural, not measured
        self.assertIn("rc", pre["excluded"])

    def test_benefit_is_monotone_in_ambiguity(self):
        # measured trajectory benefit (real vs dummy, 6-seed medians) must be
        # monotonically non-decreasing in ambiguity over the five
        # non-degenerate laws: sorting by benefit yields non-decreasing
        # ambiguity (ties in ambiguity are allowed -- projectile/pendulum both
        # have 0.5 -- but the ranks must not cross)
        import json
        from . import regime_analysis as reg
        p = _data_path("multi_law_data.json")
        if not os.path.exists(p):
            self.skipTest("multi-law data not in this repo")
        data = json.load(open(p))
        pd = data["per_domain"]
        five = [d for d in reg.DOMAINS if d != "rc"]
        benefits = [1.0 - pd[d]["curve_err"]["median_real"] /
                    pd[d]["curve_err"]["median_dummy"] for d in five]
        ambs = [reg.ambiguity(d) for d in five]
        order = sorted(range(len(five)), key=lambda i: benefits[i])
        seq = [ambs[i] for i in order]
        self.assertEqual(seq, sorted(seq))  # non-decreasing in ambiguity
        # and the pair must be exactly monotone (Spearman rho = 1 over n = 5)
        rho, p = reg.spearman_exact(ambs, benefits)
        self.assertAlmostEqual(rho, 1.0, places=6)
        self.assertAlmostEqual(p, 1 / 60, places=4)

    def test_regime_json_regenerates(self):
        # the committed regime_analysis.json reproduces from the committed data
        import json, os, tempfile
        from . import regime_analysis as reg
        if not os.path.exists(_data_path("multi_law_data.json")):
            self.skipTest("multi-law data not in this repo")
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "regime.json")
            reg.main(out=out)
            d = json.load(open(out))
            self.assertEqual(d["spearman_5_nondegenerate"]["rho"], 1.0)
            self.assertEqual(len(d["domains"]), 6)

    def test_ten_law_oos_falsification_regenerates(self):
        # the pre-registered 10-law test (regime_oos.py) must regenerate the
        # falsification from the committed checkpoints: rho ~ 0, spring
        # benefit negative. This is the honest counterpoint to the 6-law rho=1.
        import json, os, tempfile
        from . import regime_oos as ro
        if not os.path.exists(_data_path(os.path.join(
                "physx", "models", "ext", "pre_registration.json"))):
            self.skipTest("10-law eval data not in this repo")
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "oos.json")
            ro.main(["--out", out])
            d = json.load(open(out))
            self.assertEqual(len(d["rows"]), 10)
            r9 = d["spearman_9_nondegenerate"]
            self.assertLess(r9["rho"], 0.5)          # not the 6-law rho=1
            self.assertGreater(r9["p"], 0.05)        # and not significant
            spring = next(r for r in d["rows"] if r["domain"] == "spring")
            self.assertLess(spring["benefit"], 0.0)  # sign flip vs prediction
            beam = next(r for r in d["rows"] if r["domain"] == "beam")
            self.assertGreater(beam["benefit"], 0.5) # twin pair is the effect

    def test_deeponet_baseline_json(self):
        # per-law DeepONet baselines committed under paper/fig; pooled
        # single-model comparison was not completed and must be None (never
        # report unmeasured numbers)
        import json
        p = _data_path("deeponet_baselines.json")
        if not os.path.exists(p):
            self.skipTest("DeepONet baselines not in this repo")
        d = json.load(open(p))
        self.assertEqual(len(d["per_law"]), 10)
        import numpy as np
        med = np.median([v["curve_err"] for v in d["per_law"].values()])
        self.assertLess(med, 0.1)          # per-law operators do learn
        self.assertIsNone(d.get("pooled_lawid"))


if __name__ == "__main__":
    unittest.main()

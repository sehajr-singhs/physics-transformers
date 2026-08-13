"""dataset.py — physics datasets.

Generates problems of the form (params -> answer + reference trajectory) for
each domain, sampling physically valid parameter ranges. The reference
trajectories come from the closed-form solutions, so the training data is
exact; the physics layer additionally enforces the governing equations.

Also computes normalization stats per domain so the transformer sees
well-scaled inputs.
"""

import json
import math
import random

import numpy as np

from . import sim

# parameter name -> (min, max) sampling ranges (SI units; angles in degrees)
RANGES = {
    "projectile": {"v0": (5.0, 30.0), "angle": (10.0, 80.0)},
    "pendulum": {"L": (0.5, 3.0), "theta0": (5.0, 60.0)},
    "spring": {"k": (5.0, 50.0), "m": (0.2, 5.0), "A": (0.1, 1.0)},
    "beam": {"L": (1.0, 6.0), "P": (1e3, 5e4), "E": (69e9, 210e9), "I": (1e-7, 2e-5), "h": (0.05, 0.4)},
    "cantilever": {"L": (1.0, 6.0), "P": (1e3, 5e4), "E": (69e9, 210e9), "I": (1e-7, 2e-5), "h": (0.05, 0.4)},
    "burgers": {"nu": (0.02, 0.1), "A": (0.6, 2.0), "sigma": (0.2, 0.5)},
    "rc": {"R": (1e3, 1e6), "C": (1e-6, 1e-3), "V0": (1.0, 24.0)},
    "heat2d": {"A": (100.0, 500.0), "k": (1.0, 3.0), "l": (1.0, 3.0)},
    "damped": {"k": (5.0, 50.0), "m": (0.2, 5.0), "c": (0.05, 1.5), "A": (0.1, 1.0)},
    "kepler": {"a": (1e11, 6e11), "M": (1e28, 2e30), "e": (0.05, 0.7)},
    "lc": {"L": (1e-6, 1e-3), "C": (1e-9, 1e-6), "V0": (1.0, 12.0)},
    "drag": {"m": (0.5, 10.0), "b": (0.1, 2.0)},
}

TRAJ_STEPS = 50

# answers spanning many orders of magnitude are learned in log10 space
LOG_ANSWER = {"beam", "cantilever", "rc", "kepler", "lc"}

# parameters whose physical values span orders of magnitude are sampled
# log-uniformly (component values and orbital scales are log-distributed)
LOG_UNIFORM = {"kepler": ["a", "M"], "lc": ["L", "C"]}


def answer_transform(domain, a):
    return math.log10(a) if domain in LOG_ANSWER else a


def answer_inverse(domain, y):
    return 10.0 ** y if domain in LOG_ANSWER else y


def traj_stats(problems):
    """Per-channel mean/std of trajectories (columns have very different
    scales, e.g. beam x in metres vs deflection in millimetres)."""
    arr = np.array([p["traj"] for p in problems])  # (N, T, C)
    mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(float).tolist(), std.astype(float).tolist()


def answer_stats(problems, domain):
    ys = [answer_transform(domain, p["answer"]) for p in problems]
    mean = sum(ys) / len(ys)
    std = (sum((y - mean) ** 2 for y in ys) / len(ys)) ** 0.5 or 1.0
    return mean, std


def sample_params(domain, rng):
    log_keys = LOG_UNIFORM.get(domain, ())
    out = {}
    for k, v in RANGES[domain].items():
        if k in log_keys:
            out[k] = rng.uniform(np.log(v[0]), np.log(v[1]))
            out[k] = float(np.exp(out[k]))
        else:
            out[k] = rng.uniform(*v)
    return out


def make_problem(domain, params, steps=TRAJ_STEPS):
    """One problem: params + answer + reference trajectory (as nested lists)."""
    if domain == "heat2d":
        # spatial mode numbers are physically integer-valued; rounding keeps
        # the boundary condition uniform (sine terms vanish on the edges)
        params = {k: float(round(v)) if k in ("k", "l") else float(v)
                  for k, v in params.items()}
    cl = sim.closed(domain, params)
    traj = sim.trajectory(domain, params, steps)
    return {
        "domain": domain,
        "params": {k: float(v) for k, v in params.items()},
        "answer": float(cl["answer"]),
        "unit": cl["unit"],
        "traj": traj.astype(float).tolist(),
    }


def generate(domain, n=256, seed=0):
    rng = random.Random(seed)
    return [make_problem(domain, sample_params(domain, rng)) for _ in range(n)]


def stats(problems, domain):
    """per-param mean/std from the problems (answers kept raw)."""
    keys = list(RANGES[domain])
    mean = {k: 0.0 for k in keys}
    for p in problems:
        for k in keys:
            mean[k] += p["params"][k]
    n = len(problems)
    for k in keys:
        mean[k] /= n
    std = {k: 0.0 for k in keys}
    for p in problems:
        for k in keys:
            std[k] += (p["params"][k] - mean[k]) ** 2
    for k in keys:
        std[k] = (std[k] / n) ** 0.5
        std[k] = std[k] if std[k] > 0 else 1.0
    return {"mean": mean, "std": std, "keys": keys}


def normalize(params, st):
    return [ (params[k] - st["mean"][k]) / st["std"][k] for k in st["keys"] ]


def denormalize(x, st, k):
    return x * st["std"][k] + st["mean"][k]


def save_jsonl(problems, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

"""laws.py — the operator vocabulary and law signatures behind Law-Conditioned
Attention (LCA).

The central idea of LCA is that the *governing equation itself* is part of the
model input. Each physics domain is described by a fixed symbolic vocabulary of
differential and algebraic operators (first/second time derivatives, spatial
derivatives, self-advection, energy conservation, ...). A domain's governing
equation is expressed as a sparse binary signature over this vocabulary, and
that signature is embedded and injected into every attention layer of the
transformer as a cross-attention key/value stream.

The vocabulary is shared across all domains, so the same network can be
conditioned on *different laws*: one model answers problems governed by beam
bending, projectile motion, pendulum dynamics, RC relaxation, etc. The
signature is the only per-sample information about which law is in force, so a
model trained with LCA must learn to use the equation itself — not a task id,
not a head — to interpret the parameters.

The signatures are built by hand from each governing equation (they are
symbolic facts about the physics, not learned):

    projectile : y = x tan(a) - g x^2 / (2 v0^2 cos^2 a)        (algebraic)
    pendulum   : E = 1/2 L^2 w^2 + gL(1 - cos th) = const       (conservation)
    spring     : m x'' + k x = 0                                 (harmonic ODE)
    beam       : EI w'' = M(x), M = Px/2 for x <= L/2            (bending ODE)
    cantilever : EI w'' = P(L - x)                               (bending ODE)
    rc         : V' + V/tau = V0/tau                             (relaxation ODE)
    burgers    : u_t + u u_x = nu u_xx                           (conservation law)
    heat2d     : u_xx + u_yy = f(x, y)                           (elliptic PDE)
    damped     : m x'' + c x' + k x = 0                          (damped oscillator)
    kepler     : r'' = -GM r / r^3                               (central force orbit)
    lc         : L q'' + q/C = 0                                 (LC circuit)
    drag       : m v' = mg - b v                                 (linear drag)
"""

# Fixed operator vocabulary (order matters; do not reorder — signatures are
# stored as dense vectors and the paper's Table 1 lists them by index).
LAW_VOCAB = [
    "first_time_deriv",        #  0  d/dt
    "second_time_deriv",       #  1  d^2/dt^2
    "first_space_deriv",       #  2  d/dx
    "second_space_deriv",      #  3  d^2/dx^2
    "self_advection",          #  4  u u_x
    "linear_restoring",        #  5  k x
    "energy_conservation",     #  6  E = const
    "exponential_relaxation",  #  7  V/tau
    "nonlinear_trig",          #  8  sin/cos of state
    "gravity_coupling",        #  9  g
    "algebraic_closed_form",   # 10  closed-form trajectory
    "piecewise_load",          # 11  piecewise source
    "separable_modes",         # 12  sin(k pi x) sin(l pi y)
    "diffusion",               # 13  nu u_xx
    "moment_curvature",        # 14  EI w''
    "laplacian_2d",            # 15  u_xx + u_yy
    "quadratic_in_space",      # 16  x^2
    "trig_of_param",           # 17  cos(a)
    "angular_velocity_squared",# 18  w^2
    "harmonic",                # 19  sqrt(k/m)
    "linear_in_space",         # 20  (L - x)
    "poisson_source",          # 21  f(x, y)
]

# domain -> indices of the operators in its governing equation
_LAW_SIG = {
    "projectile": [0, 9, 10, 16, 17],                    # y = x tan a - gx^2/(2v0^2 cos^2 a)
    "pendulum": [6, 8, 9, 18],                           # E = 1/2 L^2 w^2 + gL(1 - cos th)
    "spring": [1, 5, 19],                                # m x'' + k x = 0
    "beam": [3, 11, 14, 20],                             # EI w'' = M(x), M = Px/2 (x<=L/2)
    "cantilever": [3, 14, 20],                           # EI w'' = P(L - x)
    "rc": [0, 7],                                        # V' + V/tau = V0/tau
    "burgers": [0, 2, 3, 4, 13],                         # u_t + u u_x = nu u_xx
    "heat2d": [3, 12, 15, 21],                           # u_xx + u_yy = f(x, y)
    "damped": [0, 1, 5, 7, 19],                          # m x'' + c x' + k x = 0
    "kepler": [1, 6, 9],                                 # r'' = -GM r / r^3
    "lc": [1, 6, 19],                                    # L q'' + q/C = 0
    "drag": [0, 7, 9],                                   # m v' = mg - b v
}

# the same signatures as human-readable operator sets (used in Table 1)
LAW_SIG_NAMES = {
    d: [LAW_VOCAB[i] for i in inds] for d, inds in _LAW_SIG.items()
}

# governing equations, verbatim (used in Table 1 and the Methods section)
LAW_EQUATIONS = {
    "projectile": r"$y = x\tan\alpha - gx^2/(2v_0^2\cos^2\alpha)$",
    "pendulum": r"$\tfrac{1}{2}L^2\omega^2 + gL(1-\cos\theta) = \mathrm{const}$",
    "spring": r"$m\ddot{x} + kx = 0$",
    "beam": r"$EI\,w'' = M(x),\ M = \tfrac{Px}{2}\ (x\le\tfrac{L}{2})$",
    "cantilever": r"$EI\,w'' = P(L-x)$",
    "rc": r"$\dot{V} + V/\tau = V_0/\tau$",
    "burgers": r"$u_t + uu_x = \nu u_{xx}$",
    "heat2d": r"$\nabla^2 u = f(x,y)$",
    "damped": r"$m\ddot{x} + c\dot{x} + kx = 0$",
    "kepler": r"$\ddot{\mathbf{r}} = -GM\,\mathbf{r}/r^3$",
    "lc": r"$L\ddot{q} + q/C = 0$",
    "drag": r"$m\dot{v} = mg - bv$",
}

VOCAB_SIZE = len(LAW_VOCAB)


def signature(domain):
    """Dense binary signature vector (VOCAB_SIZE,) for a domain."""
    v = [0.0] * VOCAB_SIZE
    for i in _LAW_SIG[domain]:
        v[i] = 1.0
    return v


def signature_tensor(domains):
    """(len(domains), VOCAB_SIZE) tensor for a list of domains."""
    import torch
    return torch.tensor([signature(d) for d in domains], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Physical-quantity vocabulary
#
# In the multi-law generalist the input tokens are labeled by *physical
# quantity*, not by domain: a shared embedding table maps {length, force,
# modulus, ...} to vectors, so beam and cantilever present the IDENTICAL
# token sequence (length, force, modulus, inertia, thickness). Which law
# turns those quantities into an answer is supplied only by the operator
# signature above — the two channels of information (what is given, and
# which law governs it) are kept strictly separate.
# ---------------------------------------------------------------------------

QUANTITY_VOCAB = [
    "length",      # 0  L
    "force",       # 1  P
    "modulus",     # 2  E
    "inertia",     # 3  I
    "thickness",   # 4  h
    "speed",       # 5  v0
    "angle",       # 6  theta0 / launch angle
    "stiffness",   # 7  k
    "mass",        # 8  m
    "amplitude",   # 9  A
    "resistance",  # 10 R / fluid drag coefficient b
    "capacitance", # 11 C
    "voltage",     # 12 V0
    "damping",     # 13 viscous damping coefficient c
    "inductance",  # 14 L
]

QUANTITY_IDS = {q: i for i, q in enumerate(QUANTITY_VOCAB)}

# domain -> canonical ordered list of quantity token ids (one per parameter)
DOMAIN_QUANTITIES = {
    "beam": [QUANTITY_IDS["length"], QUANTITY_IDS["force"], QUANTITY_IDS["modulus"],
             QUANTITY_IDS["inertia"], QUANTITY_IDS["thickness"]],
    "cantilever": [QUANTITY_IDS["length"], QUANTITY_IDS["force"], QUANTITY_IDS["modulus"],
                    QUANTITY_IDS["inertia"], QUANTITY_IDS["thickness"]],
    "projectile": [QUANTITY_IDS["speed"], QUANTITY_IDS["angle"]],
    "pendulum": [QUANTITY_IDS["length"], QUANTITY_IDS["angle"]],
    "spring": [QUANTITY_IDS["stiffness"], QUANTITY_IDS["mass"], QUANTITY_IDS["amplitude"]],
    "rc": [QUANTITY_IDS["resistance"], QUANTITY_IDS["capacitance"], QUANTITY_IDS["voltage"]],
    "damped": [QUANTITY_IDS["stiffness"], QUANTITY_IDS["mass"], QUANTITY_IDS["amplitude"],
               QUANTITY_IDS["damping"]],
    "kepler": [QUANTITY_IDS["length"], QUANTITY_IDS["mass"], QUANTITY_IDS["angle"]],
    "lc": [QUANTITY_IDS["inductance"], QUANTITY_IDS["capacitance"], QUANTITY_IDS["voltage"]],
    "drag": [QUANTITY_IDS["mass"], QUANTITY_IDS["resistance"]],
}

# domains whose 50-step trajectories share the geometry used by the
# shared-head multi-law experiment (see train_multi.py)
SHARED_HEAD_DOMAINS = ["beam", "cantilever", "projectile", "pendulum", "spring", "rc",
                       "damped", "kepler", "lc", "drag"]

# the six original laws only (the 6-law generalist / original papers)
SHARED_HEAD_DOMAINS_6 = ["beam", "cantilever", "projectile", "pendulum", "spring", "rc"]

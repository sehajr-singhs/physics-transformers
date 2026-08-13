"""sim.py — deterministic physics simulators with closed-form ground truth.

Every domain exposes:
  closed(params)     -> dict(answer, unit, ...)   exact analytic solution
  trajectory(params) -> np.ndarray (T, 2)         reference trajectory
  verify(params)     -> dict(residual, ok, ...)   independent numeric cross-check

The numeric cross-checks (Euler/RK4 integration, finite-difference beam
solver) are deliberately implemented independently of the closed forms, so the
engineer can *verify* an answer against physics, not just against a formula.
"""

import numpy as np

G = 9.81

# ------------------------------------------------------------- projectile

def projectile_closed(p):
    v0 = float(p["v0"])
    a = np.radians(float(p.get("angle", 45.0)))
    rng = v0 ** 2 * np.sin(2 * a) / G
    flight = 2 * v0 * np.sin(a) / G
    hmax = (v0 * np.sin(a)) ** 2 / (2 * G)
    return {
        "answer": rng, "unit": "m",
        "range": rng, "flight_time": flight, "max_height": hmax,
    }


def projectile_traj(p, steps=50):
    v0 = float(p["v0"])
    a = np.radians(float(p.get("angle", 45.0)))
    flight = 2 * v0 * np.sin(a) / G
    t = np.linspace(0, flight, steps)
    x = v0 * np.cos(a) * t
    y = v0 * np.sin(a) * t - 0.5 * G * t ** 2
    return np.stack([x, y], axis=-1)


def projectile_verify(p, steps=2000):
    """Euler integration (no closed form used) vs analytic range/height."""
    v0 = float(p["v0"])
    a = np.radians(float(p.get("angle", 45.0)))
    dt = 2 * v0 * np.sin(a) / (G * steps)
    x = y = 0.0
    vx, vy = v0 * np.cos(a), v0 * np.sin(a)
    xmax = ymax = 0.0
    while y >= 0.0:
        x += vx * dt
        y += vy * dt
        vy -= G * dt
        xmax = max(xmax, x)
        ymax = max(ymax, y)
    rng = projectile_closed(p)
    rel = abs(xmax - rng["range"]) / max(rng["range"], 1e-9)
    return {"residual": rel, "ok": rel < 2e-3, "numeric_range": xmax, "numeric_max_height": ymax}


# ---------------------------------------------------------------- pendulum

def _rk4_pendulum(L, th0_deg, dt=None, tmax=None, steps=2000):
    """Integrate d2th/dt2 = -(g/L) sin(th) with RK4; returns (t, th, w).

    dt defaults to tmax/steps (2000 steps over the horizon) — RK4 phase error
    ~ (2 pi / steps)^4 ~ 1e-11, far below any reported precision, while the
    old fixed dt=1e-4 cost 60k pure-Python steps per integration. Verifiers
    may still pass an explicit dt for an independent tighter check."""
    if tmax is None:
        tmax = max(5.0, 2.0 * 2 * np.pi * np.sqrt(L / G))  # a couple of periods
    if dt is None:
        dt = tmax / steps
    th = np.radians(th0_deg)
    w = 0.0
    ts, ths, ws = [], [], []
    t = 0.0
    while t < tmax:
        ts.append(t); ths.append(th); ws.append(w)
        # RK4 for (th, w): dth/dt = w, dw/dt = -(g/L) sin(th)
        k1w, k1th = -(G / L) * np.sin(th), w
        k2w, k2th = -(G / L) * np.sin(th + 0.5 * dt * k1th), w + 0.5 * dt * k1w
        k3w, k3th = -(G / L) * np.sin(th + 0.5 * dt * k2th), w + 0.5 * dt * k2w
        k4w, k4th = -(G / L) * np.sin(th + dt * k3th), w + dt * k3w
        th += dt * (k1th + 2 * k2th + 2 * k3th + k4th) / 6.0
        w += dt * (k1w + 2 * k2w + 2 * k3w + k4w) / 6.0
        t += dt
    return np.array(ts), np.array(ths), np.array(ws)


def _first_zero(ths, ts):
    """Linear-interpolated time of the first theta zero crossing."""
    for i in range(1, len(ths)):
        if ths[i - 1] > 0 and ths[i] <= 0:
            return ts[i - 1] + (ts[i] - ts[i - 1]) * ths[i - 1] / (ths[i - 1] - ths[i])
    return None


def pendulum_numeric_period(L, th0_deg):
    ts, ths, _ = _rk4_pendulum(L, th0_deg)
    t0 = _first_zero(ths, ts)
    if t0 is None:
        return 2 * np.pi * np.sqrt(L / G)
    return 4.0 * t0


def pendulum_closed(p):
    L = float(p["L"])
    th0 = float(p.get("theta0", 30.0))
    small = 2 * np.pi * np.sqrt(L / G)
    T = pendulum_numeric_period(L, th0)
    return {
        "answer": T, "unit": "s",
        "period_numeric": T, "period_small_angle": small,
    }


def pendulum_traj(p, steps=100):
    L = float(p["L"])
    th0 = float(p.get("theta0", 30.0))
    # ONE integration to a horizon that provably covers the nonlinear period
    # (T <= 1.07 x small-angle period for th0 <= 60 deg); the period and the
    # trajectory come from the same RK4 run
    tmax = max(5.0, 2.0 * 2 * np.pi * np.sqrt(L / G))
    ts, ths, ws = _rk4_pendulum(L, th0, tmax=tmax)
    t0 = _first_zero(ths, ts)
    T = 4.0 * t0 if t0 is not None else 2 * np.pi * np.sqrt(L / G)
    idx = np.searchsorted(ts, np.linspace(0, T, steps))
    idx = np.clip(idx, 0, len(ths) - 1)
    return np.stack([ths[idx], ws[idx]], axis=-1)


def pendulum_verify(p):
    """Convergence check: the period computed at dt and dt/2 must agree."""
    L = float(p["L"])
    th0 = float(p.get("theta0", 30.0))
    T1 = pendulum_numeric_period(L, th0)
    T2 = pendulum_numeric_period(L, th0)
    # refine: recompute with smaller dt
    ts, ths, _ = _rk4_pendulum(L, th0, dt=5e-5)
    T2 = 4.0 * _first_zero(ths, ts)
    rel = abs(T1 - T2) / T1
    return {"residual": rel, "ok": rel < 1e-3, "period_numeric": T1}


# ------------------------------------------------------------------ spring

def spring_closed(p):
    k = float(p["k"]); m = float(p["m"])
    w = np.sqrt(k / m)
    return {"answer": w, "unit": "rad/s", "omega": w, "period": 2 * np.pi / w}


def spring_traj(p, steps=100):
    k = float(p["k"]); m = float(p["m"])
    A = float(p.get("A", 0.5))
    w = np.sqrt(k / m)
    t = np.linspace(0, 2 * (2 * np.pi / w), steps)
    x = A * np.cos(w * t)
    v = -A * w * np.sin(w * t)
    return np.stack([x, v], axis=-1)


def spring_verify(p, steps=400):
    """Second-order ODE residual: m*x'' + k*x = 0, by central differences."""
    k = float(p["k"]); m = float(p["m"])
    A = float(p.get("A", 0.5))
    w = np.sqrt(k / m)
    t = np.linspace(0, 2 * (2 * np.pi / w), steps)
    dt = t[1] - t[0]
    x = A * np.cos(w * t)
    xdd = np.zeros_like(x)
    xdd[1:-1] = (x[2:] - 2 * x[1:-1] + x[:-2]) / dt ** 2
    res = m * xdd + k * x
    # boundary points have one-sided xdd=0, so judge the interior only
    rel = float(np.max(np.abs(res[1:-1])) / (k * A + 1e-9))
    return {"residual": rel, "ok": rel < 1e-3}


# -------------------------------------------------------------------- beam

def beam_closed(p):
    L = float(p["L"]); P = float(p["P"])
    E = float(p["E"]); I = float(p["I"])
    wmax = P * L ** 3 / (48 * E * I)
    out = {"answer": wmax, "unit": "m", "max_deflection": wmax}
    if "h" in p:
        Mmax = P * L / 4
        sig = Mmax * float(p["h"]) / 2 / I
        out["max_stress"] = sig
        out["stress_unit"] = "Pa"
    return out


def beam_traj(p, steps=50):
    """Deflection curve w(x) for a simply supported beam, center point load.

    The x grid is fully determined by L, so the trajectory is the single
    dependent column w(x) on that grid — the model predicts only the physical
    quantity, never a redundant coordinate."""
    L = float(p["L"]); P = float(p["P"])
    E = float(p["E"]); I = float(p["I"])
    x = np.linspace(0, L, steps)
    half = L / 2
    w = np.where(
        x <= half,
        P * x * (3 * L ** 2 - 4 * x ** 2) / (48 * E * I),
        P * (L - x) * (3 * L ** 2 - 4 * (L - x) ** 2) / (48 * E * I),
    )
    return w.reshape(-1, 1)


def beam_fd(p, n=400):
    """Finite-difference solver for EI*w'''' = q, simply supported (w=0, M=0).

    Independent of the closed form — this is the "numeric simulator" the
    engineer uses to verify analytic answers.
    """
    L = float(p["L"]); P = float(p["P"])
    E = float(p["E"]); I = float(p["I"])
    x = np.linspace(0, L, n)
    dx = x[1] - x[0]
    q = np.zeros(n)
    q[n // 2] = P / dx  # concentrated load spread over one element
    A = np.zeros((n, n))
    # interior stencil d4w/dx4 ~ [1,-4,6,-4,1]/dx^4
    for i in range(2, n - 2):
        A[i, i - 2:i + 3] = [1, -4, 6, -4, 1]
    # boundaries: w=0 at ends; w''=0 at ends (ghost point w[-1] = -w[1])
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    A[1, 1] = 5; A[1, 2] = -4; A[1, 3] = 1
    A[n - 2, n - 4] = 1; A[n - 2, n - 3] = -4; A[n - 2, n - 2] = 5
    w = np.linalg.solve(A, dx ** 4 * q / (E * I))
    return x, w


def beam_verify(p):
    x, w = beam_fd(p)
    wmax = float(np.max(np.abs(w)))
    analytic = beam_closed(p)["max_deflection"]
    rel = abs(wmax - analytic) / analytic
    return {"residual": rel, "ok": rel < 2e-2, "numeric_max_deflection": wmax}


# ------------------------------------------------------------ cantilever

def cantilever_closed(p):
    """Cantilever beam, tip point load P. wmax = P L^3 / (3 E I)."""
    L = float(p["L"]); P = float(p["P"])
    E = float(p["E"]); I = float(p["I"])
    wmax = P * L ** 3 / (3 * E * I)
    out = {"answer": wmax, "unit": "m", "max_deflection": wmax}
    if "h" in p:
        Mmax = P * L  # bending moment at the fixed end
        sig = Mmax * float(p["h"]) / 2 / I
        out["max_stress"] = sig
        out["stress_unit"] = "Pa"
    return out


def cantilever_traj(p, steps=50):
    """Deflection curve w(x) for a cantilever with tip load: single dependent
    column w(x) on the true x grid (determined by L)."""
    L = float(p["L"]); P = float(p["P"])
    E = float(p["E"]); I = float(p["I"])
    x = np.linspace(0, L, steps)
    w = P * x ** 2 * (3 * L - x) / (6 * E * I)
    return w.reshape(-1, 1)


def cantilever_fd(p, n=400):
    """Independent finite-difference solver: integrate the moment equation
    EI*w'' = -M(x) = P*(L-x) twice (w(0)=0, w'(0)=0). The moment distribution
    comes from static equilibrium, never from the closed-form deflection.

    The fixed end uses a ghost point (w_{-1} = w_1 for the zero slope) so the
    ODE is enforced at x=0 and the clamped condition genuinely couples the
    interior — pinning w_0 and w_1 as separate rows would leave a spurious
    linear null mode."""
    L = float(p["L"]); P = float(p["P"])
    E = float(p["E"]); I = float(p["I"])
    x = np.linspace(0, L, n)
    dx = x[1] - x[0]
    f = P * (L - x) / (E * I)  # w''(x) = P(L-x)/(EI)
    # unknowns w_0..w_{n-1} plus a free-end ghost w_n (n+1 total) so every
    # interior index carries a second-difference stencil. The clamped end
    # uses the ghost w_{-1}=w_1 (zero slope) folded into the ODE at x=0.
    m = n + 1
    A = np.zeros((m, m)); b = np.zeros(m)
    A[0, 0] = 1                            # w(0) = 0
    A[1, 0] = -2; A[1, 1] = 2              # ODE at x=0: (w_1 - 2w_0 + w_{-1})/dx^2 = f_0
    b[1] = dx ** 2 * f[0]
    for j in range(2, m):                  # row j = ODE at i = j-1
        A[j, j - 2] = 1; A[j, j - 1] = -2; A[j, j] = 1
        b[j] = dx ** 2 * f[j - 1]
    w = np.linalg.solve(A, b)[:n]
    return x, w


def cantilever_verify(p):
    x, w = cantilever_fd(p)
    wmax = float(np.max(np.abs(w)))
    analytic = cantilever_closed(p)["max_deflection"]
    rel = abs(wmax - analytic) / analytic
    return {"residual": rel, "ok": rel < 2e-2, "numeric_max_deflection": wmax}


# ----------------------------------------------------------------- burgers
#
# Viscous Burgers equation u_t + u u_x = nu u_xx on x in [-1, 1], t in [0, TF],
# with the Gaussian initial condition u(x, 0) = A exp(-x^2 / (2 sigma^2)).
# This is the deliberately *hard* domain: a nonlinear conservation law whose
# solutions steepen into shocks (large gradients) as the viscosity decreases.
# The exact solution comes from the Cole-Hopf transform (u = -2 nu phi_x / phi
# with phi the heat evolution of phi(x,0) = exp(-(1/2nu) int_-inf^x u0)), and the
# independent verifier is a finite-volume upwind scheme written from the
# conservation form — never from the closed form.

NX = 25          # spatial grid for the model trajectory
NT = 8           # temporal grid (t = 0 .. TF)
XL, XR = -1.0, 1.0
TF = 0.4


def _cole_hopf(nu, A, sigma, x, t):
    """Exact viscous-Burgers solution u(x, t) via the Cole-Hopf transform.

    phi(x, t) = (4 pi nu t)^-1/2  int phi0(y) exp(-(x-y)^2/(4 nu t)) dy  with
    phi0(x) = exp(-(1/2nu) int_-inf^x u0). The heat evolution is evaluated by
    direct kernel quadrature (no FFT): phi0 is monotone over many orders of
    magnitude and non-periodic, so an FFT derivative would ring at the edges.
    The quadrature grid extends well beyond x so the kernel is fully covered;
    the output x grid is the small model grid (25 points), so this is cheap.
    """
    from scipy.special import erf
    if t == 0.0:
        return A * np.exp(-x ** 2 / (2.0 * sigma ** 2))
    # u(x,0) = A exp(-x^2/(2 sigma^2));  int_-inf^x u0 = A sigma sqrt(2pi) Phi(x/sigma)
    nq = 3000
    y = np.linspace(-4.5, 4.5, nq)
    dy = y[1] - y[0]
    I = A * sigma * np.sqrt(2.0 * np.pi) * 0.5 * (1.0 + erf(y / (sigma * np.sqrt(2.0))))
    phi0 = np.exp(-I / (2.0 * nu))
    # K(z) = exp(-z^2/(4 nu t)) / sqrt(4 pi nu t);  K'(z) = -(z/(2 nu t)) K(z)
    z = x[:, None] - y[None, :]
    K = np.exp(-z ** 2 / (4.0 * nu * t)) / np.sqrt(4.0 * np.pi * nu * t)
    phi = (K * phi0[None, :]).sum(axis=1) * dy
    dphi = -(z / (2.0 * nu * t) * K * phi0[None, :]).sum(axis=1) * dy
    return -2.0 * nu * dphi / np.maximum(phi, 1e-300)


def burgers_field(nu, A, sigma, t):
    """u(x, t) on the model x-grid [-1, 1] (exact, no interpolation)."""
    return _cole_hopf(nu, A, sigma, np.linspace(XL, XR, NX), t)


def burgers_closed(p):
    nu = float(p["nu"]); A = float(p["A"]); sigma = float(p["sigma"])
    u = burgers_field(nu, A, sigma, TF)
    peak = float(np.max(np.abs(u)))
    # shock sharpness: the maximum spatial gradient of the final field
    dx = (XR - XL) / (NX - 1)
    grad = np.abs(np.gradient(u, dx))
    return {
        "answer": peak, "unit": "m/s",
        "peak_u": peak, "max_gradient": float(np.max(grad)),
    }


def burgers_traj(p, steps=50):
    """Flattened u(x, t) field on the true (x, t) grid: (NX*NT, 1). The grid is
    fully determined by the domain constants, so only the physical quantity is
    predicted — never a coordinate."""
    nu = float(p["nu"]); A = float(p["A"]); sigma = float(p["sigma"])
    ts = np.linspace(0.0, TF, NT)
    rows = [burgers_field(nu, A, sigma, t) for t in ts]
    field = np.stack(rows, axis=0)  # (NT, NX)
    return field.reshape(-1, 1)


def burgers_fv(p, n=2000, nout=None):
    """Independent finite-volume upwind solver for u_t + (u^2/2)_x = nu u_xx.
    Conservation form, first-order upwind flux, explicit diffusion — written
    without any reference to the Cole-Hopf closed form."""
    nu = float(p["nu"]); A = float(p["A"]); sigma = float(p["sigma"])
    xlo, xhi = -1.5, 1.5
    xc = np.linspace(xlo, xhi, n)
    dx = xc[1] - xc[0]
    u = A * np.exp(-xc ** 2 / (2.0 * sigma ** 2))
    t = 0.0
    while t < TF:
        umax = float(np.max(np.abs(u))) + 1e-9
        dt = min(0.3 * dx / umax, 0.45 * dx ** 2 / (nu + 1e-12), TF - t)
        # upwind flux F_{i+1/2} = u_i^2/2 for u >= 0 (our solutions stay >= 0)
        flux = u ** 2 / 2.0
        adv = np.zeros_like(u)
        adv[1:-1] = (flux[1:-1] - flux[:-2]) / dx  # right side minus left side
        # boundary: zero flux through the walls (u ~ 0 there)
        adv[0] = flux[1] / dx
        adv[-1] = -flux[-2] / dx
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
        u = u - dt * adv + nu * dt * lap
        t += dt
    if nout is None:
        return np.interp(np.linspace(XL, XR, NX), xc, u)
    return np.interp(np.linspace(XL, XR, nout), xc, u)


def burgers_verify(p):
    """Finite-volume vs Cole-Hopf reference on the model x-grid."""
    u_fv = burgers_fv(p)
    u_ref = burgers_field(float(p["nu"]), float(p["A"]), float(p["sigma"]), TF)
    rel = float(np.max(np.abs(u_fv - u_ref)) / (np.max(np.abs(u_ref)) + 1e-9))
    return {"residual": rel, "ok": rel < 5e-2, "numeric_peak": float(np.max(u_fv))}


# ---------------------------------------------------------------------- rc

def rc_closed(p):
    R = float(p["R"]); C = float(p["C"])
    return {"answer": R * C, "unit": "s", "tau": R * C}


def rc_traj(p, steps=100):
    R = float(p["R"]); C = float(p["C"])
    V0 = float(p.get("V0", 12.0))
    tau = R * C
    t = np.linspace(0, 4 * tau, steps)
    V = V0 * (1 - np.exp(-t / tau))
    # t grid is fully determined by tau; predict only V(t)
    return V.reshape(-1, 1)


def rc_verify(p, steps=4000):
    """Euler integration of dV/dt = (V0 - V)/tau vs the analytic curve at t=4tau."""
    R = float(p["R"]); C = float(p["C"])
    V0 = float(p.get("V0", 12.0))
    tau = R * C
    dt = 4 * tau / steps
    V = 0.0
    for _ in range(steps):
        V += (V0 - V) / tau * dt
    analytic = V0 * (1 - np.exp(-4.0))
    rel = abs(V - analytic) / V0
    return {"residual": rel, "ok": rel < 2e-3, "numeric_final_voltage": V, "analytic_final_voltage": analytic}


# ---------------------------------------------------------------- heat2d
#
# Two-dimensional steady-state heat conduction on the unit square (Poisson
# equation, nabla^2 u = f). This is the deliberately *multi-dimensional*
# domain: the trajectory is a full 2D temperature field, not a 1D curve.
# Solutions are generated by the method of manufactured solutions:
#
#     u*(x, y) = (A/2) (1 + sin(k pi x) sin(l pi y))        peak = A K
#     f(x, y)  = -(A pi^2 (k^2 + l^2) / 2) sin(k pi x) sin(l pi y)
#
# with uniform Dirichlet BC u = A/2 on the boundary (the sine terms vanish on
# the edges). Physically this is a square plate whose edges are held at A/2 K
# while a distributed volumetric source heats the interior. The exact field
# is the manufactured u*, and the independent verifier is a finite-difference
# Poisson solver written from the PDE and the source term alone — it never
# sees the closed form.

H2D_N = 20  # model grid: 20 x 20 = 400 field points


def _heat2d_shape(k, l, x, y):
    """Normalized shape (1 + sin sin) / 2 in [0, 1]; scale = peak A."""
    return (1.0 + np.sin(k * np.pi * x) * np.sin(l * np.pi * y)) / 2.0


def heat2d_closed(p):
    A = float(p["A"]); k = float(p["k"]); l = float(p["l"])
    x = np.linspace(0.0, 1.0, H2D_N)
    y = np.linspace(0.0, 1.0, H2D_N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = A * _heat2d_shape(k, l, X, Y)
    # peak temperature is A (attained where sin(k pi x) sin(l pi y) = 1); the
    # center probe temperature is a useful second quantity
    center = float(u[H2D_N // 2, H2D_N // 2])
    return {
        "answer": A, "unit": "K", "peak_temperature": A,
        "center_temperature": center,
    }


def heat2d_traj(p, steps=None):
    """Flattened 2D temperature field on the true 16x16 grid: (256, 1). The
    grid is fixed domain geometry, so only the physical quantity is predicted."""
    A = float(p["A"]); k = float(p["k"]); l = float(p["l"])
    x = np.linspace(0.0, 1.0, H2D_N)
    y = np.linspace(0.0, 1.0, H2D_N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = A * _heat2d_shape(k, l, X, Y)
    return u.reshape(-1, 1)


def heat2d_fd(p, n=48):
    """Independent finite-difference Poisson solver: 5-point stencil for
    nabla^2 u = f with Dirichlet BC u = A/2, sparse direct solve. Written from
    the PDE and the source term f(x, y) alone — never from the closed form."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    A = float(p["A"]); k = float(p["k"]); l = float(p["l"])
    x = np.linspace(0.0, 1.0, n)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = -(A * np.pi ** 2 * (k ** 2 + l ** 2) / 2.0) * np.sin(k * np.pi * X) * np.sin(l * np.pi * Y)
    N = n * n
    Lap = lil_matrix((N, N))
    for i in range(n):
        for j in range(n):
            r = i * n + j
            if i in (0, n - 1) or j in (0, n - 1):
                Lap[r, r] = 1.0  # Dirichlet row: u = A/2
            else:
                Lap[r, r] = -4.0 / dx ** 2
                Lap[r, r + 1] = 1.0 / dx ** 2
                Lap[r, r - 1] = 1.0 / dx ** 2
                Lap[r, r + n] = 1.0 / dx ** 2
                Lap[r, r - n] = 1.0 / dx ** 2
    rhs = f.reshape(-1)
    for i in range(n):
        for j in range(n):
            if i in (0, n - 1) or j in (0, n - 1):
                rhs[i * n + j] = A / 2.0
    u = spsolve(Lap.tocsr(), rhs).reshape(n, n)
    return x, u


def heat2d_verify(p):
    """Finite-difference solver vs manufactured closed form on the model grid."""
    A = float(p["A"]); k = float(p["k"]); l = float(p["l"])
    x, u = heat2d_fd(p)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u_ref = A * _heat2d_shape(k, l, X, Y)
    rel = float(np.max(np.abs(u - u_ref)) / (np.max(np.abs(u_ref)) + 1e-9))
    return {"residual": rel, "ok": rel < 2e-2, "numeric_peak": float(np.max(u))}


# ---------------------------------------------------------------- damped
#
# Damped harmonic oscillator m x'' + c x' + k x = 0, underdamped regime
# (zeta < 1). x(t) = A exp(-zeta wn t) cos(wd t). Answer = damped natural
# frequency wd. The independent verifier is an RK4 integration of the ODE
# (never the closed form), measuring the damped period from zero crossings.

def damped_closed(p):
    k = float(p["k"]); m = float(p["m"]); c = float(p["c"])
    wn = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 0.0))
    return {"answer": wd, "unit": "rad/s", "omega_damped": wd,
            "omega_natural": wn, "zeta": zeta}


def damped_traj(p, steps=50):
    k = float(p["k"]); m = float(p["m"]); c = float(p["c"])
    A = float(p.get("A", 0.5))
    wn = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 0.0))
    t = np.linspace(0.0, 4.0 * np.pi / wd, steps)  # two damped periods
    env = A * np.exp(-zeta * wn * t)
    x = env * np.cos(wd * t)
    v = env * (-zeta * wn * np.cos(wd * t) - wd * np.sin(wd * t))
    return np.stack([x, v], axis=-1)


def damped_verify(p, steps=4000):
    """RK4 of m x'' + c x' + k x = 0; damped period from zero crossings."""
    k = float(p["k"]); m = float(p["m"]); c = float(p["c"])
    A = float(p.get("A", 0.5))
    wn = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 0.0))
    dt = (4.0 * np.pi / wd) / steps
    x, v = A, 0.0
    crossings = []
    prev = x
    t = 0.0
    for _ in range(steps):
        k1v, k1x = -(c * v + k * x) / m, v
        k2v, k2x = -(c * (v + 0.5 * dt * k1v) + k * (x + 0.5 * dt * k1x)) / m, v + 0.5 * dt * k1v
        k3v, k3x = -(c * (v + 0.5 * dt * k2v) + k * (x + 0.5 * dt * k2x)) / m, v + 0.5 * dt * k2v
        k4v, k4x = -(c * (v + dt * k3v) + k * (x + dt * k3x)) / m, v + dt * k3v
        x += dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        v += dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
        t += dt
        # sign flips are spaced by the half-period (zeros of cos(wd t))
        if (prev > 0) != (x > 0):
            crossings.append(t - dt + dt * prev / (prev - x))
        prev = x
    if len(crossings) >= 2:
        T_num = 2.0 * (crossings[1] - crossings[0])
        T_ref = 2.0 * np.pi / wd
        rel = abs(T_num - T_ref) / T_ref
    else:
        T_num, rel = float("nan"), 1.0
    return {"residual": rel, "ok": rel < 5e-3, "period_numeric": T_num}


# ------------------------------------------------------------------ kepler
#
# Two-body Kepler problem: the smaller body follows an ellipse
#   r(E) = a(1 - e cos E),  x = a(cos E - e),  y = a sqrt(1-e^2) sin E,
# with mean anomaly Me = E - e sin E = n t, n = sqrt(GM/a^3). Answer = orbital
# period T = 2 pi sqrt(a^3 / (GM)). The independent verifier is a leapfrog
# integration of r'' = -GM r / r^3 (written from the ODE alone), detecting the
# return to periapsis.

G_KEP = 6.674e-11


def _kepler_E(Me, e, tol=1e-12):
    """Solve Kepler's equation E - e sin E = Me by Newton iteration."""
    E = Me + e * np.sin(Me)
    for _ in range(80):
        dE = (E - e * np.sin(E) - Me) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def kepler_closed(p):
    a = float(p["a"]); M = float(p["M"]); e = float(p.get("e", 0.05))
    T = 2.0 * np.pi * np.sqrt(a ** 3 / (G_KEP * M))
    return {"answer": T, "unit": "s", "period": T,
            "semi_major": a, "eccentricity": e}


def kepler_traj(p, steps=50):
    a = float(p["a"]); M = float(p["M"]); e = float(p.get("e", 0.05))
    T = 2.0 * np.pi * np.sqrt(a ** 3 / (G_KEP * M))
    t = np.linspace(0.0, T, steps)
    n = 2.0 * np.pi / T
    E = _kepler_E(n * t, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1.0 - e ** 2) * np.sin(E)
    return np.stack([x, y], axis=-1)


def kepler_verify(p, steps=4000):
    """Leapfrog integration of r'' = -GM r / r^3; period from periapsis return."""
    a = float(p["a"]); M = float(p["M"]); e = float(p.get("e", 0.05))
    T_ref = 2.0 * np.pi * np.sqrt(a ** 3 / (G_KEP * M))
    dt = T_ref / steps
    v0 = np.sqrt(G_KEP * M / a * (1.0 + e) / (1.0 - e))
    x, y = a * (1.0 - e), 0.0
    vx, vy = 0.0, v0
    t = 0.0
    prev_y = 0.0
    found = None
    for _ in range(steps):
        r = np.hypot(x, y)
        ax = -G_KEP * M * x / r ** 3
        ay = -G_KEP * M * y / r ** 3
        vx += 0.5 * ax * dt
        vy += 0.5 * ay * dt
        x += vx * dt
        y += vy * dt
        r = np.hypot(x, y)
        ax = -G_KEP * M * x / r ** 3
        ay = -G_KEP * M * y / r ** 3
        vx += 0.5 * ax * dt
        vy += 0.5 * ay * dt
        t += dt
        # the first downward y crossing after launch is the apoapsis (t = T/2)
        if t > 0.15 * T_ref and prev_y > 0 and y <= 0:
            found = 2.0 * (t - dt + dt * prev_y / (prev_y - y))
            break
        prev_y = y
    if found is None:
        return {"residual": 1.0, "ok": False, "period_numeric": float("nan")}
    rel = abs(found - T_ref) / T_ref
    return {"residual": rel, "ok": rel < 2e-3, "period_numeric": found}


# ---------------------------------------------------------------------- lc
#
# Ideal LC circuit: L q'' + q/C = 0 (the electrical harmonic oscillator).
# q(t) = C V0 cos(w t), i(t) = -C V0 w sin(w t), w = 1/sqrt(LC). Answer =
# angular frequency w. Independent verifier: symplectic Euler integration of
# the two first-order equations L i' = -q/C, q' = i (never the closed form).

def lc_closed(p):
    L = float(p["L"]); C = float(p["C"])
    w = 1.0 / np.sqrt(L * C)
    return {"answer": w, "unit": "rad/s", "omega": w, "period": 2.0 * np.pi / w}


def lc_traj(p, steps=50):
    L = float(p["L"]); C = float(p["C"])
    V0 = float(p.get("V0", 5.0))
    w = 1.0 / np.sqrt(L * C)
    q0 = C * V0
    t = np.linspace(0.0, 4.0 * np.pi / w, steps)  # two periods
    q = q0 * np.cos(w * t)
    i = -q0 * w * np.sin(w * t)
    return np.stack([q, i], axis=-1)


def lc_verify(p, steps=4000):
    """Symplectic Euler for q' = i, i' = -q/(LC); frequency from zero crossings."""
    L = float(p["L"]); C = float(p["C"])
    V0 = float(p.get("V0", 5.0))
    w_ref = 1.0 / np.sqrt(L * C)
    dt = (4.0 * np.pi / w_ref) / steps
    q, i = C * V0, 0.0
    crossings = []
    prev = q
    t = 0.0
    for _ in range(steps):
        i += (-q / (L * C)) * dt
        q += i * dt
        t += dt
        # sign flips are spaced by the half-period (zeros of cos(w t))
        if (prev > 0) != (q > 0):
            crossings.append(t - dt + dt * prev / (prev - q))
        prev = q
    if len(crossings) >= 2:
        T_num = 2.0 * (crossings[1] - crossings[0])
        rel = abs(T_num - 2.0 * np.pi / w_ref) / (2.0 * np.pi / w_ref)
    else:
        T_num, rel = float("nan"), 1.0
    return {"residual": rel, "ok": rel < 2e-3, "period_numeric": T_num}


# -------------------------------------------------------------------- drag
#
# Free fall with linear drag m v' = mg - b v. Closed form: v(t) = v_t(1 -
# e^{-t/tau}), x(t) = v_t (t - tau(1 - e^{-t/tau})), v_t = mg/b, tau = m/b.
# Answer = terminal velocity v_t. Independent verifier: Euler integration of
# dv/dt = g - (b/m)v (never the closed form).

def drag_closed(p):
    m = float(p["m"]); b = float(p["b"])
    vt = m * G / b
    return {"answer": vt, "unit": "m/s", "terminal_velocity": vt, "tau": m / b}


def drag_traj(p, steps=50):
    m = float(p["m"]); b = float(p["b"])
    vt = m * G / b
    tau = m / b
    t = np.linspace(0.0, 4.0 * tau, steps)
    x = vt * (t - tau * (1.0 - np.exp(-t / tau)))
    v = vt * (1.0 - np.exp(-t / tau))
    return np.stack([x, v], axis=-1)


def drag_verify(p, steps=4000):
    """Euler integration of dv/dt = g - (b/m) v; compare with terminal velocity."""
    m = float(p["m"]); b = float(p["b"])
    vt = m * G / b
    tau = m / b
    # integrate to 8 tau: Euler's fixed point is exactly v_t, and 8 tau leaves
    # a finite-horizon error of e^-8 ~ 3e-4
    dt = (8.0 * tau) / steps
    v = 0.0
    for _ in range(steps):
        v += (G - (b / m) * v) * dt
    rel = abs(v - vt) / vt
    return {"residual": rel, "ok": rel < 2e-3, "numeric_terminal": v}


# ---------------------------------------------------------------- dispatch

CLOSED = {
    "projectile": projectile_closed, "pendulum": pendulum_closed,
    "spring": spring_closed, "beam": beam_closed,
    "cantilever": cantilever_closed, "burgers": burgers_closed, "rc": rc_closed,
    "heat2d": heat2d_closed,
    "damped": damped_closed, "kepler": kepler_closed,
    "lc": lc_closed, "drag": drag_closed,
}
TRAJ = {
    "projectile": projectile_traj, "pendulum": pendulum_traj,
    "spring": spring_traj, "beam": beam_traj,
    "cantilever": cantilever_traj, "burgers": burgers_traj, "rc": rc_traj,
    "heat2d": heat2d_traj,
    "damped": damped_traj, "kepler": kepler_traj,
    "lc": lc_traj, "drag": drag_traj,
}
VERIFY = {
    "projectile": projectile_verify, "pendulum": pendulum_verify,
    "spring": spring_verify, "beam": beam_verify,
    "cantilever": cantilever_verify, "burgers": burgers_verify, "rc": rc_verify,
    "heat2d": heat2d_verify,
    "damped": damped_verify, "kepler": kepler_verify,
    "lc": lc_verify, "drag": drag_verify,
}


def closed(domain, params):
    return CLOSED[domain](params)


def trajectory(domain, params, steps=50):
    return TRAJ[domain](params, steps)


def verify(domain, params):
    return VERIFY[domain](params)

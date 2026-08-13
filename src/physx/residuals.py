"""residuals.py — the physics-consistency layer used during training.

For each domain we know a governing equation that holds *everywhere* along the
trajectory. The physics layer of PhysFormer computes how far a predicted
trajectory strays from that equation, so the model is trained against BOTH
data (closed-form trajectories) AND physics (ODE / conservation residuals).
This is the PINN-style "physics-informed" idea: the network learns dynamics,
and the physics layer keeps it honest.

All residuals are differentiable in torch; params values are (B,) tensors,
trajectories are (B, T, 2).
"""

import torch

G = 9.81


def residual(domain, traj, params):
    """traj: (B, T, 2) predicted trajectory. params: dict of (B,) tensors.
    Returns (B,) non-negative residual; 0 = perfectly physical."""
    if domain == "projectile":
        return _projectile(traj, params)
    if domain == "pendulum":
        return _pendulum(traj, params)
    if domain == "spring":
        return _spring(traj, params)
    if domain == "beam":
        return _beam(traj, params)
    if domain == "cantilever":
        return _cantilever(traj, params)
    if domain == "burgers":
        return _burgers(traj, params)
    if domain == "rc":
        return _rc(traj, params)
    if domain == "heat2d":
        return _heat2d(traj, params)
    if domain == "damped":
        return _damped(traj, params)
    if domain == "kepler":
        return _kepler(traj, params)
    if domain == "lc":
        return _lc(traj, params)
    if domain == "drag":
        return _drag(traj, params)
    raise ValueError(domain)


def _projectile(traj, params):
    # y(x) must satisfy y = x tan(a) - g x^2 / (2 v0^2 cos^2(a))
    v0 = params["v0"].unsqueeze(-1)
    a = torch.deg2rad(params["angle"]).unsqueeze(-1)
    x, y = traj[..., 0], traj[..., 1]
    x = x - x[:, :1]  # start at origin
    expect = x * torch.tan(a) - G * x ** 2 / (2 * v0 ** 2 * torch.cos(a) ** 2)
    scale = torch.abs(expect).max(dim=-1).values + 1e-6
    return ((y - expect) ** 2).mean(dim=-1) / scale


def _pendulum(traj, params):
    # energy conservation: 0.5 L^2 w^2 + g L (1 - cos th) = const
    L = params["L"].unsqueeze(-1)
    th, w = traj[..., 0], traj[..., 1]
    E = 0.5 * L ** 2 * w ** 2 + G * L * (1 - torch.cos(th))
    scale = E[:, :1].abs() + 1e-6
    return (((E - E[:, :1]) / scale) ** 2).mean(dim=-1)


def _spring(traj, params):
    # m x'' + k x = 0
    k = params["k"].unsqueeze(-1)
    m = params["m"].unsqueeze(-1)
    x, _ = traj[..., 0], traj[..., 1]
    n = x.shape[-1]
    w = torch.sqrt(k / m)
    dt = (2 * (2 * torch.pi) / w) / (n - 1)
    xdd = (x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]) / dt ** 2
    res = m * xdd + k * x[..., 1:-1]
    scale = k * x[..., 1:-1].abs().max(dim=-1).values + 1e-6
    return ((res / scale.unsqueeze(-1)) ** 2).mean(dim=-1)


def _beam(traj, params):
    # moment-curvature relation: EI * w'' = M(x)  (holds everywhere, unlike
    # the 4th-order ODE which has a Dirac source at a point load).
    # The x grid is fully determined by L, so we use the TRUE grid from the
    # params — gradients only flow through the predicted deflection w, never
    # through a learned coordinate.
    E = params["E"].unsqueeze(-1)
    I = params["I"].unsqueeze(-1)
    P = params["P"].unsqueeze(-1)
    L = params["L"].unsqueeze(-1)
    w = traj[..., 0]
    n = w.shape[-1]
    x = torch.linspace(0.0, 1.0, n, device=traj.device) * L  # (B, n) true grid
    dx = L / (n - 1)
    wpp = (w[..., 2:] - 2 * w[..., 1:-1] + w[..., :-2]) / dx ** 2
    xc = x[..., 1:-1]
    half = L / 2
    M = torch.where(xc <= half, (P / 2) * xc, (P / 2) * (L - xc))
    # w is positive-downward in our convention, so EI w'' = -M(x)
    expect = -M / (E * I)
    scale = expect.abs().max(dim=-1).values + 1e-12
    return (((wpp - expect) / scale.unsqueeze(-1)) ** 2).mean(dim=-1)


def _cantilever(traj, params):
    # moment-curvature relation for a cantilever with tip load P:
    #   EI * w'' = -M(x),  M(x) = -P(L-x)   ->   w'' = P(L-x)/(EI)
    # Holds everywhere along the span (no Dirac source); the x grid is fully
    # determined by L, so gradients flow only through the predicted w.
    E = params["E"].unsqueeze(-1)
    I = params["I"].unsqueeze(-1)
    P = params["P"].unsqueeze(-1)
    L = params["L"].unsqueeze(-1)
    w = traj[..., 0]
    n = w.shape[-1]
    x = torch.linspace(0.0, 1.0, n, device=traj.device) * L  # true grid
    dx = L / (n - 1)
    wpp = (w[..., 2:] - 2 * w[..., 1:-1] + w[..., :-2]) / dx ** 2
    xc = x[..., 1:-1]
    expect = P * (L - xc) / (E * I)
    scale = expect.abs().max(dim=-1).values + 1e-12
    return (((wpp - expect) / scale.unsqueeze(-1)) ** 2).mean(dim=-1)


def _burgers(traj, params):
    # Viscous Burgers: u_t + u u_x = nu u_xx on the true (x, t) grid. The
    # predicted trajectory is the flattened field (B, NX*NT, 1); gradients
    # flow only through u, never through the grid (fixed domain constants).
    from . import sim
    nu = params["nu"].unsqueeze(-1).unsqueeze(-1)
    u = traj[..., 0].view(traj.shape[0], sim.NT, sim.NX)
    dx = (sim.XR - sim.XL) / (sim.NX - 1)
    dt = sim.TF / (sim.NT - 1)
    # interior time (forward difference) and space (central differences)
    ut = (u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / dt
    ux = (u[:, 1:, 2:] - u[:, 1:, :-2]) / (2 * dx)
    uxx = (u[:, 1:, 2:] - 2 * u[:, 1:, 1:-1] + u[:, 1:, :-2]) / dx ** 2
    uu = u[:, 1:, 1:-1]
    res = ut + uu * ux - nu * uxx
    # per-sample scale: the magnitude of the dominant terms
    scale = (torch.abs(uu * ux) + torch.abs(nu * uxx) + 1e-8).max(dim=-1).values
    return ((res / scale.unsqueeze(-1)) ** 2).mean(dim=(-1, -2))


def _heat2d(traj, params):
    # Poisson: u_xx + u_yy = f on the true N x N grid, interior points only.
    # The predicted trajectory is the flattened field (B, N*N, 1); gradients
    # flow only through u, never through the grid (fixed domain geometry).
    # A fourth-order Laplacian stencil keeps the truncation floor far below
    # the residuals seen during training (the 5-point stencil's floor is a
    # few percent of the source magnitude at the highest mode numbers).
    from . import sim
    A = params["A"].unsqueeze(-1).unsqueeze(-1)
    k = params["k"].unsqueeze(-1).unsqueeze(-1)
    l = params["l"].unsqueeze(-1).unsqueeze(-1)
    u = traj[..., 0].view(traj.shape[0], sim.H2D_N, sim.H2D_N)
    dx = 1.0 / (sim.H2D_N - 1)
    n = sim.H2D_N
    # interior (i, j in 2..N-3): 4th-order central second differences
    def d2(v, dim):
        if dim == 0:
            a, b, c, d, e = v[:, :-4, 2:-2], v[:, 1:-3, 2:-2], v[:, 2:-2, 2:-2], v[:, 3:-1, 2:-2], v[:, 4:, 2:-2]
        else:
            a, b, c, d, e = v[:, 2:-2, :-4], v[:, 2:-2, 1:-3], v[:, 2:-2, 2:-2], v[:, 2:-2, 3:-1], v[:, 2:-2, 4:]
        return (-a + 16 * b - 30 * c + 16 * d - e) / (12 * dx ** 2)
    lap = d2(u, 0) + d2(u, 1)
    # source term on the same interior grid (fixed x, y coordinates)
    xi = torch.linspace(0.0, 1.0, n, device=traj.device)[2:-2]
    yj = torch.linspace(0.0, 1.0, n, device=traj.device)[2:-2]
    X, Y = torch.meshgrid(xi, yj, indexing="ij")
    f = -(A * torch.pi ** 2 * (k ** 2 + l ** 2) / 2.0) * torch.sin(k * torch.pi * X) * torch.sin(l * torch.pi * Y)
    res = lap - f
    scale = f.abs().max(dim=-1).values.unsqueeze(-1) + 1e-8
    return ((res / scale) ** 2).mean(dim=(-1, -2))


def _damped(traj, params):
    # m x'' + c x' + k x = 0. The t grid is 0..4 pi / wd, fully determined by
    # (k, m, c); gradients flow only through the predicted x.
    k = params["k"].unsqueeze(-1)
    m = params["m"].unsqueeze(-1)
    c = params["c"].unsqueeze(-1)
    x, _ = traj[..., 0], traj[..., 1]
    n = x.shape[-1]
    wn = torch.sqrt(k / m)
    zeta = c / (2.0 * torch.sqrt(k * m))
    wd = wn * torch.sqrt(torch.clamp(1.0 - zeta ** 2, min=1e-8))
    dt = (4.0 * torch.pi / wd) / (n - 1)
    xdd = (x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]) / dt ** 2
    xd = (x[..., 2:] - x[..., :-2]) / (2 * dt)
    res = m * xdd + c * xd + k * x[..., 1:-1]
    scale = (k * x[..., 1:-1].abs().max(dim=-1).values + 1e-6)
    return ((res / scale.unsqueeze(-1)) ** 2).mean(dim=-1)


def _kepler(traj, params):
    # vis-viva (specific orbital energy): v^2 = GM (2/r - 1/a), evaluated from
    # the predicted (x, y) on the TRUE time grid determined by (a, M).
    from . import sim
    a = params["a"].unsqueeze(-1)
    M = params["M"].unsqueeze(-1)
    x, y = traj[..., 0], traj[..., 1]
    n = x.shape[-1]
    T = 2.0 * torch.pi * torch.sqrt(a ** 3 / (sim.G_KEP * M))
    t = torch.linspace(0.0, 1.0, n, device=traj.device).unsqueeze(0) * T
    dt = T / (n - 1)
    vx = (x[..., 2:] - x[..., :-2]) / (2 * dt)
    vy = (y[..., 2:] - y[..., :-2]) / (2 * dt)
    r = torch.sqrt(x[..., 1:-1] ** 2 + y[..., 1:-1] ** 2)
    v2 = vx ** 2 + vy ** 2
    expect = sim.G_KEP * M * (2.0 / r - 1.0 / a)
    scale = expect.abs().max(dim=-1).values + 1e-12
    return (((v2 - expect) / scale.unsqueeze(-1)) ** 2).mean(dim=-1)


def _lc(traj, params):
    # L q'' + q/C = 0 (the electrical harmonic oscillator). Computed in
    # dimensionless form q~'' + q~ = 0 with q~ = q/(C V0) and tau = w t: raw
    # charge values are ~1e-5 C and raw second differences divide by dt^2 ~
    # 1e-17, so float32 catastrophic cancellation would swamp the signal.
    L = params["L"].unsqueeze(-1)
    C = params["C"].unsqueeze(-1)
    V0 = params["V0"].unsqueeze(-1)
    q, _ = traj[..., 0], traj[..., 1]
    n = q.shape[-1]
    w = 1.0 / torch.sqrt(L * C)
    q0 = C * V0
    qn = q / q0
    dtau = (4.0 * torch.pi) / (n - 1)
    qdd = (qn[..., 2:] - 2 * qn[..., 1:-1] + qn[..., :-2]) / dtau ** 2
    res = qdd + qn[..., 1:-1]
    return (res ** 2).mean(dim=-1)


def _drag(traj, params):
    # m v' = mg - b v. The t grid is 0..4 tau, determined by (m, b).
    m = params["m"].unsqueeze(-1)
    b = params["b"].unsqueeze(-1)
    _, v = traj[..., 0], traj[..., 1]
    n = v.shape[-1]
    tau = m / b
    dt = (4.0 * tau) / (n - 1)
    dvdt = (v[..., 2:] - v[..., :-2]) / (2 * dt)
    res = dvdt + (b / m) * v[..., 1:-1] - G
    scale = G + 1e-6
    return ((res / scale) ** 2).mean(dim=-1)


def _rc(traj, params):
    # dV/dt + V/tau = V0/tau. The t grid is 0..4*tau, fully determined by
    # R and C — use the true grid so gradients flow only through V.
    R = params["R"].unsqueeze(-1)
    C = params["C"].unsqueeze(-1)
    V0 = params["V0"].unsqueeze(-1)
    V = traj[..., 0]
    n = V.shape[-1]
    tau = R * C
    t = torch.linspace(0.0, 4.0, n, device=traj.device) * tau  # (B, n) true grid
    dt = (4 * tau) / (n - 1)
    dVdt = (V[..., 2:] - V[..., :-2]) / (2 * dt.unsqueeze(-1))
    res = dVdt + V[..., 1:-1] / tau - V0 / tau
    scale = V0 / tau
    return ((res / scale) ** 2).mean(dim=-1)

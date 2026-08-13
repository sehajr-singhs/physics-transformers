"""physformer.py — PhysFormer: a transformer adjusted for physics.

Architecture (reasoning layers + physics layers, together):

  input tokens : (param_id, value) pairs  ->  learned param embedding
                                             + continuous value projection
                                             + positional encoding
  reasoning    : stack of standard transformer encoder layers
                 (multi-head self-attention + FFN + layer norm) — learns
                 patterns in the problem parameters
  physics      : a projection layer maps the hidden state to a trajectory
                 AND a scalar answer; the trajectory is refined by a learned
                 physics gain; a *physics-consistency layer* then computes
                 how far the trajectory strays from the governing equation
                 (residuals.py) and that residual is added to the loss.

The model is therefore trained on BOTH the data (closed-form trajectories,
exact answers) and physics (ODE / conservation residuals) — a PINN-style
physics-informed transformer.
"""

import math

import torch
import torch.nn as nn

from . import residuals


class _MLPReasoning(nn.Module):
    """Baseline encoder with the same capacity budget as the transformer's
    reasoning stack, but no attention: a stack of MLP blocks over the flattened
    token sequence. Identical heads, so the only difference vs PhysFormer is
    whether the parameters interact through attention."""

    def __init__(self, n_params, d_model, hidden, n_blocks=3, dropout=0.05):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(
            nn.Linear(n_params * d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            *layers,
            nn.Linear(hidden, d_model),
        )

    def forward(self, x, mask=None):
        b, p, d = x.shape
        return self.net(x.reshape(b, p * d))


class PhysFormer(nn.Module):
    def __init__(
        self,
        domain,
        n_params=6,
        d_model=48,
        nhead=4,
        n_layers=3,
        dim_ff=96,
        traj_steps=50,
        traj_dim=2,
        traj_hidden=None,
        dropout=0.05,
        kind="physformer",
        sigmoid_traj=False,
    ):
        super().__init__()
        self.domain = domain
        self.n_params = n_params
        self.traj_steps = traj_steps
        self.traj_dim = traj_dim
        self.d_model = d_model
        self.kind = kind
        # shape-normalized targets live in [0, 1] by construction (deflection /
        # peak, voltage / V0, Burgers field / A): constraining the head output
        # to (0, 1) is physically principled and prevents unphysical negatives
        self.sigmoid_traj = sigmoid_traj

        self.param_emb = nn.Embedding(n_params, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        if kind == "mlp":
            # data-only baseline: same capacity, no attention, no physics term
            self.reasoning = _MLPReasoning(n_params, d_model, dim_ff, n_blocks=n_layers)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            # reasoning layers
            self.reasoning = nn.TransformerEncoder(layer, num_layers=n_layers)

        # physics layer: trajectory projector. A single hidden layer (width
        # = traj_hidden, default d_model) is used: deeper or wider heads
        # destabilized training at our data scale (256--512 samples) — a
        # documented negative result, see manuscript Sec. "Ablations".
        if traj_hidden is None:
            traj_hidden = d_model
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, traj_hidden),
            nn.GELU(),
            nn.Linear(traj_hidden, traj_steps * traj_dim),
        )

        # answer head (regresses the engineering answer scalar)
        self.ans_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, pids, vals, mask=None):
        """pids: (B, P) param ids; vals: (B, P) normalized values."""
        b, p = pids.shape
        x = self.param_emb(pids) + self.value_proj(vals.unsqueeze(-1))
        x = x + self.pos[:, :p, :]
        h = self.reasoning(x, mask=mask)
        if h.dim() == 3:
            h = h[:, -1]
        ans = self.ans_head(h).squeeze(-1)
        traj = self.traj_head(h).view(b, self.traj_steps, self.traj_dim)
        if self.sigmoid_traj:
            traj = torch.sigmoid(traj)
        return ans, traj

    def physics_residual(self, traj, params):
        """The physics-consistency layer: governing-equation residual (B,)."""
        return residuals.residual(self.domain, traj, params)


class AdamW(torch.optim.AdamW):
    pass


def build(domain, st, **kw):
    n_params = len(st["keys"])
    # beam/cantilever/rc trajectories are the single dependent column on a
    # true grid (coordinate columns are deterministic and never predicted)
    kw.setdefault("traj_dim", 2 if domain in ("projectile", "pendulum", "spring") else 1)
    return PhysFormer(domain, n_params=n_params, **kw)


# ---------------------------------------------------------------------------
# PhysFormerLCA — Law-Conditioned Attention (the multi-law generalist)
#
# The novel mechanism: the *governing equation itself* is part of the input.
# A fixed operator vocabulary (laws.LAW_VOCAB) expresses each domain's law as
# a sparse binary signature; an MLP embeds that signature into a law vector,
# and every transformer layer cross-attends to it. Attention is therefore
# conditioned on the differential structure of the physics in force — a new
# way to feed physics into a transformer that is orthogonal to (and combined
# with) the physics-consistency loss.
#
# The multi-law model shares BOTH the body and the output heads across all
# domains: for a fixed trajectory geometry (50 steps, <= 2 channels) one
# answer head and one trajectory head serve every law. Per-sample information
# about which law applies comes only from the law signature, so the model
# must use the equation itself to interpret the (per-domain normalized)
# parameters. The physics residual is still computed per sample from the
# governing equation of that sample's domain.
# ---------------------------------------------------------------------------


class _LawCrossAttention(nn.Module):
    """Cross-attention from the token sequence (queries) to the law vector
    (keys/values, replicated over the token axis). Residual + layer norm."""

    def __init__(self, d_model, nhead, dropout=0.05):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, law_kv):
        out, _ = self.attn(h, law_kv, law_kv, need_weights=False)
        return self.norm(h + out)


class PhysFormerLCA(nn.Module):
    """Multi-law transformer with Law-Conditioned Attention.

    domains: list of domain names sharing the trajectory geometry.
    law_vocab: size of the operator vocabulary (laws.LAW_VOCAB).
    law_mode: "real" uses the per-sample law signature (the invention);
              "dummy" feeds a constant signature so the conditioning stream
              exists but carries no equation information (the ablation);
              "none" omits the conditioning stream entirely.
    """

    def __init__(
        self,
        domains,
        law_vocab,
        law_mode="real",
        n_params=6,
        d_model=48,
        nhead=4,
        n_layers=3,
        dim_ff=96,
        traj_steps=50,
        traj_dim=2,
        traj_hidden=None,
        dropout=0.05,
        sigmoid_traj=False,
    ):
        super().__init__()
        self.domains = domains
        self.law_vocab = law_vocab
        self.law_mode = law_mode
        self.n_params = n_params
        self.traj_steps = traj_steps
        self.traj_dim = traj_dim
        self.d_model = d_model
        self.sigmoid_traj = sigmoid_traj
        # one shared parameter vocabulary across domains: each domain's params
        # occupy a contiguous offset block (pids are domain_offset + index)
        self.param_emb = nn.Embedding(n_params, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.layers = self.encoder.layers

        if law_mode != "none":
            self.law_mlp = nn.Sequential(
                nn.Linear(law_vocab, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            )
            self.lca = nn.ModuleList([
                _LawCrossAttention(d_model, nhead, dropout) for _ in range(n_layers)
            ])
            # law-gated readout: the law vector also sets the scale/bias of the
            # shared heads, so the model can invert per-domain output
            # normalizations (log10 vs raw answers) from the equation itself
            self.head_gate = nn.Linear(d_model, 2 * d_model)
        else:
            self.law_mlp = None
            self.lca = None
            self.head_gate = None

        if traj_hidden is None:
            traj_hidden = d_model
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, traj_hidden),
            nn.GELU(),
            nn.Linear(traj_hidden, traj_steps * traj_dim),
        )
        self.ans_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def _law_vec(self, law_sig, b, device):
        """(B, d_model) law vector (real signature or constant for dummy)."""
        if self.law_mode == "real":
            return self.law_mlp(law_sig)
        return self.law_mlp(torch.ones(b, self.law_vocab, device=device))

    def forward(self, pids, vals, law_sig=None):
        """pids: (B, P) unified param ids; vals: (B, P) per-domain normalized.
        law_sig: (B, law_vocab) binary signature (ignored for dummy/none)."""
        b, p = pids.shape
        x = self.param_emb(pids) + self.value_proj(vals.unsqueeze(-1))
        x = x + self.pos[:, :p, :]
        h = x
        law = None
        if self.lca is not None:
            law = self._law_vec(law_sig, b, x.device)
            law_kv = law.unsqueeze(1).expand(b, p, -1)
            for layer, lca in zip(self.layers, self.lca):
                h = layer(h)
                h = lca(h, law_kv)
        else:
            h = self.encoder(h)
        h = h[:, -1]
        if self.head_gate is not None:
            gamma, beta = self.head_gate(law).chunk(2, dim=-1)
            h = gamma * h + beta
        ans = self.ans_head(h).squeeze(-1)
        traj = self.traj_head(h).view(b, self.traj_steps, self.traj_dim)
        if self.sigmoid_traj:
            traj = torch.sigmoid(traj)
        return ans, traj

    def physics_residual(self, domain, traj, params):
        """Governing-equation residual for one domain: (B,)."""
        return residuals.residual(domain, traj, params)

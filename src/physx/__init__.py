"""physx — a physics-AI core for AGE (Artificial General Engineer).

Deterministic simulators with closed-form ground truth, a physics dataset
generator, a physics-adjusted transformer (reasoning layers + physics layers),
and an engineering solve CLI.

Domains: projectile, pendulum, spring, beam, rc.
"""

__all__ = ["DOMAINS"]

DOMAINS = ("projectile", "pendulum", "spring", "beam", "rc")

# Physics Transformers

> **The equation is fed to the transformer as input tokens, not only as a
> penalty — and a pre-registered theory of *when* that should pay was tested
> on ten laws and falsified. The failure is the paper's sharpest result.**

A transformer adjusted for physics — **PhysFormer** — and a new way to feed
physics into it: *Law-Conditioned Attention* (LCA). The governing equation is
tokenized into a fixed symbolic vocabulary, embedded into a law vector, and
injected as a cross-attention key/value stream in every layer. Physics enters
through two separate channels with two different jobs:

- **The loss channel** — a differentiable physics-consistency layer that
  penalizes violation of the governing equation. It buys *consistency*
  (6–8× reduction in governing-equation residual) but not held-out accuracy.
- **The input channel** — the invention of this paper. The equation signature
  is part of the *input*, so the model can use it at inference, not just
  during training.

**Paper:** [manuscript.pdf](manuscript.pdf) · [supplementary
information](supplementary_information.pdf) (Nature Machine Intelligence
format). Every number in the paper reads from a committed JSON under
`results/` and `figs/` — never hand-typed.

## What survived contact with the mathematics

Three results, each attacked before being written down:

1. **The equation is causally active, not decoratively present.** Swapping
   the equation signature at inference steers the prediction (p < 0.0001)
   while a constant-signature control is exactly insensitive. Across 36
   paired runs, law-conditioned attention cuts trajectory error by 21%
   (p = 0.0003), and its effect concentrates on the one pair of laws whose
   parameter tokens are *literally identical* (beam/cantilever).
2. **The pre-registered regime theory failed — and that is the result.**
   Before any ten-law training, we filed a prediction: LCA benefit should be
   monotone in the token-vocabulary ambiguity of each law, computed from the
   vocabulary alone. Measured after 6 full trainings (3 seeds ×
   generalist/control): Spearman ρ = 0.07 (p = 0.88), leave-one-out ρ =
   −0.58. The failure analysis shows why — the overlap measure conflates
   token-superset relations with genuine indistinguishability — and leaves a
   falsifiable open problem instead of a swept-under null. The
   pre-registration is committed at `results/pre_registration.json`; the
   eval files that falsified it are in the same directory.
3. **External baseline: DeepONet.** Ten dedicated per-law DeepONets on the
   identical data splits reach median held-out trajectory error 0.037 vs.
   the single generalist's 0.110 — but they are ten separate models with no
   cross-law structure. The generalist matches or beats per-law specialists
   outright on spring and LC while serving all ten laws with one set of
   weights and no law identity at inference.

## Measured results (nothing extrapolated)

| Claim | Number | Where |
|---|---|---|
| LCA reduces trajectory error (36 paired runs) | 21%, p = 0.0003 | §law-conditioned attention |
| Equation swap steers prediction at inference | p < 0.0001; control exactly insensitive | §causal test |
| Few-shot: generalist adapts to held-out law | 2.9× lower error than specialist at 25% data | §adaptation |
| Pre-registered regime prediction | **falsified**: ρ = 0.07, p = 0.88; LOO ρ = −0.58 | §regime test |
| Loss channel: residual reduction | 6–8× (consistency) with no accuracy gain | §channels |
| DeepONet per-law vs single generalist | 0.037 vs 0.110 median (9 non-degenerate laws) | §baselines |
| 2D field (heat plate) | 5.9% canonical / ~29% honest held-out peak error | §fields |

## Repo layout

```
manuscript.tex / .pdf     the paper (NMI format)
supplementary_information.tex / .pdf
figs/                     figures + the JSONs they read + make scripts
results/                  pre_registration.json + 6 trained eval files
src/physx/                the full AGE physics core (train_multi, laws,
                          sim, residuals, physformer, baselines, regime_oos)
tests/                    unit tests (49 physics tests incl. the 10-law set)
```

## Reproduce

```bash
pip install -r requirements.txt
python -m unittest tests.test_physx     # 49 tests: closed forms, verifiers, 10-law set

# regenerate every figure from committed JSONs
python figs/make_figures.py
python figs/make_figures_ext.py

# re-run the out-of-sample regime analysis (vectorized permutation test)
python src/physx/regime_oos.py --out figs/regime_oos.json

# re-run a full multi-law training (real + dummy control, 3 seeds)
python src/physx/train_multi.py --ext --seeds 3

# DeepONet external baseline (per-law)
python src/physx/baselines.py --per-law-only
```

## Honest gaps

The pooled single-model DeepONet (law identity as an explicit one-hot branch
input — information the generalist never receives) did not complete under the
CPU contention available during this project; the per-law comparison is
reported as the baseline rather than claiming an unmeasured result. The
refined regime hypothesis (conditioning pays when tokens alone cannot
identify the law) remains an open problem: pendulum shows consistent benefit
with no token twin, and the learned embeddings do not confound its tokens
(max cosine 0.30).

## License

MIT — see [LICENSE](LICENSE).

"""Ground-truth assay simulator: a low-input NGS library-prep protocol.

This module is the "wet lab". The optimization agent and the world model are
never allowed to look inside it; they only get to submit a protocol (4 knobs)
and receive noisy measured readouts, exactly like a real bench run.

The response surfaces are physically motivated, not arbitrary:
  - more input DNA and more PCR cycles raise yield, with saturation
  - over-cycling relative to the input mass destroys library complexity
    (duplication), so the complexity optimum is an interior ridge, not a corner
  - high primer concentration and low input drive adapter-dimer formation
  - SPRI bead ratio trades library retention against dimer carryover

Measurement noise is heteroscedastic: low-input runs are noisier. That is the
whole reason the downstream QC gate needs calibrated uncertainty rather than a
single point prediction.

Units are ASCII throughout (nM, ng, uM, x SPRI).
"""

from __future__ import annotations

import numpy as np

# Design space (the protocol knobs the agent controls), in real units.
# name -> (low, high)
BOUNDS = {
    "input_dna_ng": (0.1, 10.0),   # precious single-cell / clinical material
    "pcr_cycles": (6.0, 18.0),     # amplification cycles
    "bead_ratio": (0.6, 1.2),      # x SPRI cleanup ratio
    "primer_conc_uM": (0.2, 2.0),  # adapter/primer concentration
}
PARAM_NAMES = list(BOUNDS.keys())

# Objectives. All are framed as "maximize". direction/threshold encode the
# release acceptance criteria used by the QC gate.
#   yield_nM   : usable library molarity, maximize, must clear >= 4 nM
#   complexity : unique-molecule fraction, maximize, must clear >= 0.60
#   purity     : 1 - adapter_dimer_fraction, maximize, must clear >= 0.90
OBJECTIVES = ("yield_nM", "complexity", "purity")
ACCEPTANCE = {
    "yield_nM": 15.0,
    "complexity": 0.75,
    "purity": 0.94,
}
# For hypervolume and scalarization we normalize each objective to ~[0, 1].
OBJ_SCALE = {
    "yield_nM": 50.0,  # yields above ~50 nM are all "plenty"
    "complexity": 1.0,
    "purity": 1.0,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _to_unit(X: np.ndarray) -> np.ndarray:
    """Map real-unit params to [0, 1] per dimension. input_dna is log-scaled."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lo = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
    hi = np.array([BOUNDS[p][1] for p in PARAM_NAMES])
    u = (X - lo) / (hi - lo)
    # input DNA acts on a log scale biologically; remap that axis.
    log_lo, log_hi = np.log10(lo[0]), np.log10(hi[0])
    u[:, 0] = (np.log10(np.clip(X[:, 0], lo[0], hi[0])) - log_lo) / (log_hi - log_lo)
    return np.clip(u, 0.0, 1.0)


def latent_outcomes(X: np.ndarray) -> dict[str, np.ndarray]:
    """Noise-free ground-truth outcomes (the true response surface)."""
    u = _to_unit(X)
    u_in, u_cyc, u_bead, u_prim = u[:, 0], u[:, 1], u[:, 2], u[:, 3]

    # SPRI retention: tight ratios lose short library; loose ratios keep dimers.
    library_keep = _sigmoid(6.0 * (u_bead - 0.15))     # keep desired library
    dimer_keep = _sigmoid(7.0 * (u_bead - 0.55))       # carry over dimers

    # Yield: driven by input and cycles, saturating, scaled by library retention.
    amp = 1.0 - np.exp(-3.0 * u_cyc)
    yield_nM = 50.0 * (0.12 + 0.88 * u_in) * amp * library_keep
    yield_nM = np.clip(yield_nM, 0.0, None)

    # Complexity: capacity set by input; destroyed by over-cycling past what the
    # input mass supports (duplication). Interior ridge along u_cyc ~ f(u_in).
    capacity = 0.15 + 0.85 * u_in
    over_cycle = _sigmoid(6.0 * (u_cyc - u_in - 0.10))  # duplication fraction
    complexity = capacity * (1.0 - 0.9 * over_cycle) * library_keep
    complexity = np.clip(complexity, 0.0, 1.0)

    # Adapter dimers: worse with high primer, low input, and dimer carryover.
    primer_factor = 0.15 + 0.85 * u_prim
    lowinput_factor = 1.0 + 1.6 * (1.0 - u_in)
    dimer = 0.02 + 0.55 * primer_factor * (lowinput_factor / 2.6) * dimer_keep
    dimer = np.clip(dimer, 0.0, 0.95)
    purity = 1.0 - dimer

    return {"yield_nM": yield_nM, "complexity": complexity, "purity": purity}


def _noise_scale(u_in: np.ndarray) -> np.ndarray:
    """Heteroscedastic noise multiplier: low input is much noisier."""
    return 0.03 + 0.14 * (1.0 - u_in)


def run_assay(X: np.ndarray, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Submit protocol(s), get noisy measured readouts (one simulated bench run)."""
    if rng is None:
        rng = np.random.default_rng()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    u_in = _to_unit(X)[:, 0]
    truth = latent_outcomes(X)
    s = _noise_scale(u_in)

    yield_obs = truth["yield_nM"] * (1.0 + rng.normal(0.0, s))
    complexity_obs = truth["complexity"] + rng.normal(0.0, 0.6 * s)
    purity_obs = truth["purity"] + rng.normal(0.0, 0.6 * s)

    return {
        "yield_nM": np.clip(yield_obs, 0.0, None),
        "complexity": np.clip(complexity_obs, 0.0, 1.0),
        "purity": np.clip(purity_obs, 0.0, 1.0),
    }


def passes_acceptance(outcomes: dict[str, np.ndarray]) -> np.ndarray:
    """Release gate: does a run clear every acceptance criterion?"""
    ok = np.ones(np.shape(next(iter(outcomes.values()))), dtype=bool)
    for name, thresh in ACCEPTANCE.items():
        ok &= np.asarray(outcomes[name]) >= thresh
    return ok


def sample_design(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random protocols in the real-unit design box."""
    lo = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
    hi = np.array([BOUNDS[p][1] for p in PARAM_NAMES])
    return lo + (hi - lo) * rng.random((n, len(PARAM_NAMES)))

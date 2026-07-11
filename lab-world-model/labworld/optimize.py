"""Closed-loop optimization: propose -> run -> update -> repeat.

The agent runs multi-objective Bayesian optimization (ParEGO). Each round it:
  1. draws a random preference weight over the objectives,
  2. scalarizes the observed data with an augmented Tchebycheff utility,
  3. fits a GP to that scalar utility (the world model for this round),
  4. picks the next protocol by maximizing Expected Improvement,
  5. submits it to the assay and folds the result back in.

Random search is included as the honest baseline: it is what a scientist does by
gridding or guessing without a model. The point of the demo is the gap between
the two in experiments-to-result, which is hands-on-time and reagent burn.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from . import assay
from .world_model import normalize_X

_LO = np.array([assay.BOUNDS[p][0] for p in assay.PARAM_NAMES])
_HI = np.array([assay.BOUNDS[p][1] for p in assay.PARAM_NAMES])

# Fixed Monte-Carlo pool for a low-variance hypervolume estimate, reused across
# every call so curves are comparable.
_HV_RNG = np.random.default_rng(20260710)
_HV_POOL = _HV_RNG.random((40000, len(assay.OBJECTIVES)))


def objectives_to_unit(outcomes: dict[str, np.ndarray]) -> np.ndarray:
    """Stack objectives into a [n, 3] array normalized to ~[0, 1], all maximize."""
    cols = []
    for name in assay.OBJECTIVES:
        cols.append(np.asarray(outcomes[name]) / assay.OBJ_SCALE[name])
    return np.clip(np.stack(cols, axis=1), 0.0, 1.0)


def hypervolume(points_unit: np.ndarray) -> float:
    """Monte-Carlo dominated hypervolume over [0,0,0] .. [1,1,1] (maximize)."""
    if points_unit.size == 0:
        return 0.0
    P = np.atleast_2d(points_unit)
    # A sample is dominated if some point beats it on all objectives.
    dominated = np.zeros(_HV_POOL.shape[0], dtype=bool)
    for p in P:
        dominated |= np.all(_HV_POOL <= p, axis=1)
    return float(dominated.mean())


def _tchebycheff(Yu: np.ndarray, w: np.ndarray, rho: float = 0.05) -> np.ndarray:
    """Augmented Tchebycheff utility for maximization (higher is better)."""
    wg = w * Yu
    return wg.min(axis=1) + rho * wg.sum(axis=1)


def _fit_scalar_gp(Xn: np.ndarray, s: np.ndarray) -> GaussianProcessRegressor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(len(assay.PARAM_NAMES)), nu=2.5)
        + WhiteKernel(0.05, (1e-4, 1.0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=1, alpha=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(Xn, s)
    return gp


def _expected_improvement(gp, Xc_n: np.ndarray, best: float, xi: float = 0.01):
    mu, sd = gp.predict(Xc_n, return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


@dataclass
class RunResult:
    X: np.ndarray                       # protocols tried, real units
    outcomes: dict                      # measured objective arrays
    hv_curve: np.ndarray                # true-objective hypervolume vs experiment
    first_pass: int                     # experiment index of first QC pass (or -1)
    label: str = ""
    extra: dict = field(default_factory=dict)


def _hv_curve_and_pass(X: np.ndarray, outcomes: dict) -> tuple[np.ndarray, int]:
    # Score on the *true* surface so noise does not flatter the curve.
    truth = assay.latent_outcomes(X)
    Yu = objectives_to_unit(truth)
    passed = assay.passes_acceptance(outcomes)  # judged on the measured run
    hv, first = [], -1
    for i in range(len(X)):
        hv.append(hypervolume(Yu[: i + 1]))
        if first < 0 and passed[i]:
            first = i + 1
    return np.array(hv), first


def random_search(n_iter: int, seed: int, n_init: int = 6) -> RunResult:
    rng = np.random.default_rng(seed)
    X = assay.sample_design(n_init + n_iter, rng)
    outcomes = assay.run_assay(X, rng)
    hv, first = _hv_curve_and_pass(X, outcomes)
    return RunResult(X, outcomes, hv, first, label="random")


def parego(n_iter: int, seed: int, n_init: int = 6, n_cand: int = 1500) -> RunResult:
    rng = np.random.default_rng(seed)
    X = assay.sample_design(n_init, rng)
    Y = assay.run_assay(X, rng)

    for _ in range(n_iter):
        Yu = objectives_to_unit(Y)
        w = rng.dirichlet(np.ones(len(assay.OBJECTIVES)))
        s = _tchebycheff(Yu, w)
        Xn = normalize_X(X)
        gp = _fit_scalar_gp(Xn, s)

        # Candidate pool: random draws plus local perturbations of the incumbent.
        cand = assay.sample_design(n_cand, rng)
        best_idx = int(np.argmax(s))
        local = X[best_idx] + rng.normal(0, 0.05, (n_cand // 5, len(assay.PARAM_NAMES))) * (_HI - _LO)
        cand = np.clip(np.vstack([cand, local]), _LO, _HI)

        ei = _expected_improvement(gp, normalize_X(cand), best=s.max())
        x_next = cand[int(np.argmax(ei))][None, :]
        y_next = assay.run_assay(x_next, rng)

        X = np.vstack([X, x_next])
        Y = {k: np.concatenate([Y[k], y_next[k]]) for k in Y}

    hv, first = _hv_curve_and_pass(X, Y)
    return RunResult(X, Y, hv, first, label="parego")

"""The world model: a probabilistic surrogate of the assay learned from data.

For each objective we fit a Gaussian process (Matern-5/2 + white noise). A GP is
the right "world model" for wet-lab data volumes: it is sample efficient (useful
from ~10 runs), and, crucially, it reports calibrated predictive uncertainty so
the agent knows where it is guessing. That uncertainty is what drives both the
acquisition function (explore vs exploit) and the conformal QC gate.

This is deliberately not a deep network. At 20-60 experiments a neural surrogate
overfits and its uncertainty is not trustworthy; that is a real-vs-hype call, not
a limitation to apologize for.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .assay import BOUNDS, OBJECTIVES, PARAM_NAMES

_LO = np.array([BOUNDS[p][0] for p in PARAM_NAMES])
_HI = np.array([BOUNDS[p][1] for p in PARAM_NAMES])


def normalize_X(X: np.ndarray) -> np.ndarray:
    """Scale real-unit protocols to the unit cube for the GP."""
    return (np.atleast_2d(np.asarray(X, dtype=float)) - _LO) / (_HI - _LO)


class WorldModel:
    """One GP per objective, sharing a common input normalization."""

    def __init__(self, objectives: tuple[str, ...] = OBJECTIVES, seed: int = 0):
        self.objectives = objectives
        self.seed = seed
        self.gps: dict[str, GaussianProcessRegressor] = {}

    def _new_gp(self) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e3))
            * Matern(length_scale=np.ones(len(PARAM_NAMES)), nu=2.5)
            + WhiteKernel(0.05, (1e-4, 1.0))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            alpha=1e-8,
            random_state=self.seed,  # seeded restarts, so the gate repeats
        )

    def fit(self, X: np.ndarray, Y: dict[str, np.ndarray]) -> "WorldModel":
        Xn = normalize_X(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in self.objectives:
                gp = self._new_gp()
                gp.fit(Xn, np.asarray(Y[name], dtype=float))
                self.gps[name] = gp
        return self

    def predict(self, X: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return {objective: (mean, std)} for each query protocol."""
        Xn = normalize_X(X)
        out = {}
        for name, gp in self.gps.items():
            mu, sd = gp.predict(Xn, return_std=True)
            out[name] = (mu, sd)
        return out

    def predict_mean(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {k: v[0] for k, v in self.predict(X).items()}

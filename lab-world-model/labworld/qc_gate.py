"""Conformal QC gate: calibrated accept / reject / escalate decisions.

A point prediction from the world model is not a release decision. This gate wraps
the world model in split conformal prediction to get finite-sample, distribution-
free prediction intervals with a coverage guarantee, then triages each proposed
protocol against the acceptance criteria:

  - interval entirely clears the threshold  -> AUTO-ACCEPT (release)
  - interval entirely below the threshold    -> AUTO-REJECT (do not run)
  - interval straddles the threshold         -> ESCALATE to the bench for ground truth

This is the machine encoding of senior-scientist judgment about when the model is
allowed to decide and when a human/robot must generate real data. It also bounds
how wrong the auto-decisions can be, which is what "decision-grade" has to mean.

We use normalized nonconformity (residual divided by GP sigma), so intervals widen
exactly where the assay is noisy (low input) - the heteroscedastic regime the naive
point gate gets wrong.
"""

from __future__ import annotations

import numpy as np

from . import assay
from .world_model import WorldModel

ACCEPT, REJECT, ESCALATE = "accept", "reject", "escalate"


class ConformalQCGate:
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha  # target miscoverage; 0.10 -> 90% intervals
        self.model = WorldModel()
        self.q: dict[str, float] = {}

    def fit(self, X_train, Y_train, X_cal, Y_cal) -> "ConformalQCGate":
        self.model.fit(X_train, Y_train)
        preds = self.model.predict(X_cal)
        n = len(X_cal)
        # Split-conformal quantile level with finite-sample correction.
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        for name in assay.OBJECTIVES:
            mu, sd = preds[name]
            resid = np.abs(np.asarray(Y_cal[name]) - mu) / (sd + 1e-6)
            self.q[name] = float(np.quantile(resid, level))
        return self

    def intervals(self, X) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        preds = self.model.predict(X)
        out = {}
        for name in assay.OBJECTIVES:
            mu, sd = preds[name]
            half = self.q[name] * (sd + 1e-6)
            out[name] = (mu - half, mu + half)
        return out

    def triage(self, X) -> tuple[np.ndarray, dict]:
        """Return per-protocol decision plus the per-objective calls."""
        iv = self.intervals(X)
        n = len(np.atleast_2d(X))
        per_obj = {}
        confident_pass = np.ones(n, dtype=bool)
        any_fail = np.zeros(n, dtype=bool)
        for name in assay.OBJECTIVES:
            lo, hi = iv[name]
            t = assay.ACCEPTANCE[name]
            obj_pass = lo >= t          # whole interval clears
            obj_fail = hi < t           # whole interval below
            per_obj[name] = np.where(obj_pass, ACCEPT, np.where(obj_fail, REJECT, ESCALATE))
            confident_pass &= obj_pass
            any_fail |= obj_fail
        decision = np.where(confident_pass, ACCEPT, np.where(any_fail, REJECT, ESCALATE))
        return decision, {"intervals": iv, "per_objective": per_obj}

    def empirical_coverage(self, X_test, Y_test) -> dict[str, float]:
        """Fraction of held-out truth captured by the intervals (target 1-alpha)."""
        iv = self.intervals(X_test)
        cov = {}
        for name in assay.OBJECTIVES:
            lo, hi = iv[name]
            y = np.asarray(Y_test[name])
            cov[name] = float(np.mean((y >= lo) & (y <= hi)))
        return cov

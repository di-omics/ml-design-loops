"""LEARN arm: attribute measured brightness to mutations, and gate promotions.

A ridge regression on one-hot mutation features fits log(brightness) as a sum of
per-mutation effects. Because variants stack several mutations, a plain per-well
average would confound them; the linear model disentangles each mutation's
contribution, which is what lets the loop combine winners across rounds.

The promotion gate is the acceptance decision: promote a variant only if it is
confidently brighter than wild type and actually grew. (The rigorous conformal
accept/reject/escalate version lives in the sibling repo lab-world-model; here we
use its lightweight cousin.)
"""

from __future__ import annotations

import numpy as np

from . import params

Mutation = tuple[int, str]


def variant_mutations(wt: str, seq: str) -> list[Mutation]:
    return [(i, b) for i, (a, b) in enumerate(zip(wt, seq)) if a != b]


class LearnedModel:
    def __init__(self, wt: str, ridge_lambda: float = 1.0):
        self.wt = wt
        self.ridge_lambda = ridge_lambda
        self.obs_muts: list[list[Mutation]] = []
        self.obs_logrfu: list[float] = []
        self.effects: dict[Mutation, float] = {}
        self.intercept: float = 0.0

    def observe(self, seqs: list[str], rfu: np.ndarray) -> None:
        for s, r in zip(seqs, rfu):
            self.obs_muts.append(variant_mutations(self.wt, s))
            self.obs_logrfu.append(float(np.log(max(r, 1e-3))))

    def fit(self) -> None:
        index: dict[Mutation, int] = {}
        for muts in self.obs_muts:
            for m in muts:
                index.setdefault(m, len(index))
        if not index:
            return
        n, p = len(self.obs_muts), len(index)
        X = np.zeros((n, p))
        for r, muts in enumerate(self.obs_muts):
            for m in muts:
                X[r, index[m]] = 1.0
        y = np.array(self.obs_logrfu)
        self.intercept = float(y.mean())
        yc = y - self.intercept
        A = X.T @ X + self.ridge_lambda * np.eye(p)
        w = np.linalg.solve(A, X.T @ yc)
        self.effects = {m: float(w[j]) for m, j in index.items()}

    def top_mutations(self, k: int) -> list[Mutation]:
        # Best positive-effect substitution per position, so stacking winners
        # combines distinct positions instead of colliding at one.
        best: dict[int, tuple[Mutation, float]] = {}
        for (i, a), e in self.effects.items():
            if e > 0.06 and (i not in best or e > best[i][1]):  # confident winners only
                best[i] = ((i, a), e)
        ranked = sorted(best.values(), key=lambda t: t[1], reverse=True)
        return [m for m, _ in ranked[:k]]

    def avoid_positions(self, threshold: float = -0.3) -> set[int]:
        return {m[0] for m, e in self.effects.items() if e < threshold}

    def predict_log(self, seq: str) -> float:
        return self.intercept + sum(self.effects.get(m, 0.0)
                                    for m in variant_mutations(self.wt, seq))


def promotion_gate(rfu: np.ndarray, od600: np.ndarray) -> np.ndarray:
    """Per-variant decision: 'accept' / 'reject' / 'escalate' (retest)."""
    decision = np.full(len(rfu), "escalate", dtype=object)
    accept = (rfu >= params.ACCEPT_REL_BRIGHTNESS) & (od600 >= params.ACCEPT_MIN_OD)
    reject = rfu < 1.0
    decision[accept] = "accept"
    decision[reject & ~accept] = "reject"
    return decision

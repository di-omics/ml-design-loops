"""The hidden "wet lab": a synthetic, learnable GFP brightness landscape.

This stands in for the physical assay so the closed loop runs without hardware.
It is explicitly synthetic - not a fitted DMS dataset - but its structure is real
GFP biology and it is deliberately an ADDITIVE, learnable landscape so that a
competent design->learn loop can climb it and be seen to work:

  - the chromophore (S65-Y66-G67) must stay intact or the protein is dark;
  - a set of beneficial positions carry a positive effect for a target residue,
    with partial credit for chemically similar residues (a gradient the loop can
    follow). The largest-effect beneficial mutations are the canonical, published
    GFP brighteners (F64L, S65T = EGFP; plus folding-reporter substitutions);
  - a buried structural core is intolerant to mutation;
  - every other substitution is mildly deleterious on average.

Effects are additive in log-brightness, with a soft cap (mild diminishing
returns) so stacking winners does not explode. A good loop should climb several
fold and rank the true brighteners at the top; it should avoid chromophore/core.
"""

from __future__ import annotations

import numpy as np

from .sequences import AVGFP, CHROMOPHORE_IDX

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}

# Physicochemical groups, for partial-credit gradients around a target residue.
_GROUPS = ["AILMVC", "FWY", "STNQ", "KRH", "DE", "GP"]

# Canonical avGFP brighteners: (0-indexed pos, from, to, log bonus). Kept only if
# from matches wild type (validated at construction).
_CANONICAL = [
    (63, "F", "L", 0.50),   # F64L (EGFP)
    (64, "S", "T", 0.55),   # S65T (EGFP)
    (71, "S", "A", 0.24),   # S72A
    (98, "F", "S", 0.28),   # F99S
    (152, "M", "T", 0.32),  # M153T
    (162, "V", "A", 0.26),  # V163A
    (166, "I", "T", 0.24),  # I167T
]
_CORE_IDX = (23, 90, 115, 173, 26, 28)
LOG_CAP = np.log(12.0)  # soft cap at ~12x wild-type brightness


def _group_of(aa: str) -> str:
    for g in _GROUPS:
        if aa in g:
            return g
    return ""


class GFPOracle:
    def __init__(self, wt: str = AVGFP, seed: int = 0, n_synthetic_beneficial: int = 48):
        self.wt = wt
        L = len(wt)
        rng = np.random.default_rng(1234)  # fixed landscape, independent of run seed
        self.rng = np.random.default_rng(seed)

        # Baseline: every substitution is mildly deleterious.
        self.E = rng.normal(-0.12, 0.05, (L, 20))
        for i in range(L):
            self.E[i, AA_INDEX[wt[i]]] = 0.0  # no change = no effect

        # Structural core: strongly intolerant.
        for i in _CORE_IDX:
            self.E[i, :] = -1.3
            self.E[i, AA_INDEX[wt[i]]] = 0.0

        # Chromophore: broken unless it is a known-good substitution.
        for i in CHROMOPHORE_IDX:
            self.E[i, :] = -4.0
            self.E[i, AA_INDEX[wt[i]]] = 0.0

        # Beneficial positions: canonical brighteners (validated) + synthetic ones.
        # Only the primary target residue per position is a "brightener"; chemical
        # neighbours get partial credit in E but are not counted as ground truth.
        self.canonical = []
        primary = []
        for (i, frm, to, bonus) in _CANONICAL:
            if wt[i] == frm:
                self._set_beneficial(i, to, bonus)
                self.canonical.append((i, to))
                primary.append((i, to))
        taken = {i for (i, _) in self.canonical} | set(_CORE_IDX) | set(CHROMOPHORE_IDX)
        pool = [i for i in range(L) if i not in taken]
        for i in rng.choice(pool, size=n_synthetic_beneficial, replace=False):
            to = AA[int(rng.integers(20))]
            while to == wt[i]:
                to = AA[int(rng.integers(20))]
            self._set_beneficial(int(i), to, float(rng.uniform(0.20, 0.45)))
            primary.append((int(i), to))

        # Primary beneficial targets only (the ground-truth brighteners).
        self.brighteners = {(i, to): self.E[i, AA_INDEX[to]] for (i, to) in primary}

    def _set_beneficial(self, i: int, target: str, bonus: float) -> None:
        self.E[i, AA_INDEX[target]] = bonus
        grp = _group_of(target)
        for a in grp:
            if a != target and a != self.wt[i]:
                self.E[i, AA_INDEX[a]] = max(self.E[i, AA_INDEX[a]], 0.4 * bonus)

    def _log_brightness(self, seq: str) -> float:
        lb = sum(self.E[i, AA_INDEX[b]] for i, (a, b) in enumerate(zip(self.wt, seq)) if a != b)
        # soft cap (diminishing returns) on the positive side only
        if lb > 0:
            lb = LOG_CAP * np.tanh(lb / LOG_CAP)
        return float(lb)

    def measure(self, seqs: list[str]) -> dict[str, np.ndarray]:
        """Per-variant (rfu, od600) as a Cytation 5 would report, relative to WT."""
        lb = np.array([self._log_brightness(s) for s in seqs])
        true_rel = np.exp(lb)
        load = np.array([sum(a != b for a, b in zip(self.wt, s)) for s in seqs])
        od = np.clip(0.9 - 0.010 * load + self.rng.normal(0, 0.03, len(seqs)), 0.05, 1.2)
        noise = self.rng.normal(0, 0.03 + 0.06 * (1.0 / (1.0 + true_rel)), len(seqs))
        rfu = np.clip(true_rel * (1.0 + noise) * (od / 0.9), 0.0, None)
        return {"rfu": rfu, "od600": od, "true_rel_brightness": true_rel}

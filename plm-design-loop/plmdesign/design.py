"""Closed-loop in-silico directed evolution guided by the language model.

Simulated annealing over point mutations that climbs the ESM-2 pseudo-log-
likelihood (the model's "naturalness" score), under two engineering constraints:

  - a mutation budget (at most `max_mut` substitutions from wild type), and
  - a set of fixed positions that must not be touched (e.g. catalytic residues).

Each step: score the current sequence once, pick a position the model is least
happy with, propose a substitution drawn from the model's own distribution there,
and accept it with a Metropolis rule. Same accept/propose/score loop shape as the
wet-lab optimizer in lab-world-model, just with a foundation model as the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scorer import AA, AA_INDEX, ESMScorer


@dataclass
class DesignResult:
    wild_type: str
    designed: str
    wt_pll: float
    designed_pll: float
    mutations: list[str]           # e.g. ["A26V", "K101R", ...]
    trajectory: list[dict]         # per-step pll / n_mut / accepted


def _mutations(wt: str, var: str) -> list[str]:
    return [f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(wt, var)) if a != b]


def evolve(
    scorer: ESMScorer,
    wt: str,
    n_steps: int = 150,
    max_mut: int = 20,
    fixed: frozenset[int] = frozenset(),
    seed: int = 0,
    t_start: float = 2.0,
    t_end: float = 0.2,
    sample_temp: float = 1.0,
) -> DesignResult:
    rng = np.random.default_rng(seed)
    cur = list(wt)
    cur_pll, cur_lp = scorer.pll_and_lp(wt)
    best, best_pll = cur[:], cur_pll
    cand_positions = [i for i in range(len(wt)) if i not in fixed]

    traj = [{"step": 0, "pll": cur_pll, "n_mut": 0, "accepted": True}]
    for step in range(1, n_steps + 1):
        temp = t_start * (t_end / t_start) ** ((step - 1) / max(1, n_steps - 1))

        # Prefer positions where the model dislikes the current residue.
        cur_res_lp = np.array([cur_lp[i, AA_INDEX[cur[i]]] for i in cand_positions])
        w = np.exp(-cur_res_lp)
        w /= w.sum()
        pos = int(rng.choice(cand_positions, p=w))

        # Propose a substitution from the model's distribution at that position.
        probs = np.exp(cur_lp[pos] / sample_temp)
        probs[AA_INDEX[cur[pos]]] = 0.0
        probs /= probs.sum()
        new_aa = AA[int(rng.choice(20, p=probs))]

        cand = cur[:]
        cand[pos] = new_aa
        if sum(a != b for a, b in zip(cand, wt)) > max_mut:
            traj.append({"step": step, "pll": cur_pll,
                         "n_mut": sum(a != b for a, b in zip(cur, wt)), "accepted": False})
            continue

        cand_pll, cand_lp = scorer.pll_and_lp("".join(cand))
        delta = cand_pll - cur_pll
        if delta > 0 or rng.random() < np.exp(delta / max(temp, 1e-6)):
            cur, cur_pll, cur_lp = cand, cand_pll, cand_lp
            accepted = True
        else:
            accepted = False
        if cur_pll > best_pll:
            best, best_pll = cur[:], cur_pll
        traj.append({"step": step, "pll": cur_pll,
                     "n_mut": sum(a != b for a, b in zip(cur, wt)), "accepted": accepted})

    designed = "".join(best)
    return DesignResult(
        wild_type=wt,
        designed=designed,
        wt_pll=traj[0]["pll"],
        designed_pll=best_pll,
        mutations=_mutations(wt, designed),
        trajectory=traj,
    )

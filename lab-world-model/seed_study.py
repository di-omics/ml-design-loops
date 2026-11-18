"""Does the demo's result hold over more seeds?

run_demo.py uses 12 seeds. This runs the same optimizer and the same random
search, with the same budget, over 100 seeds and writes seed_study.json.

    python seed_study.py        # about three minutes on a laptop
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from labworld.optimize import parego, random_search

N_SEEDS = 100
N_ITER = 30           # same budget as run_demo.py
N_INIT = 6
BUDGET = N_INIT + N_ITER
N_BOOT = 5000


def censored(first_pass: int) -> int:
    # A seed that never passes within budget counts as the full budget,
    # the same rule run_demo.py uses.
    return first_pass if first_pass > 0 else BUDGET


def interval(values) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 1), round(float(hi), 1)]


def main() -> None:
    agent, rand, agent_hv, rand_hv = [], [], [], []
    for seed in range(N_SEEDS):
        a = parego(N_ITER, seed, n_init=N_INIT)
        r = random_search(N_ITER, seed, n_init=N_INIT)
        agent.append(censored(a.first_pass))
        rand.append(censored(r.first_pass))
        agent_hv.append(float(a.hv_curve[-1]))
        rand_hv.append(float(r.hv_curve[-1]))
    agent, rand = np.array(agent, float), np.array(rand, float)
    agent_hv, rand_hv = np.array(agent_hv), np.array(rand_hv)

    # Bootstrap over seeds, resampling the agent and the baseline together.
    rng = np.random.default_rng(0)
    fewer, gain = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, N_SEEDS, N_SEEDS)
        fewer.append(100 * (1 - np.median(agent[i]) / np.median(rand[i])))
        gain.append(100 * (agent_hv[i].mean() / rand_hv[i].mean() - 1))

    out = {
        "seeds": N_SEEDS,
        "budget": BUDGET,
        "hypervolume_mean_agent": round(float(agent_hv.mean()), 4),
        "hypervolume_mean_random": round(float(rand_hv.mean()), 4),
        "hypervolume_gain_pct": round(
            100 * (agent_hv.mean() / rand_hv.mean() - 1), 1),
        "hypervolume_gain_pct_95": interval(gain),
        "seeds_agent_higher_hypervolume": int((agent_hv > rand_hv).sum()),
        "never_passed_agent": int((agent >= BUDGET).sum()),
        "never_passed_random": int((rand >= BUDGET).sum()),
        "median_first_pass_agent": float(np.median(agent)),
        "median_first_pass_random": float(np.median(rand)),
        "fewer_experiments_pct": round(
            100 * (1 - np.median(agent) / np.median(rand)), 1),
        "fewer_experiments_pct_95": interval(fewer),
        "seeds_agent_faster": int((agent < rand).sum()),
        "seeds_tied": int((agent == rand).sum()),
        "seeds_random_faster": int((agent > rand).sum()),
    }
    path = Path(__file__).parent / "seed_study.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

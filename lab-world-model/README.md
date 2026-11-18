# lab-world-model

A closed-loop, uncertainty-aware optimizer for a wet-lab assay, in ~600 lines of
dependency-light Python. It stands up the three ML pieces that an autonomous-lab
platform actually needs and that survive contact with real experimental data:

1. a **world model** of the assay (a probabilistic surrogate you can plan against),
2. a **closed-loop agent** that designs the next experiment (multi-objective
   Bayesian optimization), and
3. a **conformal QC gate** that turns calibrated uncertainty into accept / reject /
   escalate-to-bench decisions with a coverage guarantee.

Everything runs on CPU in under 30 seconds. No wet lab required: a physically
motivated simulator plays the role of the bench so the loop can be exercised and
scored end to end.

```
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_demo.py
```

Outputs land in `figures/` and `metrics.json`.

## Why this design (the real-vs-hype calls)

The point of the repo is not "we used ML". It is the set of judgments about which
ML is load-bearing at wet-lab data volumes:

- **Gaussian process, not a deep net, as the world model.** At 20-60 experiments a
  neural surrogate overfits and its uncertainty is not trustworthy. A GP is sample
  efficient and its predictive variance is calibrated enough to drive both the
  acquisition function and the QC gate. This is a feature, not a compromise.
- **Bayesian optimization, not deep RL, for the loop.** Model-free RL needs orders
  of magnitude more episodes than a bench can afford, and the sim-to-real gap on a
  learned assay model is unforgiving. Sequential model-based optimization is what
  "self-optimizing experimental design" should mean in a data-poor regime.
- **Conformal prediction for the release decision.** A point prediction is not a
  release call. Split conformal gives distribution-free, finite-sample coverage,
  so the gate can bound how wrong its auto-decisions are and escalate the rest.
  This is the machine encoding of "know where your expertise ends and go get
  ground truth."
- **Multi-objective on purpose.** Yield, library complexity, and purity trade off
  against each other (over-cycling kills complexity; loose bead ratios rescue yield
  but carry over adapter dimers). Optimizing a single scalar hides the tradeoff a
  senior scientist actually reasons about; ParEGO keeps the whole Pareto front.

## What the demo shows (12-seed run)

| metric | closed-loop agent | random search |
| --- | --- | --- |
| Pareto hypervolume after 36 runs | 0.86 | 0.75 (+14%) |
| median experiments to first release-grade protocol | 12.5 | 15.5 (~19% fewer) |
| seeds that never found a passing protocol in budget | 1 / 12 | 2 / 12 |

The run is seeded end to end, so `run_demo.py` prints these numbers every time.

Conformal QC gate (90% target coverage, trained on ~120 runs):

- empirical coverage: yield 96%, complexity 92%, purity 94% (calibrated).
- auto-decides ~79% of proposed protocols at 0% error; escalates ~21% to the bench.
- a naive point gate (no uncertainty) is wrong on 0% of the protocols the conformal
  gate auto-decides, but 12.5% of the ones it escalates. The gate is targeting its
  "go get ground truth" calls exactly where they are needed.

Figures:

- `01_convergence.png` hypervolume vs experiment count, agent vs random.
- `02_first_pass.png` experiments to first release-grade protocol.
- `03_pareto.png` protocols explored; the agent concentrates on the release corner.
- `04_conformal_qc.png` calibration, triage rates, and where errors concentrate.

## Does it hold over more seeds? (100-seed check)

Twelve seeds is a small sample. `seed_study.py` runs the same optimizer and the same
random search, with the same budget, over 100 seeds and writes `seed_study.json`.

| metric | closed-loop agent | random search |
| --- | --- | --- |
| mean Pareto hypervolume after 36 runs | 0.854 | 0.741 |
| seeds that never found a passing protocol in budget | 3 / 100 | 20 / 100 |
| median experiments to first release-grade protocol | 15 | 16 |

Two results hold. The agent ends with about 15% more hypervolume (95% interval 14 to
17) and is ahead on all 100 seeds, and it misses the release bar within budget far
less often.

One does not. The gap in the median is about 6% with a 95% interval of -10% to +33%,
which spans zero: the agent gets there first on 60 seeds, ties on 7, and is slower on
33. The 19% in the 12-seed table is one draw of a noisy statistic. Read it as an
example, not as the effect size.

## Architecture

```
labworld/
  assay.py         ground-truth wet-lab simulator (hidden from the agent)
  world_model.py   GP surrogate per objective = the learned world model
  optimize.py      ParEGO multi-objective BO + random baseline + hypervolume
  qc_gate.py       split-conformal accept / reject / escalate gate
run_demo.py        runs the loop over seeds, exercises the gate, writes figures
seed_study.py      the same agent-vs-random comparison over 100 seeds
```

The simulator's response surfaces are physically motivated (log-scaled input mass,
saturating amplification, duplication from over-cycling, SPRI retention tradeoff)
with heteroscedastic noise that is larger at low input. That heteroscedasticity is
the whole reason the QC gate needs calibrated intervals rather than a fixed margin.

## Where this extends (the platform roadmap)

The scaffold is provider- and assay-agnostic. Natural next layers:

- **Foundation-model priors.** Warm-start the world model or featurize protocols
  with genomics/protein FMs (Evo2 for variant and off-target effect, Enformer for
  regulatory readout, ESM for enzyme design, scGPT/Geneformer for single-cell
  perturbation) instead of learning every surface from scratch.
- **Real backends.** Swap `assay.run_assay` for a PyLabRobot-driven liquid handler
  and a plate reader; the loop is unchanged.
- **LLM orchestration.** An agent that compiles a protocol brief into the design
  space, proposes acceptance criteria, and reads the QC gate's escalations.
- **Batched / cost-aware acquisition.** qEHVI or batch ParEGO for parallel decks,
  with per-objective reagent cost in the utility.

## License

Open source. Author attribution follows the di-omics house convention.

# plm-design-loop

A protein-language-model design loop: use a pretrained **ESM-2** foundation model
as a zero-shot fitness oracle to (1) scan every possible point mutation of a
protein in one shot, and (2) run closed-loop in-silico directed evolution toward
higher model "naturalness" under real engineering constraints.

Real foundation model, real published scoring method, runs on an **Apple M1 GPU
(MPS) in ~15 seconds**. No CUDA box, no fine-tuning, ~900 MB self-contained venv.

```
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_demo.py
# heavier model if you want sharper predictions:
ESM_MODEL=facebook/esm2_t30_150M_UR50D ./.venv/bin/python run_demo.py
```

Outputs: `figures/`, `designed_variant.fasta`, `mutational_scan.npy`, `metrics.json`.

## What it does

**1. Single-shot mutational scan.** For the demo protein (wild-type avGFP, 238
residues) the model scores all 4,522 point mutants using the ESM-2 masked-marginal
method: mask each position, read the model's distribution, score every substitution
as its log-odds against the wild-type residue. This is a deep-mutational-scan
prediction with zero wet-lab data. Output is a 20 x L heatmap.

**2. Closed-loop directed evolution.** Simulated annealing over point mutations
that climbs the ESM-2 pseudo-log-likelihood (naturalness), subject to two
engineering constraints a protein engineer actually imposes:
  - a mutation budget (at most N substitutions from wild type), and
  - fixed positions that must not be touched (here, the chromophore tripeptide).

Same propose / score / accept loop shape as the wet-lab optimizer in the companion
repo `lab-world-model`; here the oracle is a foundation model instead of a GP.

## What the demo shows (esm2_t12_35M, M1 MPS)

- 4,522 point mutants scored in ~10 s on the M1 GPU.
- The design loop lifts naturalness from -116.2 to -107.4 with 17 mutations while
  holding the chromophore fixed.
- A biology sanity check that is more honest than "the model conserves the
  functional core":
  - the most-conserved positions are **buried structural residues, glycine-rich**
    (G24, G91, G116, ...), which is textbook beta-barrel conservation;
  - at the chromophore **S65-Y66-G67**, the model keeps **G67** conserved but rates
    **Y66 highly substitutable (88th percentile)** - which is correct family
    biology, because Y66 is exactly where nature engineers GFP color variants
    (Y66H -> BFP, Y66W -> CFP);
  - it also (correctly) reads **S65 as tolerant**, the site of the famous folding
    mutation S65T in EGFP.

The point: **ESM conservation reflects evolutionary constraint across the protein
family, not single-protein functional criticality.** Knowing that difference is
what keeps a foundation model from being used as an oracle it is not.

Figures:
- `01_mutational_scan.png` full 20 x L deep-mutational-scan heatmap.
- `02_chromophore_zoom.png` chromophore region with wild-type residues boxed.
- `03_design_trajectory.png` naturalness and mutation count vs design step.

## Architecture

```
plmdesign/
  scorer.py     ESM-2 wrapper: wt-marginal / masked-marginal scans, pseudo-LL,
                MPS with CPU fallback
  design.py     simulated-annealing closed-loop designer with budget + fixed sites
  sequences.py  demo wild-type protein (avGFP)
run_demo.py     scan + design + figures + metrics
```

## Honest limits (the real-vs-hype notes)

- **Naturalness is not fitness.** ESM pseudo-log-likelihood is a strong, cheap
  prior for foldability and family plausibility. It is not a stand-in for the
  actual assay (brightness, expression, thermostability). In a real program this
  proposes candidates; the wet lab (or a task-specific model) is ground truth.
- **A small model for speed.** t12_35M is used so the whole thing runs in seconds
  on a laptop GPU. Predictions sharpen with 150M / 650M at higher cost.
- **Point mutations only, single objective.** Epistasis, indels, and multi-property
  objectives are the obvious next layer (and pair naturally with the multi-objective
  Bayesian optimizer in `lab-world-model`).

## Where this goes next

- Multi-objective design: combine naturalness with a structure model (ESMFold /
  AlphaFold pLDDT) or a task head, optimized with batch Bayesian optimization.
- Swap avGFP for an enzyme of interest (e.g. a transposase) to propose stabilizing
  or activity-preserving variants for expression.
- Wire the accepted designs into an ordering / synthesis and expression workflow.

## License

Open source. Author attribution follows the di-omics house convention.

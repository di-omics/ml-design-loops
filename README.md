# ml-design-loops

Machine learning for laboratory biology, built evaluation-first. Three closed-loop
systems that each decide what to do next and score the result against ground truth
with an explicit quality gate. Everything here runs in simulation, on CPU. The wet-lab
arm sits behind PyLabRobot backends that hardware can replace; no hardware run is
included in this repo.

The common thread is the gate, not the model: acceptance criteria, calibrated
uncertainty, and QC checkpoints that define what "correct" means before anything is
promoted. That is the part that has to hold when a method leaves the lab that
invented it.

## Components

Each folder is self-contained and has its own README and requirements. All numbers
below come from simulation.

### [lab-world-model](lab-world-model)
A closed-loop assay optimizer: a Gaussian-process world model of a low-input
library-prep assay, multi-objective Bayesian optimization (ParEGO), and a
split-conformal accept / reject / escalate QC gate with a coverage guarantee.
Over 100 seeds it ends with about 15 percent more Pareto hypervolume than random
search, ahead on every seed, and fails to find a release-grade protocol within budget
on 3 of the 100 seeds, against 20 for random search. The gate auto-decides roughly 79 percent of protocols at zero error
while escalating the rest to the bench. sklearn / scipy, the demo runs in about 20
seconds.

### [plm-design-loop](plm-design-loop)
A protein language model design loop: ESM-2 zero-shot deep-mutational scanning
(masked-marginal) plus simulated-annealing in-silico directed evolution over
pseudo-log-likelihood. Runs on Apple MPS in about 15 seconds. The evaluation is a
biology check: the model conserves the structural core and rates chromophore
position 66 as substitutable, which is exactly where nature makes GFP color
variants.

### [autonomous-protein-design](autonomous-protein-design)
An agentic Design, Build, Test, Learn engine: ESM-2 proposes variants, a liquid handler
STARlet builds and tests them (real PyLabRobot, with ODTC and Heater-Shaker), a
Cytation 5 reads GFP brightness, and a per-mutation model closes the loop through a
promotion gate. Simulation only here; each instrument sits behind a backend that
hardware can replace. The
avGFP demo climbs to about 2x wild type and recovers a canonical brightener.

## Running any component

```
cd <component>
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_demo.py     # run_dbtl.py in autonomous-protein-design
```

## House style

Figures use a shared style (`vizstyle.py` per component): Manrope, a pastel
blue / green / purple palette, and thin black lines on white.

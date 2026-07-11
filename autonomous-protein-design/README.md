# autonomous-protein-design

An agentic **Design -> Build -> Test -> Learn** engine for protein engineering. An
ESM-2 language model proposes variants, a liquid handler builds and tests them,
a Cytation 5 reads GFP brightness, and a per-mutation model closes the loop by
choosing what to make next. It runs **sim-first, CPU-only, end to end** on a laptop;
switching to hardware is a one-line backend change per instrument.

This is the wet-lab arm that turns in-silico design into a testable, self-improving
campaign. It sits between the sibling repos: `plm-design-loop` (the ESM design
method) feeds it, and `lab-world-model` (Bayesian optimization + the rigorous
conformal QC gate) is the model-based learner it grows into.

```
                 Design                    Build + Test                     Learn
   ESM-2 proposes variants  ->  STARlet + ODTC + Heater-Shaker  ->  per-mutation model
   (foundation-model prior)     assemble, transform, express         + promotion gate
                                Cytation 5 reads GFP / OD                 |
        ^_________________________ the agent picks the next batch ________|
```

## Quickstart

```
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python run_dbtl.py            # ESM design arm if torch is present
./.venv/bin/python run_dbtl.py --no-esm   # light surrogate designer, no torch
./.venv/bin/python run_dbtl.py --hardware # drive real STAR/ODTC/HHS/Cytation5 (bench only)
```

`--hardware` swaps every simulator backend for the real one via `autoprotein/hardware.py`
(STARBackend, ExperimentalODTCBackend, HamiltonHeaterShakerBackend, CytationBackend);
connection params live in `params.py` marked `TODO(hardware)`. The default runs
fully in simulation.

The core install is light (PyLabRobot + numpy + matplotlib). The ESM design arm is
an optional extra: `pip install torch transformers` (an ESM-2 35M model, runs on
Apple MPS). Outputs: `figures/`, `metrics.json`, and `run_log.txt` (the first
round's real STARlet + ODTC + Heater-Shaker + Cytation 5 choreography).

## What the demo shows (avGFP, 8 rounds x 24 variants, seed 0)

- Closed-loop brightness climbs **1.0x -> ~2x wild type**, monotonically in the
  best-kept variant, by discovering and recombining brightening mutations.
- The learner is **~67% precise**: two thirds of its top-ranked mutations are true
  brighteners, and it **recovers a canonical GFP brightener (V163A)** from scratch.
- It correctly **penalizes the chromophore and structural core** (largest negative
  learned effects), i.e. it learns what not to touch.
- **7 variants pass the QC promotion gate** (confidently brighter than wild type
  and grew).
- `run_log.txt` is ~1300 lines of genuine PyLabRobot deck operations (tip pickups,
  trough aspirations, plate dispenses, thermocycler profile, shake, reader reads).

Figures:
- `01_dbtl_improvement.png` brightness vs DBTL round (best-kept, best-in-round, mean).
- `02_mutation_recovery.png` learned per-mutation effects: brighteners recovered
  (green) vs chromophore/core penalized (orange).

## Architecture

```
autoprotein/
  design.py      ESM-2 / surrogate variant proposal (Design)
  build_star.py  STARlet assembly + transformation protocol, real PyLabRobot (Build)
  test_star.py   STARlet expression + phenotyping protocol (Test)
  readout.py     Cytation 5 GFP + OD readout; the sim/hardware seam
  oracle.py      synthetic ground-truth GFP landscape (the simulated wet lab)
  learn.py       ridge per-mutation effect model + promotion gate (Learn)
  dbtl.py        the agent that runs the closed loop
  starlib.py     shared PyLabRobot deck + transfer helpers
  params.py      every wet-lab value, each a TODO(protocol) placeholder
run_dbtl.py      run the loop, write figures / metrics / protocol log
```

## Sim-first, hardware-ready

Every robot call is real PyLabRobot against the chatterbox simulator. Going to the
bench is a one-line backend swap per instrument, and the protocol logic is unchanged:

| instrument | simulator (here) | hardware |
| --- | --- | --- |
| liquid handler | `LiquidHandlerChatterboxBackend` | `STARBackend` |
| Inheco ODTC | `ThermocyclerChatterboxBackend` | `thermocycling.inheco.ExperimentalODTCBackend` |
| Heater-Shaker | `HeaterShakerChatterboxBackend` | `HamiltonHeaterShakerBackend` |
| BioTek Cytation 5 | `PlateReaderChatterboxBackend` | `Cytation5Backend` |

## Honesty notes (the real-vs-hype calls)

- **The oracle is synthetic and labeled as such.** It is not a fitted DMS dataset.
  It encodes real GFP structure (intact chromophore required; canonical brighteners
  like F64L/S65T/V163A are the biggest wins; a buried core is intolerant) as an
  additive, learnable landscape, so the loop can be *seen* to work. On hardware the
  oracle is replaced by the Cytation 5; nothing else changes.
- **ESM naturalness is a prior, not fitness.** The design arm biases exploration
  toward plausible substitutions; the wet-lab readout is ground truth.
- **Pure ESM-tolerance weighting over-focuses.** It skips conserved-but-beneficial
  sites (the chromophore neighbours F64/S65), so exploration blends ESM with uniform
  sampling. That mix is why the loop can still find near-chromophore brighteners.
- **No invented protocol values.** Every reagent volume, temperature, and time lives
  in `params.py` marked `TODO(protocol)` and must be sourced from the real bench
  protocol before any hardware run.

## License

Open source. Author attribution follows the di-omics house convention.

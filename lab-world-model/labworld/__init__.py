"""labworld: a closed-loop, world-model-driven assay-optimization demo.

Modules:
  assay        ground-truth wet-lab simulator (hidden from the agent)
  world_model  GP surrogate = the learned, uncertainty-aware world model
  optimize     ParEGO multi-objective Bayesian optimization + baselines
  qc_gate      conformal-prediction accept / reject / escalate QC gate
"""

from . import assay, optimize, qc_gate, world_model  # noqa: F401

__all__ = ["assay", "world_model", "optimize", "qc_gate"]

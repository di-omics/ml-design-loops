"""autoprotein: an agentic Design-Build-Test-Learn engine for protein design.

  design      ESM-2 / surrogate proposal of candidate variants (Design)
  build_star  liquid handler variant-assembly + transformation protocol (Build)
  test_star   STARlet expression + phenotyping protocol (Test)
  readout     Cytation 5 GFP brightness readout, sim/hardware seam
  oracle      synthetic ground-truth GFP brightness (the simulated wet lab)
  learn       per-mutation effect model + promotion gate (Learn)
  dbtl        the agent that runs the closed loop
  params      all wet-lab values, every one a TODO(protocol) placeholder
"""

from . import (build_star, dbtl, design, hardware, learn, oracle, params,  # noqa: F401
               readout, starlib, test_star)

__all__ = ["design", "build_star", "test_star", "readout", "oracle", "learn",
           "dbtl", "params", "starlib", "hardware"]

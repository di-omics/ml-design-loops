"""READOUT: GFP brightness + OD on a BioTek Cytation 5.

This is the sim/hardware seam. In "sim" mode a synthetic GFP oracle supplies the
ground truth while the real PyLabRobot reader call path is still exercised
(chatterbox), so the choreography is identical to a hardware run. Switching
`mode="hardware"` builds a PlateReader on the real Cytation5Backend and parses
its fluorescence (GFP channel) and absorbance (OD600) reads instead.

    reader = Cytation5Readout(oracle=GFPOracle(), mode="sim")
    result = await reader.read(variants, read_plate)
"""

from __future__ import annotations

import numpy as np
from pylabrobot.plate_reading import PlateReader
from pylabrobot.resources.coordinate import Coordinate

from . import params
from .hardware import plate_reader_backend


def _values_from_reads(read_result) -> np.ndarray:
    """Best-effort: pull one numeric per well dict from a PlateReader read.

    TODO(hardware): pin to the exact Cytation5Backend return schema.
    """
    out = []
    for d in read_result:
        nums = [v for v in d.values() if isinstance(v, (int, float))]
        out.append(float(nums[0]) if nums else float("nan"))
    return np.asarray(out)


class Cytation5Readout:
    def __init__(self, oracle=None, mode: str = "sim"):
        if mode == "sim" and oracle is None:
            raise ValueError("sim mode needs a GFP oracle to supply ground truth")
        self.oracle = oracle
        self.mode = mode

    def _make_reader(self) -> PlateReader:
        return PlateReader(name="cytation5", size_x=200, size_y=140, size_z=200,
                           backend=plate_reader_backend(self.mode == "hardware"),
                           child_location=Coordinate(0, 0, 0))

    async def read(self, variants: list[str], read_plate=None) -> dict:
        reader = self._make_reader()
        await reader.setup()
        await reader.open()
        wells = read_plate.get_all_items()[:len(variants)] if read_plate is not None else None
        fl, ab = None, None
        try:
            fl = await reader.read_fluorescence(
                excitation_wavelength=int(params.GFP_EX_NM),
                emission_wavelength=int(params.GFP_EM_NM),
                focal_height=params.READ_FOCAL_HEIGHT_MM, wells=wells)
            ab = await reader.read_absorbance(wavelength=int(params.OD_WAVELENGTH_NM), wells=wells)
        except Exception:
            pass  # chatterbox may decline a read with no plate loaded; not fatal
        await reader.close()

        if self.mode == "hardware":
            # Real Cytation 5: GFP channel is brightness, absorbance is OD600.
            rfu, od = _values_from_reads(fl), _values_from_reads(ab)
            return {"rfu": rfu, "od600": od, "true_rel_brightness": rfu}
        m = self.oracle.measure(variants)
        return {"rfu": np.asarray(m["rfu"]), "od600": np.asarray(m["od600"]),
                "true_rel_brightness": np.asarray(m["true_rel_brightness"])}

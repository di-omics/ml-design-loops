"""TEST arm: express variants and stage them for phenotyping on a STARlet.

Choreography (real PyLabRobot calls, chatterbox by default):
  1. dispense expression media into the deep-well culture plate,
  2. inoculate from the transformed/colony source plate and mix,
  3. add inducer,
  4. incubate and shake on the heater-shaker,
  5. transfer an aliquot into the flat-bottom read plate for the Cytation 5.

The read plate is handed to the readout module. Volumes/temperatures/times come
from params.py (all TODO(protocol)).
"""

from __future__ import annotations

from pylabrobot.resources import (PLT_CAR_L5AC_A00, hamilton_1_trough_200mL_Vb,
    Cor_96_wellplate_2mL_Vb, CellVis_96_wellplate_350uL_Fb)
from pylabrobot.heating_shaking import HeaterShaker
from pylabrobot.resources.coordinate import Coordinate

from . import params
from .hardware import heater_shaker_backend
from .starlib import (make_liquid_handler, TipStream, distribute_from_trough,
    transfer_plate_to_plate, n_columns)


async def run_test(variants: list[str], hardware: bool = False):
    n = len(variants)
    ncol = n_columns(n)
    n_racks = max(1, (4 * ncol + 11) // 12)
    lh, racks = make_liquid_handler(n_tip_racks=n_racks, hardware=hardware)
    tips = TipStream(racks)

    carrier = PLT_CAR_L5AC_A00(name="test_carrier")
    media = hamilton_1_trough_200mL_Vb(name="expression_media")
    inducer = hamilton_1_trough_200mL_Vb(name="inducer")
    source = Cor_96_wellplate_2mL_Vb(name="transformed_source")
    culture = Cor_96_wellplate_2mL_Vb(name="culture_plate")
    read_plate = CellVis_96_wellplate_350uL_Fb(name="read_plate")
    for i, r in enumerate((media, inducer, source, culture, read_plate)):
        carrier[i] = r
    lh.deck.assign_child_resource(carrier, rails=params.RAILS_PLATES)

    hs = HeaterShaker(name="hhs", size_x=145, size_y=104, size_z=70,
                      backend=heater_shaker_backend(hardware, lh.backend),
                      child_location=Coordinate(0, 0, 0))
    await lh.setup()
    await hs.setup()

    await distribute_from_trough(lh, media, culture, n, params.MEDIA_UL, tips)
    await transfer_plate_to_plate(lh, source, culture, n, params.INOCULUM_UL, tips, mix_cycles=1)
    await distribute_from_trough(lh, inducer, culture, n, params.INDUCER_UL, tips)

    await hs.set_temperature(params.EXPRESSION_C, passive=True)  # allow cooling to setpoint
    # On hardware, wait_for_temperature here; the long induction shake covers
    # settling, and the simulator does not model passive cool-down.
    await hs.shake(speed=params.SHAKE_RPM)  # induce params.EXPRESSION_S (real-timed on hardware)
    await hs.stop_shaking()

    await transfer_plate_to_plate(lh, culture, read_plate, n, params.READ_ALIQUOT_UL, tips)

    summary = {
        "arm": "test",
        "n_variants": n,
        "columns": ncol,
        "media_uL_total": round(params.MEDIA_UL * n, 1),
        "induction_C": params.EXPRESSION_C,
        "read_plate": read_plate.name,
    }
    return read_plate, summary

"""BUILD arm: assemble variant constructs and set up transformation on a STARlet.

Choreography (all real PyLabRobot calls, chatterbox by default):
  1. distribute assembly master mix from a trough into the reaction plate,
  2. transfer each variant's template/oligos from the source plate and mix,
  3. thermocycle the reaction plate (ODTC on hardware) with the assembly profile,
  4. add DpnI and hold to digest template,
  5. add competent cells and heat-shock on the heater-shaker.

On hardware the reaction plate moves between the STAR deck, the ODTC, and the
HHS via the iSWAP; that transport is omitted in the simulator and flagged below.
Every volume/temperature/time comes from params.py (all TODO(protocol)).
"""

from __future__ import annotations

from pylabrobot.resources import (PLT_CAR_L5AC_A00, hamilton_1_trough_200mL_Vb,
    Cor_96_wellplate_2mL_Vb, Azenta4titudeFrameStar_96_wellplate_200ul_Vb)
from pylabrobot.heating_shaking import HeaterShaker
from pylabrobot.thermocycling import Thermocycler
from pylabrobot.resources.coordinate import Coordinate

from . import params
from .hardware import heater_shaker_backend, thermocycler_backend
from .starlib import (make_liquid_handler, TipStream, distribute_from_trough,
    transfer_plate_to_plate, n_columns)


async def run_build(variants: list[str], hardware: bool = False) -> dict:
    n = len(variants)
    ncol = n_columns(n)
    n_racks = max(1, (4 * ncol + 11) // 12)
    lh, racks = make_liquid_handler(n_tip_racks=n_racks, hardware=hardware)
    tips = TipStream(racks)

    carrier = PLT_CAR_L5AC_A00(name="build_carrier")
    mastermix = hamilton_1_trough_200mL_Vb(name="assembly_mastermix")
    dpni = hamilton_1_trough_200mL_Vb(name="dpni")
    cells = hamilton_1_trough_200mL_Vb(name="competent_cells")
    source = Cor_96_wellplate_2mL_Vb(name="variant_dna_source")
    reaction = Azenta4titudeFrameStar_96_wellplate_200ul_Vb(name="reaction_plate")
    for i, r in enumerate((mastermix, dpni, cells, source, reaction)):
        carrier[i] = r
    lh.deck.assign_child_resource(carrier, rails=params.RAILS_PLATES)

    tc = Thermocycler(name="odtc", size_x=140, size_y=100, size_z=100,
                      backend=thermocycler_backend(hardware), child_location=Coordinate(0, 0, 0))
    hs = HeaterShaker(name="hhs", size_x=145, size_y=104, size_z=70,
                      backend=heater_shaker_backend(hardware, lh.backend),
                      child_location=Coordinate(0, 0, 0))
    await lh.setup()
    await tc.setup()
    await hs.setup()

    # 1-2. assemble reactions
    await distribute_from_trough(lh, mastermix, reaction, n, params.MASTERMIX_UL, tips)
    await transfer_plate_to_plate(lh, source, reaction, n, params.TEMPLATE_DNA_UL, tips,
                                  mix_cycles=params.MIX_CYCLES)

    # 3. thermocycle (reaction plate -> ODTC via iSWAP on hardware)
    await tc.close_lid()
    await tc.set_lid_temperature([params.LID_TEMP_C])
    p = params.ASSEMBLY_PROFILE
    await tc.run_pcr_profile(
        pre_denaturation_temp=[p[0][0]], pre_denaturation_time=p[0][1],
        denaturation_temp=[p[1][0]], denaturation_time=p[1][1],
        annealing_temp=[p[2][0]], annealing_time=p[2][1],
        extension_temp=[p[3][0]], extension_time=p[3][1],
        num_cycles=params.ASSEMBLY_CYCLES,
        final_extension_temp=[p[4][0]], final_extension_time=p[4][1],
        storage_temp=[p[5][0]], storage_time=p[5][1],
        block_max_volume=params.MASTERMIX_UL + params.TEMPLATE_DNA_UL,
        lid_temperature=[params.LID_TEMP_C],
    )
    await tc.open_lid()

    # 4. DpnI digest
    await distribute_from_trough(lh, dpni, reaction, n, params.DPNI_UL, tips)
    await tc.set_block_temperature([params.DPNI_INCUBATE_C])

    # 5. transformation: add cells, heat-shock + recover on the HHS
    await distribute_from_trough(lh, cells, reaction, n, params.COMPETENT_CELLS_UL, tips)
    await hs.set_temperature(params.HEATSHOCK_C)
    await hs.wait_for_temperature()
    await hs.set_temperature(params.RECOVERY_C, passive=True)  # 42 -> 37 is cooling
    await hs.shake(speed=params.SHAKE_RPM)  # recover params.RECOVERY_S (real-timed on hardware)
    await hs.stop_shaking()

    return {
        "arm": "build",
        "n_variants": n,
        "columns": ncol,
        "tip_columns_used": tips.col,
        "mastermix_uL_total": round(params.MASTERMIX_UL * n, 1),
        "template_uL_total": round(params.TEMPLATE_DNA_UL * n, 1),
        "thermal_cycles": params.ASSEMBLY_CYCLES,
        "output_plate": reaction.name,
    }

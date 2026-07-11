"""Thin, real PyLabRobot helpers shared by the Build and Test protocols.

Everything here issues genuine PyLabRobot calls. By default it targets the
chatterbox simulator (prints the deck choreography, no hardware). Swapping to a
real liquid handler is a one-line backend change:

    from pylabrobot.liquid_handling import STARBackend
    lh = LiquidHandler(backend=STARBackend(), deck=STARLetDeck())

and likewise ExperimentalODTCBackend / HamiltonHeaterShakerBackend /
Cytation5Backend for the peripherals. The protocol logic does not change.
"""

from __future__ import annotations

import math

from pylabrobot.liquid_handling import LiquidHandler
from pylabrobot.resources import (STARLetDeck, TIP_CAR_480_A00, PLT_CAR_L5AC_A00,
    hamilton_96_tiprack_1000uL_filter)

from . import params
from .hardware import liquid_handler_backend

_ROWS = "ABCDEFGH"


def n_columns(n_variants: int) -> int:
    return math.ceil(n_variants / params.NUM_CHANNELS)


def col_end_row(count: int) -> str:
    return _ROWS[count - 1]


def make_liquid_handler(n_tip_racks: int = 1, hardware: bool = False) -> tuple[LiquidHandler, list]:
    """A STARlet with a tip carrier. Returns (lh, tip_racks)."""
    lh = LiquidHandler(backend=liquid_handler_backend(hardware, params.NUM_CHANNELS),
                       deck=STARLetDeck())
    tip_car = TIP_CAR_480_A00(name="tip_carrier")
    racks = []
    for i in range(n_tip_racks):
        rack = hamilton_96_tiprack_1000uL_filter(name=f"tips_{i}")
        tip_car[i] = rack
        racks.append(rack)
    lh.deck.assign_child_resource(tip_car, rails=params.RAILS_TIPS)
    return lh, racks


class TipStream:
    """Hand out fresh tip columns across one or more racks."""

    def __init__(self, racks: list):
        self.racks = racks
        self.col = 0

    def next_range(self, count: int) -> tuple:
        rack = self.racks[self.col // 12]
        c = self.col % 12
        self.col += 1
        return rack, f"A{c + 1}:{col_end_row(count)}{c + 1}"


async def distribute_from_trough(lh, trough, dest_plate, n_variants, vol, tips: TipStream):
    """Multi-column distribute of one reagent from a trough to dest wells."""
    for col in range(n_columns(n_variants)):
        count = min(params.NUM_CHANNELS, n_variants - col * params.NUM_CHANNELS)
        rack, tip_range = tips.next_range(count)
        await lh.pick_up_tips(rack[tip_range])
        await lh.aspirate([trough] * count, vols=[vol] * count)
        await lh.dispense(dest_plate[f"A{col + 1}:{col_end_row(count)}{col + 1}"], vols=[vol] * count)
        await lh.drop_tips(rack[tip_range])


async def transfer_plate_to_plate(lh, src_plate, dest_plate, n_variants, vol, tips: TipStream,
                                  mix_cycles: int = 0):
    """One-to-one column transfer between two plates (fresh tips per column)."""
    for col in range(n_columns(n_variants)):
        count = min(params.NUM_CHANNELS, n_variants - col * params.NUM_CHANNELS)
        rack, tip_range = tips.next_range(count)
        rng = f"A{col + 1}:{col_end_row(count)}{col + 1}"
        await lh.pick_up_tips(rack[tip_range])
        await lh.aspirate(src_plate[rng], vols=[vol] * count)
        await lh.dispense(dest_plate[rng], vols=[vol] * count)
        if mix_cycles:
            for _ in range(mix_cycles):
                await lh.aspirate(dest_plate[rng], vols=[vol] * count)
                await lh.dispense(dest_plate[rng], vols=[vol] * count)
        await lh.drop_tips(rack[tip_range])

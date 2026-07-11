"""Instrument backend selection: simulator (default) or real hardware.

Switching a run to the bench is a single flag; the protocol logic in build_star /
test_star / readout does not change. The hardware constructors follow the
PyLabRobot backend signatures:

    STARBackend(device_address=None, serial_number=None, ...)
    ExperimentalODTCBackend(ip, client_ip=None)
    HamiltonHeaterShakerBackend(index, interface)      # HHS on the STAR TCC bus
    Cytation5Backend(...)

Connection parameters live in params.py marked TODO(hardware). Nothing in this
repo is executed against real instruments; --hardware only selects the backends.
"""

from __future__ import annotations

from . import params


def liquid_handler_backend(hardware: bool, num_channels: int):
    if hardware:
        from pylabrobot.liquid_handling import STARBackend
        # TODO(hardware): set device_address or serial_number for the STARlet.
        return STARBackend(serial_number=params.STAR_SERIAL)
    from pylabrobot.liquid_handling import LiquidHandlerChatterboxBackend
    return LiquidHandlerChatterboxBackend(num_channels=num_channels)


def thermocycler_backend(hardware: bool):
    if hardware:
        from pylabrobot.thermocycling.inheco import ExperimentalODTCBackend
        # ODTC over SiLA; on starpi it sits on eth1 link-local (see di-omics notes).
        return ExperimentalODTCBackend(ip=params.ODTC_IP)
    from pylabrobot.thermocycling import ThermocyclerChatterboxBackend
    return ThermocyclerChatterboxBackend()


def heater_shaker_backend(hardware: bool, star_backend=None):
    if hardware:
        from pylabrobot.heating_shaking import HamiltonHeaterShakerBackend
        # TODO(hardware): the HHS hangs off the STAR TCC bus; the interface is the
        # STAR backend. Verify HamiltonHeaterShakerInterface wiring on starpi.
        return HamiltonHeaterShakerBackend(index=params.HHS_INDEX, interface=star_backend)
    from pylabrobot.heating_shaking import HeaterShakerChatterboxBackend
    return HeaterShakerChatterboxBackend()


def plate_reader_backend(hardware: bool):
    if hardware:
        # BioTek Cytation 5 imaging reader. Needs the FTDI extra on the bench:
        #   pip install pylabrobot[ftdi]
        # TODO(hardware): set Cytation connection parameters.
        from pylabrobot.plate_reading import CytationBackend
        return CytationBackend()
    from pylabrobot.plate_reading import PlateReaderChatterboxBackend
    return PlateReaderChatterboxBackend()

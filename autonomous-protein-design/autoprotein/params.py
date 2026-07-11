"""All wet-lab protocol values live here, and every one is a placeholder.

Per lab policy, no reagent volume, temperature, time, or catalog number is
invented and passed off as real. Each constant below is marked TODO(protocol):
it is a plausible default so the choreography runs in simulation, and it MUST be
replaced with a value from the actual, version-controlled bench protocol before
any hardware run. Grep for TODO(protocol) to find everything that needs sourcing.

Units are ASCII: uL (microliters), nM, C (Celsius), s (seconds).
"""

from __future__ import annotations

# ---- deck geometry (STARlet rails; confirm against the physical deck) --------
RAILS_TIPS = 1          # TODO(protocol): tip carrier rail position
RAILS_REAGENTS = 7      # TODO(protocol): reagent carrier rail position
RAILS_PLATES = 13       # TODO(protocol): plate/thermal carrier rail position
NUM_CHANNELS = 8        # STARlet 8-channel head

# ---- BUILD arm: variant assembly + transformation ---------------------------
MASTERMIX_UL = 15.0       # TODO(protocol): assembly/mutagenesis master mix per rxn
TEMPLATE_DNA_UL = 2.0     # TODO(protocol): variant template/oligo per rxn
MIX_CYCLES = 3            # TODO(protocol): pipette-mix cycles after combining
DPNI_UL = 1.0             # TODO(protocol): DpnI (template digest) per rxn
COMPETENT_CELLS_UL = 25.0 # TODO(protocol): competent cells per transformation

# Assembly / mutagenesis thermal profile: (temperature_C, hold_s) per step.
# TODO(protocol): replace with the real cycling program and cycle count.
ASSEMBLY_PROFILE = [
    (98.0, 30.0),   # initial denaturation
    (98.0, 10.0),   # denature
    (60.0, 20.0),   # anneal
    (72.0, 180.0),  # extend
    (72.0, 300.0),  # final extension
    (4.0, 0.0),     # hold
]
ASSEMBLY_CYCLES = 18       # TODO(protocol): number of cycles for the inner 3 steps
LID_TEMP_C = 105.0         # TODO(protocol): heated-lid setpoint
DPNI_INCUBATE_C = 37.0     # TODO(protocol)
DPNI_INCUBATE_S = 3600.0   # TODO(protocol)

# Heat-shock transformation on the heater-shaker.
HEATSHOCK_C = 42.0         # TODO(protocol)
HEATSHOCK_S = 45.0         # TODO(protocol)
RECOVERY_C = 37.0          # TODO(protocol)
RECOVERY_S = 3600.0        # TODO(protocol)

# ---- TEST arm: expression + phenotyping -------------------------------------
MEDIA_UL = 180.0           # TODO(protocol): expression media per well (deep well)
INOCULUM_UL = 5.0          # TODO(protocol): transfer volume to inoculate
INDUCER_UL = 2.0           # TODO(protocol): IPTG / arabinose per well
EXPRESSION_C = 30.0        # TODO(protocol): induction temperature
EXPRESSION_S = 6 * 3600.0  # TODO(protocol): induction time
SHAKE_RPM = 1000.0         # TODO(protocol): heater-shaker speed
READ_ALIQUOT_UL = 100.0    # TODO(protocol): transfer to the read plate

# ---- READOUT: Cytation 5 (GFP channel) --------------------------------------
GFP_EX_NM = 485.0          # TODO(protocol): GFP excitation
GFP_EM_NM = 528.0          # TODO(protocol): GFP emission
OD_WAVELENGTH_NM = 600.0   # TODO(protocol): optical density wavelength
READ_FOCAL_HEIGHT_MM = 7.0 # TODO(protocol): reader focal height

# ---- acceptance criteria (the QC gate; brightness normalized to WT = 1.0) ----
ACCEPT_REL_BRIGHTNESS = 1.5   # promote a variant only if >= 1.5x wild-type
ACCEPT_MIN_OD = 0.2           # and it actually grew

# ---- hardware connection (only used with --hardware; all TODO(hardware)) ------
STAR_SERIAL = None            # TODO(hardware): STARlet serial_number (or device_address)
ODTC_IP = "169.254.1.2"       # TODO(hardware): ODTC SiLA IP (Pi eth1 link-local range)
HHS_INDEX = 1                 # TODO(hardware): HHS node index on the STAR TCC bus

"""Demo protein: wild-type Avicularia victoria GFP (avGFP, UniProt P42212).

Benign, iconic, and heavily studied by deep mutational scanning, so the model's
predictions can be sanity-checked against known biology (the chromophore-forming
Ser-Tyr-Gly tripeptide should read as strongly conserved).
"""

AVGFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT"
    "GKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFF"
    "KDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNV"
    "YIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHY"
    "LSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

# Known chromophore tripeptide (Ser65-Tyr66-Gly67 in avGFP numbering).
CHROMOPHORE_MOTIF = "SYG"

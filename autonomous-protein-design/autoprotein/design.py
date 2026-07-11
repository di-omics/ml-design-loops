"""The Design arm: propose new single-substitution candidates to explore.

Two interchangeable designers behind one interface:

  - SurrogateDesigner: light, dependency-free, samples tolerated positions at
    random. Keeps the repo sim-first and CPU-only by default.
  - ESMDesigner: uses a pretrained ESM-2 protein language model to bias
    exploration toward substitutions the model finds natural (foundation-model
    priors). Requires torch + transformers (optional extra).

Exploitation - stacking the mutations that measured well - is handled by the
learner and the orchestrator. The designer only proposes exploration moves.
"""

from __future__ import annotations

import numpy as np

from .sequences import AVGFP, CHROMOPHORE_IDX

AA = "ACDEFGHIKLMNPQRSTVWY"


def apply_mutations(wt: str, muts: list[tuple[int, str]]) -> str:
    seq = list(wt)
    for i, aa in muts:
        seq[i] = aa
    return "".join(seq)


def mutation_label(wt: str, i: int, aa: str) -> str:
    return f"{wt[i]}{i + 1}{aa}"


class SurrogateDesigner:
    name = "surrogate"

    def __init__(self, wt: str = AVGFP):
        self.wt = wt
        self.avoid = set(CHROMOPHORE_IDX)

    def propose_singletons(self, n: int, rng: np.random.Generator,
                           avoid_positions: set[int] | None = None) -> list[tuple[int, str]]:
        avoid = self.avoid | (avoid_positions or set())
        positions = [i for i in range(len(self.wt)) if i not in avoid]
        out = []
        for _ in range(n):
            i = int(rng.choice(positions))
            aa = self.wt[i]
            while aa == self.wt[i]:
                aa = AA[int(rng.integers(20))]
            out.append((i, aa))
        return out


class ESMDesigner:
    name = "esm2"

    def __init__(self, wt: str = AVGFP, model_name: str = "facebook/esm2_t12_35M_UR50D"):
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as hfl
        hfl.set_verbosity_error()
        self.wt = wt
        self.avoid = set(CHROMOPHORE_IDX)
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        aa_ids = [self.tok.convert_tokens_to_ids(a) for a in AA]
        with torch.no_grad():
            enc = self.tok(wt, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits[0]
            lp = torch.log_softmax(logits, dim=-1)[1:1 + len(wt)][:, aa_ids]
        self.log_probs = lp.cpu().numpy()                       # [L, 20]
        # Position tolerance: low WT-residue probability => more substitutable.
        wt_lp = np.array([self.log_probs[i, AA.index(wt[i])] for i in range(len(wt))])
        self.tolerance = -wt_lp                                  # higher = more tolerant

    def propose_singletons(self, n: int, rng: np.random.Generator,
                           avoid_positions: set[int] | None = None) -> list[tuple[int, str]]:
        avoid = self.avoid | (avoid_positions or set())
        positions = np.array([i for i in range(len(self.wt)) if i not in avoid])
        # Blend ESM tolerance with uniform exploration: pure tolerance-weighting
        # over-focuses on surface positions and skips conserved-but-beneficial
        # sites (e.g. the chromophore neighbours F64/S65), so mix in uniform.
        w_esm = np.exp(self.tolerance[positions])
        w_esm /= w_esm.sum()
        w = 0.5 * w_esm + 0.5 * (np.ones(len(positions)) / len(positions))
        w /= w.sum()
        out = []
        for _ in range(n):
            i = int(rng.choice(positions, p=w))
            probs = np.exp(self.log_probs[i])
            probs[AA.index(self.wt[i])] = 0.0                   # force a real change
            probs /= probs.sum()
            aa = AA[int(rng.choice(20, p=probs))]
            out.append((i, aa))
        return out


def get_designer(prefer_esm: bool = True, wt: str = AVGFP):
    """Return an ESM designer if torch is importable, else the surrogate."""
    if prefer_esm:
        try:
            return ESMDesigner(wt=wt)
        except Exception as e:  # torch/transformers missing or model unavailable
            print(f"  [design] ESM unavailable ({e.__class__.__name__}); "
                  f"using surrogate designer")
    return SurrogateDesigner(wt=wt)

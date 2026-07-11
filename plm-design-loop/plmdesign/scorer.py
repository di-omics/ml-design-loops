"""ESM-2 protein language model as a zero-shot fitness oracle.

Wraps a pretrained ESM-2 masked language model and exposes the two scoring
methods from the ESM variant-effect work:

  - wt-marginal:     one forward pass on the wild type; score a mutation as the
                     model log-odds of the mutant vs the wild-type residue in the
                     wild-type context. Cheap; a whole deep-mutational-scan in a
                     single pass.
  - masked-marginal: mask each position and re-read the distribution there. One
                     forward pass per residue; more faithful, still seconds on an
                     M1 GPU for a small protein.

It also exposes a sequence pseudo-log-likelihood, the model's "naturalness"
score, which the design loop climbs.

Runs on Apple MPS by default (this is why it stays light on a Mac), with a CPU
fallback if an op is unsupported.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
DEFAULT_MODEL = os.environ.get("ESM_MODEL", "facebook/esm2_t12_35M_UR50D")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ESMScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        self.aa_ids = torch.tensor(
            [self.tok.convert_tokens_to_ids(a) for a in AA], device=self.device
        )

    @torch.no_grad()
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        try:
            return self.model(input_ids=input_ids.to(self.device)).logits
        except Exception:  # pragma: no cover - MPS op gap safety net
            self.device = "cpu"
            self.model = self.model.to("cpu")
            self.aa_ids = self.aa_ids.to("cpu")
            return self.model(input_ids=input_ids.to("cpu")).logits

    @torch.no_grad()
    def log_probs_per_pos(self, seq: str) -> np.ndarray:
        """[L, 20] log-softmax over canonical residues at each position, WT context."""
        enc = self.tok(seq, return_tensors="pt")
        logits = self._forward(enc["input_ids"])[0]              # [L+2, vocab]
        lp = torch.log_softmax(logits, dim=-1)
        lp_res = lp[1 : 1 + len(seq)][:, self.aa_ids]            # drop cls/eos
        return lp_res.cpu().numpy()

    def pll_and_lp(self, seq: str) -> tuple[float, np.ndarray]:
        """Pseudo-log-likelihood (sum of WT-context log-probs) plus the matrix."""
        lp = self.log_probs_per_pos(seq)
        pll = float(sum(lp[i, AA_INDEX[seq[i]]] for i in range(len(seq))))
        return pll, lp

    def wt_marginal_scan(self, seq: str) -> np.ndarray:
        """[L, 20] mutation log-odds vs wild type, one forward pass."""
        lp = self.log_probs_per_pos(seq)
        wt = np.array([lp[i, AA_INDEX[seq[i]]] for i in range(len(seq))])
        return lp - wt[:, None]

    @torch.no_grad()
    def masked_marginal_scan(self, seq: str, batch_size: int = 32) -> np.ndarray:
        """[L, 20] mutation log-odds vs wild type, masking each position in turn."""
        enc = self.tok(seq, return_tensors="pt")
        base = enc["input_ids"][0]                               # [L+2]
        mask_id = self.tok.mask_token_id
        L = len(seq)
        scores = np.zeros((L, 20), dtype=np.float32)
        for start in range(0, L, batch_size):
            chunk = list(range(start, min(start + batch_size, L)))
            batch = base.unsqueeze(0).repeat(len(chunk), 1).clone()
            for r, pos in enumerate(chunk):
                batch[r, pos + 1] = mask_id                     # +1 for cls offset
            logits = self._forward(batch)
            lp = torch.log_softmax(logits, dim=-1)
            for r, pos in enumerate(chunk):
                row = lp[r, pos + 1][self.aa_ids].cpu().numpy() # [20]
                scores[pos] = row - row[AA_INDEX[seq[pos]]]
        return scores

"""The agent: run Design -> Build -> Test -> Learn rounds end to end.

Each round the agent designs a batch (exploit the mutations that measured well,
explore new ones the model finds plausible), runs the STAR Build and Test
protocols in simulation, reads brightness on the (simulated) Cytation 5, refits
the per-mutation model, and gates promotions. Wild type is included every round
as an on-plate control.

The Build/Test/Read choreography prints a lot; it is captured per round so the
loop stays readable, and the first round's full trace is returned for the log.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from .build_star import run_build
from .design import AA, apply_mutations, get_designer
from .learn import LearnedModel, promotion_gate, variant_mutations
from .oracle import GFPOracle
from .readout import Cytation5Readout
from .sequences import CHROMOPHORE_IDX
from .test_star import run_test


def _dedup_keep_order(seqs: list[str]) -> list[str]:
    seen, out = set(), []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _tested_pairs(model):
    return {m for muts in model.obs_muts for m in muts}


def _tested_positions(model):
    return {m[0] for muts in model.obs_muts for m in muts}


def _position_best_effect(model):
    d = {}
    for (i, _a), e in model.effects.items():
        d[i] = max(d.get(i, -9.0), e)
    return d


def _aa_order_at(designer, pos, wt, exclude):
    """Residues to try at a position, best-first if an ESM prior is available."""
    if hasattr(designer, "log_probs"):
        order = sorted(range(20), key=lambda j: -designer.log_probs[pos, j])
        aas = [AA[j] for j in order]
    else:
        aas = list(AA)
    return [a for a in aas if a != wt[pos] and a not in exclude]


def _design_batch(wt, model, designer, rng, batch_size, round_idx, incumbent):
    # Round 0: broad single-mutant scan over distinct positions.
    if round_idx == 0:
        pool = designer.propose_singletons(batch_size * 4, rng, set(CHROMOPHORE_IDX))
        seen, singles = set(), []
        for (p, a) in pool:
            if p not in seen:
                seen.add(p)
                singles.append((p, a))
            if len(singles) >= batch_size - 1:
                break
        batch = [wt] + [apply_mutations(wt, [m]) for m in singles]
        return _dedup_keep_order(batch)[:batch_size]

    tested_pos = _tested_positions(model)
    tested_pairs = _tested_pairs(model)
    top = model.top_mutations(8)
    cands = [wt]

    # Recombination: stack the best distinct-position mutations found so far.
    for k in range(2, min(len(top), 8) + 1):
        cands.append(apply_mutations(wt, top[:k]))

    # Add the best not-yet-present mutation to the incumbent.
    if incumbent is not None:
        cands.append(incumbent)
        have = set(variant_mutations(wt, incumbent))
        added = 0
        for m in top:
            if m not in have:
                cands.append(apply_mutations(incumbent, [m]))
                added += 1
                if added >= 2:
                    break

    # Saturation: try more residues at the most promising positions.
    posbest = _position_best_effect(model)
    promising = sorted((i for i, e in posbest.items() if e > 0), key=lambda i: -posbest[i])[:2]
    for p in promising:
        exclude = {a for (i, a) in tested_pairs if i == p}
        for a in _aa_order_at(designer, p, wt, exclude)[:3]:
            cands.append(apply_mutations(wt, [(p, a)]))

    # Exploration: single mutants at positions not yet tested.
    n_expl = max(2, batch_size - len(cands))
    singles = designer.propose_singletons(n_expl, rng, set(CHROMOPHORE_IDX) | tested_pos)
    cands += [apply_mutations(wt, [m]) for m in singles]

    batch = _dedup_keep_order(cands)
    if wt not in batch:
        batch = [wt] + batch
    return batch[:batch_size]


async def run_dbtl(rounds: int = 6, batch_size: int = 24, seed: int = 0,
                   prefer_esm: bool = True, hardware: bool = False):
    wt = GFPOracle().wt
    rng = np.random.default_rng(seed)
    oracle = GFPOracle(seed=seed)
    designer = get_designer(prefer_esm=prefer_esm, wt=wt)
    model = LearnedModel(wt)
    reader = Cytation5Readout(oracle=None if hardware else oracle,
                              mode="hardware" if hardware else "sim")

    history = []
    promoted: dict[str, float] = {}
    incumbent, incumbent_rfu = None, -np.inf
    first_trace = ""

    for r in range(rounds):
        variants = _design_batch(wt, model, designer, rng, batch_size, r, incumbent)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_summary = await run_build(variants, hardware=hardware)
            read_plate, test_summary = await run_test(variants, hardware=hardware)
            measurement = await reader.read(variants, read_plate)
        if r == 0:
            first_trace = buf.getvalue()

        rfu, od = measurement["rfu"], measurement["od600"]
        model.observe(variants, rfu)
        model.fit()
        decisions = promotion_gate(rfu, od)

        for s, rr, dec in zip(variants, rfu, decisions):
            if dec == "accept" and rr > promoted.get(s, 0.0):
                promoted[s] = float(rr)
            if rr > incumbent_rfu and s != wt:
                incumbent, incumbent_rfu = s, float(rr)

        wt_rfu = float(rfu[variants.index(wt)]) if wt in variants else 1.0
        history.append({
            "round": r,
            "designer": designer.name,
            "n_variants": len(variants),
            "best_rfu": float(rfu.max()),
            "cum_best_rfu": float(max(incumbent_rfu, 1.0)),
            "mean_rfu": float(rfu.mean()),
            "wt_rfu": wt_rfu,
            "n_accept": int((decisions == "accept").sum()),
            "n_escalate": int((decisions == "escalate").sum()),
            "best_variant_mutations": [f"{wt[i]}{i + 1}{a}"
                                       for i, a in variant_mutations(wt, variants[int(rfu.argmax())])],
            "build_summary": build_summary,
            "test_summary": test_summary,
        })

    return {
        "wt": wt,
        "history": history,
        "model": model,
        "oracle": oracle,
        "promoted": promoted,
        "incumbent": incumbent,
        "incumbent_rfu": incumbent_rfu,
        "designer": designer.name,
        "first_round_trace": first_trace,
    }

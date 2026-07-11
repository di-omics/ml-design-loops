"""Run the full agentic DBTL loop, sim-first.

    python run_dbtl.py                 # ESM design arm if torch present, else surrogate
    python run_dbtl.py --no-esm        # force the light surrogate designer

Writes figures/, metrics.json, and run_log.txt (the first round's real STARlet +
ODTC + Heater-Shaker + Cytation 5 choreography).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vizstyle
from autoprotein.dbtl import run_dbtl
from autoprotein.learn import variant_mutations
from autoprotein.oracle import CHROMOPHORE_IDX, _CORE_IDX

vizstyle.apply()
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
GREEN, BLUE, PURPLE, INK, NEUTRAL = (vizstyle.GREEN, vizstyle.BLUE, vizstyle.PURPLE,
                                     vizstyle.INK, vizstyle.NEUTRAL)


def fig_improvement(history):
    r = [h["round"] + 1 for h in history]
    cum = [h["cum_best_rfu"] for h in history]
    best = [h["best_rfu"] for h in history]
    mean = [h["mean_rfu"] for h in history]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(r, cum, "-o", color=GREEN, lw=2.6, label="best variant so far (kept)")
    ax.plot(r, best, "--o", color=BLUE, lw=1.4, alpha=0.9, label="best in round")
    ax.plot(r, mean, "-o", color=PURPLE, lw=1.6, label="batch mean")
    ax.axhline(1.0, color=INK, ls=":", lw=1)
    ax.text(r[-1], 1.02, " wild-type GFP", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("DBTL round (each = a full Build + Test + Learn cycle on the STAR)")
    ax.set_ylabel("GFP brightness, fold over wild type")
    ax.set_title("Closed-loop design brightens GFP over rounds", fontsize=12)
    ax.set_xticks(r)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "01_dbtl_improvement.png", dpi=150)
    plt.close(fig)


def fig_recovery(model, oracle, wt):
    effects = sorted(model.effects.items(), key=lambda t: t[1], reverse=True)
    top = effects[:8] + effects[-4:]
    labels, vals, colors = [], [], []
    for (i, aa), e in top:
        labels.append(f"{wt[i]}{i + 1}{aa}")
        vals.append(e)
        if (i, aa) in oracle.brighteners:
            colors.append(GREEN)         # true brightener, correctly found
        elif i in CHROMOPHORE_IDX or i in _CORE_IDX:
            colors.append(PURPLE)        # chromophore/core, correctly penalized
        else:
            colors.append(BLUE)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("learned effect on log-brightness")
    ax.set_title("The loop learns real GFP biology: recovers true brighteners "
                 "(green), penalizes chromophore/core (purple)", fontsize=10.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="true brightener (recovered)"),
                       Patch(color=PURPLE, label="chromophore / structural core"),
                       Patch(color=BLUE, label="other")],
              fontsize=8, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "02_mutation_recovery.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--hardware", action="store_true",
                    help="drive real STAR/ODTC/HHS/Cytation5 instead of the simulator")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()

    result = asyncio.run(run_dbtl(rounds=args.rounds, batch_size=args.batch, seed=0,
                                  prefer_esm=not args.no_esm, hardware=args.hardware))
    wt = result["wt"]
    history, model, oracle = result["history"], result["model"], result["oracle"]

    fig_improvement(history)
    fig_recovery(model, oracle, wt)

    (Path(__file__).parent / "run_log.txt").write_text(result["first_round_trace"])

    def lab(i, aa):
        return f"{wt[i]}{i + 1}{aa}"

    best_muts_pairs = variant_mutations(wt, result["incumbent"]) if result["incumbent"] else []
    inc_muts = [lab(i, a) for i, a in best_muts_pairs]
    brighteners = set(oracle.brighteners)                 # primary targets (canonical + synthetic)
    canonical = set(oracle.canonical)
    top10 = model.top_mutations(10)
    precision10 = round(sum((m in brighteners) for m in top10) / max(len(top10), 1), 2)
    canonical_recovered = [lab(i, a) for (i, a) in top10 if (i, a) in canonical]
    best_true_brighteners = [lab(i, a) for (i, a) in best_muts_pairs if (i, a) in brighteners]

    metrics = {
        "designer": result["designer"],
        "rounds": args.rounds,
        "batch_size": args.batch,
        "best_brightness_fold_over_wt": round(float(result["incumbent_rfu"]), 2),
        "incumbent_mutations": inc_muts,
        "incumbent_true_brighteners": best_true_brighteners,
        "n_promoted_variants": len(result["promoted"]),
        "top10_precision_are_true_brighteners": precision10,
        "canonical_brighteners_recovered": sorted(canonical_recovered),
        "n_ground_truth_brighteners": len(brighteners),
        "per_round": [{k: h[k] for k in ("round", "best_rfu", "cum_best_rfu",
                                         "mean_rfu", "n_accept")} for h in history],
    }
    (Path(__file__).parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n==================  DBTL HEADLINE  ==================")
    print(f"Design arm: {result['designer']}   |   {args.rounds} rounds x {args.batch} variants")
    print(f"Best variant: {result['incumbent_rfu']:.2f}x wild-type GFP brightness "
          f"from {len(inc_muts)} mutations ({len(best_true_brighteners)} are true brighteners)")
    print(f"  mutations: {', '.join(inc_muts)}")
    print(f"Learned model precision: {precision10 * 100:.0f}% of its top-10 mutations "
          f"are real brighteners")
    print(f"Canonical GFP brighteners recovered: "
          f"{', '.join(canonical_recovered) if canonical_recovered else 'none this seed'}")
    print(f"Promoted variants passing the QC gate: {len(result['promoted'])}")
    print("====================================================")
    print(f"\nFigures -> {OUT}/   Metrics -> metrics.json   Protocol trace -> run_log.txt")


if __name__ == "__main__":
    main()

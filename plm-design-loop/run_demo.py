"""End-to-end demo: mutational scan + closed-loop design on avGFP.

    python run_demo.py            # esm2_t12_35M by default
    ESM_MODEL=facebook/esm2_t30_150M_UR50D python run_demo.py

Writes figures/, a designed_variant.fasta, and metrics.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plmdesign.design import evolve
from plmdesign.scorer import AA, ESMScorer
from plmdesign.sequences import AVGFP, CHROMOPHORE_MOTIF

import vizstyle
vizstyle.apply()
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
SAGE, INK, WARN, BLUE, PINK = (vizstyle.GREEN, vizstyle.INK, vizstyle.PURPLE,
                               vizstyle.BLUE, vizstyle.PINK)


def _frame(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(INK)
        s.set_linewidth(0.8)


def fig_heatmap(seq, scores):
    L = len(seq)
    fig, ax = plt.subplots(figsize=(min(26, 2 + L * 0.09), 5.2))
    vmax = np.percentile(np.abs(scores), 99)
    im = ax.imshow(scores.T, aspect="auto", cmap=vizstyle.DIVERGING, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    _frame(ax)
    ax.set_yticks(range(20))
    ax.set_yticklabels(list(AA), fontsize=7)
    ax.set_xlabel("residue position")
    ax.set_ylabel("substituted amino acid")
    ax.set_title("ESM-2 single-shot mutational scan of avGFP  "
                 "(blue = model expects the mutation to be tolerated/favorable, "
                 "pink = deleterious)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("log-odds vs wild type")
    fig.tight_layout()
    fig.savefig(OUT / "01_mutational_scan.png", dpi=150)
    plt.close(fig)


def fig_zoom(seq, scores, lo, hi):
    sub = scores[lo:hi]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    vmax = np.percentile(np.abs(scores), 99)
    ax.imshow(sub.T, aspect="auto", cmap=vizstyle.DIVERGING, vmin=-vmax, vmax=vmax)
    _frame(ax)
    ax.set_yticks(range(20))
    ax.set_yticklabels(list(AA), fontsize=8)
    ax.set_xticks(range(hi - lo))
    ax.set_xticklabels([f"{seq[i]}{i + 1}" for i in range(lo, hi)], rotation=90, fontsize=6)
    # mark the wild-type residue in each column
    for c, i in enumerate(range(lo, hi)):
        ax.add_patch(plt.Rectangle((c - 0.5, AA.index(seq[i]) - 0.5), 1, 1,
                                   fill=False, edgecolor=INK, lw=1.2))
    motif = seq.find(CHROMOPHORE_MOTIF, 60, 75)
    if lo <= motif < hi:
        ax.annotate("chromophore\nS65-Y66-G67", xy=(motif - lo + 1, 19.5),
                    xytext=(motif - lo + 1, 22.5), ha="center", color=WARN, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=WARN))
    ax.set_ylabel("substituted amino acid")
    ax.set_title("Chromophore region (black box = wild type): model conserves "
                 "structural Gly but rates Y66 as substitutable\n- exactly where "
                 "nature makes GFP color variants (Y66H/Y66W). Conservation is "
                 "family constraint, not single-protein function.", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "02_chromophore_zoom.png", dpi=150)
    plt.close(fig)


def fig_trajectory(result):
    steps = [t["step"] for t in result.trajectory]
    pll = [t["pll"] for t in result.trajectory]
    nmut = [t["n_mut"] for t in result.trajectory]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(steps, pll, color=SAGE, lw=2)
    ax.axhline(result.wt_pll, color=INK, ls=":", lw=1)
    ax.text(steps[-1], result.wt_pll, " wild-type naturalness", va="bottom", ha="right",
            fontsize=8, color=INK)
    ax.set_xlabel("design step (propose / score / accept)")
    ax.set_ylabel("ESM-2 pseudo-log-likelihood (naturalness)")
    ax.set_title(f"Closed-loop design climbs naturalness with {len(result.mutations)} "
                 f"mutations", fontsize=11)
    ax2 = ax.twinx()
    ax2.plot(steps, nmut, color=WARN, lw=1, alpha=0.6)
    ax2.set_ylabel("mutations from wild type", color=WARN)
    ax2.tick_params(axis="y", colors=WARN)
    ax.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "03_design_trajectory.png", dpi=150)
    plt.close(fig)


def main():
    seq = AVGFP
    print(f"Loading ESM-2 ...")
    t0 = time.time()
    scorer = ESMScorer()
    print(f"  model {scorer.model_name} on device={scorer.device} "
          f"({time.time() - t0:.1f}s)")

    print(f"Masked-marginal mutational scan of {len(seq)} residues "
          f"({len(seq) * 19} point mutants) ...")
    t1 = time.time()
    scores = scorer.masked_marginal_scan(seq)
    scan_s = time.time() - t1
    np.save(OUT.parent / "mutational_scan.npy", scores)
    fig_heatmap(seq, scores)

    # Conservation: positions whose mutations are most deleterious on average.
    mean_effect = scores.sum(axis=1) / 19.0  # WT column is 0, so /19 non-WT
    rank = {int(i): r for r, i in enumerate(np.argsort(mean_effect))}
    conserved = np.argsort(mean_effect)[:8]
    n_gly_conserved = sum(seq[i] == "G" for i in conserved)
    motif_start = seq.find(CHROMOPHORE_MOTIF, 60, 75)
    y66_pct = 100 * rank[motif_start + 1] / len(seq)   # Y66 is motif index +1
    g67_pct = 100 * rank[motif_start + 2] / len(seq)   # G67 is motif index +2
    fig_zoom(seq, scores, max(0, motif_start - 10), motif_start + 14)

    print("Closed-loop directed evolution (mutation budget 18, chromophore fixed) ...")
    fixed = frozenset(range(motif_start, motif_start + 3)) if motif_start >= 0 else frozenset()
    t2 = time.time()
    result = evolve(scorer, seq, n_steps=150, max_mut=18, fixed=fixed, seed=1)
    design_s = time.time() - t2
    fig_trajectory(result)

    (Path(__file__).parent / "designed_variant.fasta").write_text(
        f">avGFP_designed n_mut={len(result.mutations)} "
        f"pll={result.designed_pll:.2f}\n{result.designed}\n"
    )
    metrics = {
        "model": scorer.model_name,
        "device": scorer.device,
        "protein": "avGFP",
        "length": len(seq),
        "point_mutants_scored": len(seq) * 19,
        "scan_seconds": round(scan_s, 1),
        "wt_pseudo_log_likelihood": round(result.wt_pll, 2),
        "designed_pseudo_log_likelihood": round(result.designed_pll, 2),
        "n_mutations": len(result.mutations),
        "mutations": result.mutations,
        "design_seconds": round(design_s, 1),
        "model_predicted_conserved_positions": [f"{seq[i]}{i + 1}" for i in sorted(conserved)],
        "conserved_positions_that_are_gly": int(n_gly_conserved),
        "chromophore_SYG_at": motif_start + 1,
        "Y66_conservation_percentile": round(y66_pct, 0),
        "G67_conservation_percentile": round(g67_pct, 0),
    }
    (Path(__file__).parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n==================  HEADLINE  ==================")
    print(f"Scanned {len(seq) * 19} point mutants in {scan_s:.1f}s on {scorer.device}")
    print(f"Most-conserved positions are structural core "
          f"({n_gly_conserved}/8 are glycine): "
          f"{', '.join(metrics['model_predicted_conserved_positions'])}")
    print(f"Chromophore S65-Y66-G67: model keeps G67 (conserved, {g67_pct:.0f}th pct) "
          f"but rates Y66 highly substitutable ({y66_pct:.0f}th pct) - the GFP "
          f"color-variant site (Y66H->BFP, Y66W->CFP). Conservation = family "
          f"constraint, not single-protein function.")
    print(f"Design: {len(result.mutations)} mutations lifted naturalness "
          f"{result.wt_pll:.1f} -> {result.designed_pll:.1f} (chromophore held fixed)")
    print(f"Mutations: {', '.join(result.mutations)}")
    print("================================================")
    print(f"\nFigures -> {OUT}/   Variant -> designed_variant.fasta   Metrics -> metrics.json")


if __name__ == "__main__":
    main()

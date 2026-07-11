"""End-to-end demo: closed-loop optimization + conformal QC gate.

Runs the ParEGO agent against a random-search baseline over many seeds, then
exercises the conformal QC gate on a held-out protocol set. Saves figures and a
machine-readable metrics.json, and prints the interview headline numbers.

    python run_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from labworld import assay
from labworld.optimize import objectives_to_unit, parego, random_search
from labworld.qc_gate import ACCEPT, ESCALATE, REJECT, ConformalQCGate

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
SEEDS = list(range(12))
N_ITER = 30
N_INIT = 6

# House style: Manrope + pastel blue/green/purple, thin black lines.
import vizstyle
vizstyle.apply()
SAGE, INK, GREY, WARN, BLUE = (vizstyle.GREEN, vizstyle.INK, vizstyle.NEUTRAL,
                               vizstyle.PURPLE, vizstyle.BLUE)


def run_optimization():
    bo_runs = [parego(N_ITER, s, n_init=N_INIT) for s in SEEDS]
    rand_runs = [random_search(N_ITER, s, n_init=N_INIT) for s in SEEDS]
    return bo_runs, rand_runs


def fig_convergence(bo_runs, rand_runs):
    bo = np.stack([r.hv_curve for r in bo_runs])
    rd = np.stack([r.hv_curve for r in rand_runs])
    x = np.arange(1, bo.shape[1] + 1)

    def band(ax, data, color, label):
        m = data.mean(0)
        se = data.std(0) / np.sqrt(data.shape[0])
        ax.plot(x, m, color=color, lw=2.2, label=label)
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    band(ax, bo, SAGE, "Closed-loop agent (ParEGO BO)")
    band(ax, rd, GREY, "Random search (baseline)")
    ax.axvline(N_INIT, color=INK, ls=":", lw=1, alpha=0.5)
    ax.text(N_INIT + 0.2, ax.get_ylim()[0], " shared random init", fontsize=8, color=INK, alpha=0.6)
    ax.set_xlabel("experiments run (each = one bench/robot run)")
    ax.set_ylabel("Pareto hypervolume (higher = better trade-off set)")
    ax.set_title("The world model finds better protocols with fewer experiments", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "01_convergence.png", dpi=150)
    plt.close(fig)
    return float(bo[:, -1].mean()), float(rd[:, -1].mean())


def fig_first_pass(bo_runs, rand_runs):
    # Seeds that never clear QC within budget are right-censored at the budget,
    # so the comparison reflects difficulty instead of silently dropping them.
    budget = N_INIT + N_ITER
    bo = np.array([r.first_pass if r.first_pass > 0 else budget for r in bo_runs], dtype=float)
    rd = np.array([r.first_pass if r.first_pass > 0 else budget for r in rand_runs], dtype=float)
    bo_med, rd_med = np.median(bo), np.median(rd)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    parts = ax.boxplot([bo, rd], tick_labels=["Closed-loop\nagent", "Random\nsearch"],
                       patch_artist=True, widths=0.55, medianprops=dict(color=INK, lw=2))
    for patch, c in zip(parts["boxes"], [SAGE, GREY]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.8)
    ax.set_ylabel("experiments until first protocol\nclears all QC acceptance criteria")
    ax.set_title("Time-to-first-release-grade protocol", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "02_first_pass.png", dpi=150)
    plt.close(fig)
    return float(bo_med), float(rd_med)


def fig_pareto(bo_run, rand_run):
    def pts(run):
        t = assay.latent_outcomes(run.X)
        return t["yield_nM"], t["complexity"], t["purity"]

    by, bc, bp = pts(bo_run)
    ry, rc, rp = pts(rand_run)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(ry, rc, c=GREY, s=28, alpha=0.6, label="random search",
               edgecolor=INK, linewidth=0.3)
    sc = ax.scatter(by, bc, c=bp, s=48, cmap=vizstyle.SEQ_GREEN, vmin=0.6, vmax=1.0,
                    edgecolor=INK, linewidth=0.5, label="closed-loop agent")
    ax.axvline(assay.ACCEPTANCE["yield_nM"], color=WARN, ls="--", lw=1)
    ax.axhline(assay.ACCEPTANCE["complexity"], color=WARN, ls="--", lw=1)
    ax.text(assay.ACCEPTANCE["yield_nM"] + 0.4, 0.02, "yield gate", color=WARN, fontsize=8)
    ax.text(0.3, assay.ACCEPTANCE["complexity"] + 0.01, "complexity gate", color=WARN, fontsize=8)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("purity (1 - adapter-dimer)")
    ax.set_xlabel("yield (nM)")
    ax.set_ylabel("library complexity")
    ax.set_title("Protocols explored: agent concentrates on the release-grade corner", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "03_pareto.png", dpi=150)
    plt.close(fig)


def run_qc_gate():
    # Data-poor regime on purpose: a real lab world model is trained on tens of
    # runs, not thousands. That is where uncertainty-awareness earns its keep.
    rng = np.random.default_rng(7)
    X = assay.sample_design(520, rng)
    Y = assay.run_assay(X, rng)
    Xtr, Xcal, Xte = X[:120], X[120:220], X[220:]
    Ytr = {k: Y[k][:120] for k in Y}
    Ycal = {k: Y[k][120:220] for k in Y}
    Yte = {k: Y[k][220:] for k in Y}

    gate = ConformalQCGate(alpha=0.10).fit(Xtr, Ytr, Xcal, Ycal)
    coverage = gate.empirical_coverage(Xte, Yte)
    decision, _ = gate.triage(Xte)

    # Ground truth (noise-free) pass/fail for judging decision correctness.
    true_pass = assay.passes_acceptance(assay.latent_outcomes(Xte))

    auto = decision != ESCALATE
    escalated = decision == ESCALATE
    auto_correct = (
        ((decision == ACCEPT) & true_pass) | ((decision == REJECT) & ~true_pass)
    )
    auto_decided_rate = float(auto.mean())
    auto_error_rate = float((auto & ~auto_correct).sum() / max(auto.sum(), 1))
    escalate_rate = float(escalated.mean())

    # Naive point gate: accept iff every predicted mean clears the threshold; no
    # escalation, no uncertainty. It is forced to call every protocol, including
    # the ones the conformal gate flagged as too close to decide.
    means = gate.model.predict_mean(Xte)
    naive_accept = np.ones(len(Xte), dtype=bool)
    for name in assay.OBJECTIVES:
        naive_accept &= means[name] >= assay.ACCEPTANCE[name]
    naive_wrong = naive_accept != true_pass
    naive_err_overall = float(naive_wrong.mean())
    naive_err_on_auto = float(naive_wrong[auto].sum() / max(auto.sum(), 1))
    naive_err_on_escalated = float(naive_wrong[escalated].sum() / max(escalated.sum(), 1))
    naive_false_accept = float(np.mean(naive_accept & ~true_pass))

    _fig_qc(coverage, decision, auto_decided_rate, auto_error_rate,
            naive_err_on_auto, naive_err_on_escalated)
    return {
        "coverage": coverage,
        "target_coverage": 0.90,
        "auto_decided_rate": auto_decided_rate,
        "auto_error_rate": auto_error_rate,
        "escalate_rate": escalate_rate,
        "naive_gate_error_rate": naive_err_overall,
        "naive_gate_false_accept_rate": naive_false_accept,
        "naive_err_on_conformal_auto_set": naive_err_on_auto,
        "naive_err_on_conformal_escalated_set": naive_err_on_escalated,
    }


def _fig_qc(coverage, decision, auto_rate, auto_err, naive_err_auto, naive_err_esc):
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(12.4, 4.0))

    names = list(coverage.keys())
    a0.bar(range(len(names)), [coverage[n] for n in names], color=SAGE, alpha=0.85)
    a0.axhline(0.90, color=WARN, ls="--", lw=1.5)
    a0.text(len(names) - 1.4, 0.915, "target 90%", color=WARN, fontsize=8)
    a0.set_xticks(range(len(names)))
    a0.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    a0.set_ylim(0, 1.05)
    a0.set_ylabel("empirical interval coverage")
    a0.set_title("Conformal intervals are calibrated", fontsize=10)
    a0.spines[["top", "right"]].set_visible(False)

    counts = [np.mean(decision == d) for d in (ACCEPT, REJECT, ESCALATE)]
    a1.bar(range(3), counts, color=[SAGE, GREY, WARN], alpha=0.85)
    a1.set_xticks(range(3))
    a1.set_xticklabels(["auto\naccept", "auto\nreject", "escalate\nto bench"], fontsize=8)
    a1.set_ylabel("fraction of proposed protocols")
    a1.set_title("Gate triages into decide vs escalate", fontsize=10)
    a1.spines[["top", "right"]].set_visible(False)

    a2.bar([0, 1, 2], [auto_err, naive_err_auto, naive_err_esc],
           color=[SAGE, GREY, WARN], alpha=0.85)
    a2.set_xticks([0, 1, 2])
    a2.set_xticklabels([f"conformal, on its\nauto-decided\n({auto_rate*100:.0f}% of runs)",
                        "naive gate, on the\nsame auto set",
                        "naive gate, on the\nescalated set"], fontsize=7.5)
    a2.set_ylabel("wrong release decisions (rate)")
    a2.set_title("Errors concentrate where the gate escalates", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "04_conformal_qc.png", dpi=150)
    plt.close(fig)


def main():
    print("Running closed-loop optimization over", len(SEEDS), "seeds ...")
    bo_runs, rand_runs = run_optimization()
    bo_hv, rd_hv = fig_convergence(bo_runs, rand_runs)
    bo_fp, rd_fp = fig_first_pass(bo_runs, rand_runs)
    fig_pareto(bo_runs[0], rand_runs[0])

    print("Running conformal QC gate ...")
    qc = run_qc_gate()

    reduction = 100 * (1 - bo_fp / rd_fp) if rd_fp else float("nan")
    metrics = {
        "final_hypervolume_agent": round(bo_hv, 4),
        "final_hypervolume_random": round(rd_hv, 4),
        "median_experiments_to_first_pass_agent": bo_fp,
        "median_experiments_to_first_pass_random": rd_fp,
        "experiment_reduction_pct": round(reduction, 1),
        "qc_gate": {k: (round(v, 4) if isinstance(v, float) else
                        {kk: round(vv, 4) for kk, vv in v.items()})
                    for k, v in qc.items()},
    }
    (Path(__file__).parent / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n==================  HEADLINE NUMBERS  ==================")
    print(f"Pareto hypervolume  agent {bo_hv:.3f}  vs  random {rd_hv:.3f}"
          f"  ( +{100*(bo_hv/rd_hv-1):.0f}% )")
    print(f"Median experiments to first release-grade protocol:")
    print(f"    agent  {bo_fp:.0f}   vs   random {rd_fp:.0f}   ->  {reduction:.0f}% fewer runs")
    print(f"Conformal QC coverage (target 90%): "
          + ", ".join(f"{k} {v*100:.0f}%" for k, v in qc['coverage'].items()))
    print(f"Gate auto-decides {qc['auto_decided_rate']*100:.0f}% of protocols "
          f"at {qc['auto_error_rate']*100:.1f}% error; escalates {qc['escalate_rate']*100:.0f}% to the bench")
    print(f"Naive point gate (no uncertainty) is wrong on {qc['naive_gate_error_rate']*100:.0f}% of protocols, "
          f"incl. {qc['naive_gate_false_accept_rate']*100:.0f}% false-accepts (bad library released)")
    print("=======================================================")
    print(f"\nFigures -> {OUT}/   Metrics -> metrics.json")


if __name__ == "__main__":
    main()

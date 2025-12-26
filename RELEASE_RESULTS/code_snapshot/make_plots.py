\
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
STATS = ROOT / "artifacts" / "stats"
STD = ROOT / "artifacts" / "standardized"
PLOT = ROOT / "artifacts" / "plots"
PLOT.mkdir(parents=True, exist_ok=True)
print(f"[debug] ROOT={ROOT} PLOT={PLOT}")


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(PLOT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(PLOT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rosenbrock_lr_sweep():
    f = STATS / "summary_rosenbrock100d.csv"
    if not f.exists():
        print("[plots] missing summary_rosenbrock100d.csv; skipping")
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("lr")
        x = sub["lr"].to_numpy(float)
        y = sub["mean"].to_numpy(float)

        lo = sub["ci95_low"].to_numpy(float)
        hi = sub["ci95_high"].to_numpy(float)
        # CI error bars on log-scale must be positive
        yerr_lower = np.clip(y - lo, 0.0, np.inf)
        yerr_upper = np.clip(hi - y, 0.0, np.inf)
        yerr = np.vstack([yerr_lower, yerr_upper])

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)

        # annotate divergence rate if present
        if "divergence_rate" in sub.columns:
            for xi, yi, dr in zip(x, y, sub["divergence_rate"].to_numpy(float)):
                if dr > 0:
                    ax.annotate(f"div={dr:.0%}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate (h)")
    ax.set_ylabel("Final loss (mean; 95% CI)")
    ax.set_title("Rosenbrock-100D LR sweep (3000 grad evals)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_rosenbrock100d_lr_sweep")


def plot_ablation_bar():
    f = STATS / "summary_rosenbrock_ablation.csv"
    if not f.exists():
        print("[plots] missing summary_rosenbrock_ablation.csv; skipping")
        return
    df = pd.read_csv(f).sort_values("variant")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    labels = df["variant"].tolist()
    y = df["mean"].to_numpy(float)
    lo = df["ci95_low"].to_numpy(float)
    hi = df["ci95_high"].to_numpy(float)
    yerr = np.vstack([np.clip(y - lo, 0, np.inf), np.clip(hi - y, 0, np.inf)])

    ax.bar(labels, y)
    ax.errorbar(np.arange(len(y)), y, yerr=yerr, fmt="none", capsize=3)
    ax.set_yscale("log")
    ax.set_ylabel("Final loss (mean; 95% CI)")
    ax.set_title("Rosenbrock ablations (5 seeds)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    _save(fig, "fig_rosenbrock_ablation")


def plot_pinn_suite_final():
    f = STD / "standardized_pinn_suite.csv"
    if not f.exists():
        print("[plots] missing standardized_pinn_suite.csv; skipping")
        return
    df = pd.read_csv(f)

    problems = list(df["problem"].unique())
    pipelines = ["AdamW→LBFGS", "RMSProp→LBFGS", "FANoS→LBFGS"]
    # normalize pipeline labels
    def norm_pipe(p: str) -> str:
        p = str(p)
        return p.replace("LBFGS", "LBFGS").replace("->", "→")

    df["pipeline"] = df["pipeline"].apply(norm_pipe)

    fig, axes = plt.subplots(1, len(problems), figsize=(6.5 + 3.5*len(problems), 4.2), sharey=False)
    if len(problems) == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        sub = df[df["problem"] == prob].copy()
        data = [sub[sub["pipeline"] == p]["final_loss"].to_numpy(float) for p in pipelines if p in sub["pipeline"].unique()]
        labels = [p for p in pipelines if p in sub["pipeline"].unique()]
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_yscale("log")
        ax.set_title(prob)
        ax.set_ylabel("Final loss after L-BFGS (log)")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("PINN warm-start suite: final loss after L-BFGS (5 seeds)")
    _save(fig, "fig_pinn_suite_final_loss")


def plot_pinn_suite_warmup():
    f = STD / "standardized_pinn_suite.csv"
    if not f.exists():
        print("[plots] missing standardized_pinn_suite.csv; skipping")
        return
    df = pd.read_csv(f)

    problems = list(df["problem"].unique())
    pipelines = ["AdamW→LBFGS", "RMSProp→LBFGS", "FANoS→LBFGS"]
    df["pipeline"] = df["pipeline"].astype(str).str.replace("->", "→")

    fig, axes = plt.subplots(1, len(problems), figsize=(6.5 + 3.5*len(problems), 4.2), sharey=False)
    if len(problems) == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        sub = df[df["problem"] == prob].copy()
        data = [sub[sub["pipeline"] == p]["warmup_loss"].to_numpy(float) for p in pipelines if p in sub["pipeline"].unique()]
        labels = [p for p in pipelines if p in sub["pipeline"].unique()]
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_yscale("log")
        ax.set_title(prob)
        ax.set_ylabel("Warm-start loss (before L-BFGS, log)")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("PINN warm-start suite: warmup loss before L-BFGS (5 seeds)")
    _save(fig, "fig_pinn_suite_warmup_loss")


def plot_quadratic_sweep():
    f = STATS / "summary_quadratic_sweep.csv"
    if not f.exists():
        print("[plots] missing summary_quadratic_sweep.csv; skipping")
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("condition_number")
        x = sub["condition_number"].to_numpy(float)
        y = sub["mean"].to_numpy(float)
        lo = sub["ci95_low"].to_numpy(float)
        hi = sub["ci95_high"].to_numpy(float)
        yerr = np.vstack([np.clip(y - lo, 0, np.inf), np.clip(hi - y, 0, np.inf)])
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Condition number κ(A)")
    ax.set_ylabel("Final loss (mean; 95% CI)")
    ax.set_title("Ill-conditioned quadratic sweep (3000 grad evals)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_quadratic_sweep")


def plot_thermostat_diagnostics():
    f = STD / "standardized_thermostat_diagnostics.csv"
    if not f.exists():
        print("[plots] missing standardized_thermostat_diagnostics.csv; skipping")
        return
    df = pd.read_csv(f)

    regimes = list(df["regime"].unique())
    # 2 rows: zeta and temperature tracking
    fig, axes = plt.subplots(2, len(regimes), figsize=(6.0 + 4.0*len(regimes), 6.5), sharex=False)

    if len(regimes) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, reg in enumerate(regimes):
        sub = df[df["regime"] == reg].sort_values("step")

        ax0 = axes[0, j]
        ax0.plot(sub["step"], sub["zeta"], label="ζ (friction)")
        ax0.set_title(f"Regime: {reg}")
        ax0.set_xlabel("Step")
        ax0.set_ylabel("ζ")
        ax0.grid(True, alpha=0.3)

        ax1 = axes[1, j]
        ax1.plot(sub["step"], sub["T_inst"], label="T_inst")
        ax1.plot(sub["step"], sub["T_ema"], label="T_ema")
        ax1.plot(sub["step"], sub["T0"], label="T0 (target)")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Temperature proxy")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(fontsize=8)

    fig.suptitle("Thermostat diagnostics: friction and temperature tracking")
    _save(fig, "fig_thermostat_diagnostics")
def main():
    plot_rosenbrock_lr_sweep()
    plot_ablation_bar()
    plot_pinn_suite_final()
    plot_pinn_suite_warmup()
    plot_quadratic_sweep()
    plot_thermostat_diagnostics()


if __name__ == "__main__":
    main()

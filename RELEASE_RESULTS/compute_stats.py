\
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STD = ROOT / "artifacts" / "standardized"
STATS = ROOT / "artifacts" / "stats"
STATS.mkdir(parents=True, exist_ok=True)


def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI for the mean. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    n = x.size
    for _ in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        means.append(np.mean(samp))
    means = np.asarray(means)
    lo = np.quantile(means, alpha/2)
    hi = np.quantile(means, 1-alpha/2)
    return float(np.mean(x)), float(lo), float(hi)


def summary_rosenbrock():
    f = STD / "standardized_rosenbrock100d.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    # Only final_loss
    df = df[df["metric_name"] == "final_loss"].copy()
    # Divergence rate per method/lr
    gcols = ["method", "lr"]
    rows = []
    for (m, lr), sub in df.groupby(gcols):
        vals = sub["metric_value"].to_numpy(float)
        diverged = sub["diverged"].to_numpy(bool)
        vals_ok = vals[~diverged & np.isfinite(vals)]
        mean, lo, hi = bootstrap_ci_mean(vals_ok, seed=123)
        rows.append({
            "method": m,
            "lr": lr,
            "n": int(len(sub)),
            "mean": mean,
            "std": float(np.std(vals_ok)) if vals_ok.size else np.nan,
            "ci95_low": lo,
            "ci95_high": hi,
            "divergence_rate": float(np.mean(diverged)) if len(diverged) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["method", "lr"])
    out.to_csv(STATS / "summary_rosenbrock100d.csv", index=False)

    # Best LR per method (min mean, zero divergence)
    best_rows = []
    for m, subm in out.groupby("method"):
        stable = subm[subm["divergence_rate"] == 0.0]
        if stable.empty:
            pick = subm.loc[subm["mean"].idxmin()]
        else:
            pick = stable.loc[stable["mean"].idxmin()]
        best_rows.append(pick.to_dict())
    best = pd.DataFrame(best_rows)
    best.to_csv(STATS / "best_rosenbrock100d.csv", index=False)


def summary_ablation():
    f = STD / "standardized_rosenbrock_ablation.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    rows = []
    for variant, sub in df.groupby("variant"):
        vals = sub["metric_value"].to_numpy(float)
        div = sub["diverged"].to_numpy(bool)
        vals_ok = vals[~div & np.isfinite(vals)]
        mean, lo, hi = bootstrap_ci_mean(vals_ok, seed=456)
        rows.append({
            "variant": variant,
            "n": int(len(sub)),
            "mean": mean,
            "std": float(np.std(vals_ok)) if vals_ok.size else np.nan,
            "ci95_low": lo,
            "ci95_high": hi,
            "divergence_rate": float(np.mean(div)) if len(div) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("variant")
    out.to_csv(STATS / "summary_rosenbrock_ablation.csv", index=False)


def summary_pinn_suite():
    f = STD / "standardized_pinn_suite.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    rows = []
    warm_rows = []
    for (prob, pipe), sub in df.groupby(["problem", "pipeline"]):
        final_vals = sub["final_loss"].to_numpy(float)
        time_vals = sub["time_sec"].to_numpy(float)
        warm_vals = sub["warmup_loss"].to_numpy(float)

        mean_f, lo_f, hi_f = bootstrap_ci_mean(final_vals, seed=789)
        mean_t, lo_t, hi_t = bootstrap_ci_mean(time_vals, seed=790)

        rows.append({
            "problem": prob,
            "pipeline": pipe,
            "n": int(len(sub)),
            "final_mean": mean_f,
            "final_std": float(np.std(final_vals)),
            "final_ci95_low": lo_f,
            "final_ci95_high": hi_f,
            "time_mean": float(np.mean(time_vals)),
            "time_std": float(np.std(time_vals)),
            "time_ci95_low": lo_t,
            "time_ci95_high": hi_t,
        })

        mean_w, lo_w, hi_w = bootstrap_ci_mean(warm_vals, seed=791)
        warm_rows.append({
            "problem": prob,
            "pipeline": pipe,
            "n": int(len(sub)),
            "warm_mean": mean_w,
            "warm_std": float(np.std(warm_vals)),
            "warm_ci95_low": lo_w,
            "warm_ci95_high": hi_w,
        })

    out = pd.DataFrame(rows).sort_values(["problem", "pipeline"])
    out.to_csv(STATS / "summary_pinn_suite.csv", index=False)
    out_w = pd.DataFrame(warm_rows).sort_values(["problem", "pipeline"])
    out_w.to_csv(STATS / "summary_pinn_suite_warmup.csv", index=False)


def summary_quadratic():
    f = STD / "standardized_quadratic_sweep.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    rows = []
    for (cond, method), sub in df.groupby(["condition_number", "method"]):
        vals = sub["final_loss"].to_numpy(float)
        div = sub["diverged"].to_numpy(bool)
        vals_ok = vals[~div & np.isfinite(vals)]
        mean, lo, hi = bootstrap_ci_mean(vals_ok, seed=999)
        rows.append({
            "condition_number": cond,
            "method": method,
            "n": int(len(sub)),
            "mean": mean,
            "std": float(np.std(vals_ok)) if vals_ok.size else np.nan,
            "ci95_low": lo,
            "ci95_high": hi,
            "divergence_rate": float(np.mean(div)) if len(div) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values(["condition_number", "method"])
    out.to_csv(STATS / "summary_quadratic_sweep.csv", index=False)


def main():
    summary_rosenbrock()
    summary_ablation()
    summary_pinn_suite()
    summary_quadratic()


if __name__ == "__main__":
    main()

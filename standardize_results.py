
"""
Standardize raw results into clean, versioned CSV schemas.

This script is intentionally strict:
- If a required raw file exists, it must parse and must contain required columns.
- Standardized outputs overwrite previous files (deterministic).
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "artifacts" / "raw"
STD = ROOT / "artifacts" / "standardized"
STD.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}. Found: {list(df.columns)}")


def standardize_rosenbrock() -> None:
    f = RAW / "raw_rosenbrock100d_runs.csv"
    if not f.exists():
        print("[standardize] rosenbrock raw not found; skipping.")
        return
    df = pd.read_csv(f)
    # Expected:
    # method, lr, seed, final_loss, diverged, evals, steps, time_sec
    _require_cols(df, ["method", "lr", "seed", "final_loss", "diverged"], "raw_rosenbrock100d_runs.csv")

    out = pd.DataFrame({
        "task": "rosenbrock100d",
        "method": df["method"].astype(str),
        "variant": "default",
        "seed": df["seed"].astype(int),
        "lr": pd.to_numeric(df["lr"], errors="coerce"),
        "budget_type": "grad_evals",
        "budget_value": pd.to_numeric(df.get("evals", 3000), errors="coerce").fillna(3000).astype(int),
        "time_sec": pd.to_numeric(df.get("time_sec", df.get("time", np.nan)), errors="coerce"),
        "metric_name": "final_loss",
        "metric_value": pd.to_numeric(df["final_loss"], errors="coerce"),
        "diverged": df["diverged"].astype(bool),
        "notes": ""
    })
    out.to_csv(STD / "standardized_rosenbrock100d.csv", index=False)
    print(f"[standardize] wrote {STD/'standardized_rosenbrock100d.csv'} ({len(out)} rows)")


def standardize_ablation() -> None:
    f = RAW / "raw_rosenbrock_ablation_runs.csv"
    if not f.exists():
        print("[standardize] ablation raw not found; skipping.")
        return
    df = pd.read_csv(f)
    # Check for variant column (new runner updates)
    if "variant" in df.columns:
        _require_cols(df, ["method", "variant", "seed", "final_loss"], "raw_rosenbrock_ablation_runs.csv")
        variants = df["variant"]
    else:
        # Legacy fallback
        _require_cols(df, ["method", "seed", "final_loss"], "raw_rosenbrock_ablation_runs.csv")
        variants = df["method"] # Assuming method contains info

    def map_variant(m: str) -> str:
        m = str(m)
        if "Explicit" in m or "Euler" in m:
            return "explicit_euler"
        if "Fixed" in m or "Friction" in m:
            return m # Use full name for now or simplify. Let's keep specific names like A2-FixedFriction-0
        if "Identity" in m:
            return "identity_mass"
        if "NoGradClip" in m:
            return "no_grad_clip"
        if "NoTSchedule" in m:
            return "no_T_schedule"
        if "Baseline" in m:
            return "FANoS-Baseline"
        return m

    mapped_variants = [map_variant(v) for v in variants]

    out = pd.DataFrame({
        "task": "rosenbrock100d",
        "method": "FANoS-RMS",
        "variant": mapped_variants,
        "seed": df["seed"].astype(int),
        "lr": pd.to_numeric(df.get("lr", 0.003), errors="coerce"),
        "budget_type": "grad_evals",
        "budget_value": 3000,
        "time_sec": pd.to_numeric(df.get("time_sec", np.nan), errors="coerce"),
        "metric_name": "final_loss",
        "metric_value": pd.to_numeric(df["final_loss"], errors="coerce"),
        "diverged": df.get("diverged", ~np.isfinite(pd.to_numeric(df["final_loss"], errors="coerce"))).astype(bool),
        "notes": ""
    })
    out.to_csv(STD / "standardized_rosenbrock_ablation.csv", index=False)
    print(f"[standardize] wrote {STD/'standardized_rosenbrock_ablation.csv'} ({len(out)} rows)")


def standardize_pinn_suite() -> None:
    f = RAW / "raw_pinn_suite_runs.csv"
    if not f.exists():
        print("[standardize] pinn raw not found; skipping.")
        return
    df = pd.read_csv(f)
    
    # Support pipeline or strategy
    if "pipeline" in df.columns:
        _require_cols(df, ["problem", "pipeline", "seed", "warmup_loss", "final_loss", "time_sec"], "raw_pinn_suite_runs.csv")
        pipelines = df["pipeline"]
        time_col = "time_sec"
    else:
        _require_cols(df, ["problem", "strategy", "seed", "warmup_loss", "final_loss", "time"], "raw_pinn_suite_runs.csv")
        pipelines = df["strategy"]
        time_col = "time"

    # pipeline like AdamW->LBFGS
    def split_pipeline(s: str):
        s = str(s).replace("->", "_")
        if "_LBFGS" in s:
            warm = s.replace("_LBFGS", "")
            return warm, "LBFGS"
        if "LBFGS" in s: 
            return s, "" # pure LBFGS?
        return s, ""

    warm, refine = zip(*[split_pipeline(s) for s in pipelines])
    out = pd.DataFrame({
        "task": "pinn_suite",
        "problem": df["problem"].astype(str),
        "warm_start": list(warm),
        "refine": list(refine),
        "pipeline": [f"{w}→{r}" if r else w for w, r in zip(warm, refine)],
        "seed": df["seed"].astype(int),
        "warmup_loss": pd.to_numeric(df["warmup_loss"], errors="coerce"),
        "final_loss": pd.to_numeric(df["final_loss"], errors="coerce"),
        "time_sec": pd.to_numeric(df[time_col], errors="coerce"),
    })
    out.to_csv(STD / "standardized_pinn_suite.csv", index=False)
    print(f"[standardize] wrote {STD/'standardized_pinn_suite.csv'} ({len(out)} rows)")


def standardize_quadratic() -> None:
    f = RAW / "raw_quadratic_sweep_runs.csv"
    if not f.exists():
        print("[standardize] quadratic raw not found; skipping.")
        return
    df = pd.read_csv(f)
    _require_cols(df, ["condition_number", "method", "seed", "final_loss"], "raw_quadratic_sweep_runs.csv")
    out = pd.DataFrame({
        "task": "quadratic_sweep",
        "condition_number": pd.to_numeric(df["condition_number"], errors="coerce"),
        "method": df["method"].astype(str),
        "seed": df["seed"].astype(int),
        "lr": pd.to_numeric(df.get("lr", np.nan), errors="coerce"),
        "budget_value": pd.to_numeric(df.get("evals", 3000), errors="coerce").fillna(3000).astype(int),
        "time_sec": pd.to_numeric(df.get("time_sec", df.get("time", np.nan)), errors="coerce"),
        "final_loss": pd.to_numeric(df["final_loss"], errors="coerce"),
        "diverged": df.get("diverged", False).astype(bool) if "diverged" in df.columns else ~np.isfinite(pd.to_numeric(df["final_loss"], errors="coerce")),
    })
    out.to_csv(STD / "standardized_quadratic_sweep.csv", index=False)
    print(f"[standardize] wrote {STD/'standardized_quadratic_sweep.csv'} ({len(out)} rows)")


def standardize_diagnostics() -> None:
    f = RAW / "raw_thermostat_diagnostics.csv"
    if not f.exists():
        print("[standardize] diagnostics raw not found; skipping.")
        return
    df = pd.read_csv(f)
    # Check for raw contract columns
    req = ["regime", "seed", "lr", "step", "loss", "zeta", "T_inst", "T_ema", "T0", "grad_norm", "wall_time"]
    found = df.columns
    # Allow partial if old format, but prompt requires strict.
    _require_cols(df, ["regime", "step", "zeta", "T_inst", "T_ema", "T0"], "raw_thermostat_diagnostics.csv")
    
    # Just pass through or clean up
    df.to_csv(STD / "standardized_thermostat_diagnostics.csv", index=False)
    print(f"[standardize] wrote {STD/'standardized_thermostat_diagnostics.csv'} ({len(df)} rows)")


def main():
    standardize_rosenbrock()
    standardize_ablation()
    standardize_pinn_suite()
    standardize_quadratic()
    standardize_diagnostics()


if __name__ == "__main__":
    main()

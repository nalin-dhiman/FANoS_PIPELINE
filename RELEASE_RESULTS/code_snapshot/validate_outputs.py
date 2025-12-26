
from __future__ import annotations
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "artifacts" / "raw"
STD = ROOT / "artifacts" / "standardized"
STATS = ROOT / "artifacts" / "stats"
PLOTS = ROOT / "artifacts" / "plots"
TABLES = ROOT / "artifacts" / "tables"
LATEX = ROOT / "latex_snippets"

REQUIRED_ARTIFACTS = [
    # RAW
    RAW / "raw_rosenbrock100d_runs.csv",
    RAW / "raw_rosenbrock_ablation_runs.csv",
    RAW / "raw_quadratic_sweep_runs.csv",
    RAW / "raw_thermostat_diagnostics.csv",
    RAW / "raw_pinn_suite_runs.csv",
    
    # STANDARDIZED
    STD / "standardized_rosenbrock100d.csv",
    STD / "standardized_rosenbrock_ablation.csv",
    STD / "standardized_pinn_suite.csv",
    STD / "standardized_quadratic_sweep.csv",
    STD / "standardized_thermostat_diagnostics.csv",
    
    # STATS
    STATS / "summary_rosenbrock100d.csv",
    STATS / "best_rosenbrock100d.csv",
    STATS / "summary_rosenbrock_ablation.csv",
    STATS / "summary_pinn_suite.csv",
    STATS / "summary_pinn_suite_warmup.csv",
    STATS / "summary_quadratic_sweep.csv",
    
    # PLOTS
    PLOTS / "fig_rosenbrock100d_lr_sweep.pdf",
    PLOTS / "fig_rosenbrock100d_lr_sweep.png",
    PLOTS / "fig_rosenbrock_ablation.pdf",
    PLOTS / "fig_rosenbrock_ablation.png",
    PLOTS / "fig_pinn_suite_final_loss.pdf",
    PLOTS / "fig_pinn_suite_final_loss.png",
    PLOTS / "fig_pinn_suite_warmup_loss.pdf",
    PLOTS / "fig_pinn_suite_warmup_loss.png",
    PLOTS / "fig_quadratic_sweep.pdf",
    PLOTS / "fig_quadratic_sweep.png",
    PLOTS / "fig_thermostat_diagnostics.pdf",
    PLOTS / "fig_thermostat_diagnostics.png",
    
    # TABLES
    TABLES / "table_rosenbrock100d_summary.tex",
    TABLES / "table_rosenbrock_ablation.tex",
    TABLES / "table_pinn_suite_summary.tex",
    TABLES / "table_pinn_suite_warmup.tex",
    TABLES / "table_quadratic_sweep_summary.tex",
    
    # LATEX
    LATEX / "INCLUDE_FIGURES.tex",
    LATEX / "INCLUDE_TABLES.tex",
]

def require_cols(path: Path, cols: list[str]):
    if not path.exists():
        return # Checked in main
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"[validate] FAIL: {path.name} missing columns {missing}. Found {list(df.columns)}")

def main():
    missing = []
    for p in REQUIRED_ARTIFACTS:
        if not p.exists():
            missing.append(str(p))
        elif p.stat().st_size == 0:
            missing.append(f"{p} (empty)")
            
    if missing:
        print("[validate] FAIL: Missing or empty required files:")
        for m in missing:
            print("  -", m)
        raise SystemExit(1)

    # Strict Schema Checks (Raw)
    # per DATA/ARTIFACT CONTRACT
    require_cols(RAW / "raw_rosenbrock100d_runs.csv", 
                 ["method", "lr", "seed", "final_loss", "diverged", "evals", "steps", "time_sec"])
    require_cols(RAW / "raw_rosenbrock_ablation_runs.csv",
                 ["method", "variant", "lr", "seed", "final_loss", "diverged", "evals", "time_sec"])
    require_cols(RAW / "raw_quadratic_sweep_runs.csv",
                 ["condition_number", "method", "lr", "seed", "final_loss", "diverged", "evals", "time_sec"])
    require_cols(RAW / "raw_thermostat_diagnostics.csv",
                 ["regime", "seed", "lr", "step", "loss", "zeta", "T_inst", "T_ema", "T0", "grad_norm", "wall_time"])
    require_cols(RAW / "raw_pinn_suite_runs.csv",
                 ["problem", "pipeline", "seed", "warmup_loss", "final_loss", "time_sec"])

    # Strict Schema Checks (Standardized)

                 
    require_cols(STD / "standardized_rosenbrock100d.csv",
                 ["task", "method", "seed", "lr", "metric_value", "diverged"])

    require_cols(STD / "standardized_rosenbrock_ablation.csv",
                 ["task", "method", "variant", "seed", "metric_value", "diverged"])
                 
    require_cols(STD / "standardized_pinn_suite.csv",
                 ["task", "problem", "pipeline", "seed", "warmup_loss", "final_loss", "time_sec"])

    require_cols(STD / "standardized_quadratic_sweep.csv",
                 ["task", "condition_number", "method", "seed", "final_loss", "diverged"])

    print("[validate] PASS: All required files present and schemas valid.")

if __name__ == "__main__":
    main()

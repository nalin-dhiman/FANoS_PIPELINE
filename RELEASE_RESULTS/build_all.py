
from __future__ import annotations

import subprocess
from pathlib import Path
import sys
import pandas as pd
import os

# ROOT is the directory containing this script (FANoS_PUBLISHABLE_PIPELINE)
ROOT = Path(__file__).resolve().parent

def run(cmd: list[str]):
    print(f"[build] Executing: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(ROOT))

def main():
    print(f"[build] Root directory: {ROOT}")
    
    # ---------------------------------------------------------
    # 1. Run T1 (Rosenbrock Sweep)
    # ---------------------------------------------------------
    print("\n--- T1: Rosenbrock Sweep ---")
    if not (ROOT / "artifacts/raw/raw_rosenbrock100d_runs.csv").exists():
        run([sys.executable, "run_rosenbrock_sweep.py"])
    
    # Standardize T1 immediately to enable Best LR lookup
    run([sys.executable, "standardize_results.py"]) 
    run([sys.executable, "compute_stats.py"])
    
    # ---------------------------------------------------------
    # 2. Identify Best LR for FANoS from T1
    # ---------------------------------------------------------
    best_lr = 1e-3 # Default fallback
    best_file = ROOT / "artifacts/stats/best_rosenbrock100d.csv"
    if best_file.exists():
        df = pd.read_csv(best_file)
        fanos_row = df[df["method"] == "FANoS-RMS"]
        if not fanos_row.empty:
            best_lr = float(fanos_row.iloc[0]["lr"])
            print(f"[build] Found Best LR for FANoS-RMS: {best_lr}")
        else:
            print("[build] Warning: FANoS-RMS not found in best stats. Using default 1e-3.")
    else:
        print("[build] Warning: Best stats file not found. Using default 1e-3.")

    # ---------------------------------------------------------
    # 3. Run T2 (Ablations) using Best LR
    # ---------------------------------------------------------
    print(f"\n--- T2: Ablations (LR={best_lr}) ---")
    if not (ROOT / "artifacts/raw/raw_rosenbrock_ablation_runs.csv").exists():
         run([sys.executable, "run_rosenbrock_ablation.py", "--lr", str(best_lr)])
         
    # ---------------------------------------------------------
    # 4. Run T3 (Quadratic), T4 (Thermostat), T5 (PINN)
    # ---------------------------------------------------------
    print("\n--- T3: Quadratic Sweep ---")
    if not (ROOT / "artifacts/raw/raw_quadratic_sweep_runs.csv").exists():
        run([sys.executable, "run_quadratic_sweep.py"])

    print("\n--- T4: Thermostat Diagnostics ---")
    if not (ROOT / "artifacts/raw/raw_thermostat_diagnostics.csv").exists():
        run([sys.executable, "run_thermostat_diagnostics.py"])

    print("\n--- T5: PINN Suite ---")
    if not (ROOT / "artifacts/raw/raw_pinn_suite_runs.csv").exists():
        run([sys.executable, "run_pinn_suite.py"])

    # ---------------------------------------------------------
    # 5. Full Pipeline Construction
    # ---------------------------------------------------------
    print("\n--- Standardizing All ---")
    run([sys.executable, "standardize_results.py"])

    print("\n--- Computing Stats ---")
    run([sys.executable, "compute_stats.py"])

    print("\n--- Generating Plots ---")
    run([sys.executable, "make_plots.py"])

    print("\n--- Generating Tables ---")
    run([sys.executable, "make_tables.py"])

    # print("\n--- LaTeX Snippets ---")
    if (ROOT / "make_latex_snippets.py").exists():
        run([sys.executable, "make_latex_snippets.py"])

    # ---------------------------------------------------------
    # 6. Strict Validation
    # ---------------------------------------------------------
    print("\n--- Validating Artifacts ---")
    run([sys.executable, "validate_outputs.py"])

    print("\n[build] SUCCESS: All steps completed.")

if __name__ == "__main__":
    main()

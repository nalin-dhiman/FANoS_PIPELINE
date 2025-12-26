
import torch
import torch.nn as nn
import pandas as pd
import time
import argparse
from pathlib import Path
import math

from fanos_optimizer import FANoS

# -------------------------------------------------------------------
# Rosenbrock 100D (Reused)
# -------------------------------------------------------------------
class Rosenbrock(nn.Module):
    def __init__(self, dim=100):
        super().__init__()
        self.dim = dim
        self.theta = nn.Parameter(torch.zeros(dim))

    def forward(self):
        x = self.theta
        term1 = 100.0 * (x[1:] - x[:-1]**2)**2
        term2 = (1.0 - x[:-1])**2
        return term1.sum() + term2.sum()

# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
def run_ablation(best_lr=1e-3):
    # Setup
    seeds = range(10)
    budget = 3000
    
    # Variants: A1..A5
    # A1: explicit_euler
    # A2: fixed_friction (zeta=0, 1, 5)
    # A3: identity_mass
    # A4: no_grad_clip
    # A5: no_T_schedule (T0 = Tmax constant)
    
    # We will define a list of configs
    variants = [
        {"name": "FANoS-Baseline", "kwargs": {}}, # uses defaults/best_lr
        {"name": "A1-ExplicitEuler", "kwargs": {"explicit_euler": True}},
        {"name": "A2-FixedFriction-0", "kwargs": {"fixed_friction": True, "zeta_const": 0.0}},
        {"name": "A2-FixedFriction-1", "kwargs": {"fixed_friction": True, "zeta_const": 1.0}},
        {"name": "A2-FixedFriction-5", "kwargs": {"fixed_friction": True, "zeta_const": 5.0}},
        {"name": "A3-IdentityMass", "kwargs": {"identity_mass": True}},
        {"name": "A4-NoGradClip", "kwargs": {"grad_clip": None}}, # or huge value
        {"name": "A5-NoTSchedule", "kwargs": {"T0_min": 1e-3, "T0_max": 1e-3}}, # Constant T
    ]

    results = []
    
    out_dir = Path("artifacts/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_count = 0
    total = len(variants) * len(seeds)

    for vid, variant in enumerate(variants):
        name = variant["name"]
        kwargs = variant["kwargs"]
        
        for seed in seeds:
            run_count += 1
            print(f"[{run_count}/{total}] {name} seed={seed}")

            torch.manual_seed(seed)
            model = Rosenbrock(dim=100)
            nn.init.uniform_(model.theta, -2.0, 2.0)
            
            # Init optimizer with merged kwargs
            # Defaults for FANoS-RMS from T1
            opt_kwargs = {
                "lr": best_lr,
                "grad_clip": 1.0, # Baseline matches T1
                "identity_mass": False,
                # "T0_max": 1e-3 (default)
                # "T0_min": 0.0 (default)
            }
            opt_kwargs.update(kwargs)
            
            # Explicitly checking for None in grad_clip to pass correctly if needed
            # fanos_optimizer handles grad_clip=None
            
            opt = FANoS(model.parameters(), **opt_kwargs)

            evals = 0
            steps = 0
            final_loss = float('inf')
            diverged = False
            start_time = time.time()
            
            try:
                while evals < budget:
                    opt.zero_grad()
                    loss = model()
                    loss.backward()
                    evals += 1
                    
                    final_loss = float(loss.item())
                    if not math.isfinite(final_loss):
                        diverged = True
                        break
                    
                    opt.step()
                    steps += 1
            except Exception:
                diverged = True

            duration = time.time() - start_time
            
            results.append({
                "method": "FANoS", # Base method
                "variant": name,
                "lr": best_lr,
                "seed": seed,
                "final_loss": final_loss if not diverged else 1e9,
                "diverged": diverged,
                "evals": evals,
                "time_sec": duration
            })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "raw_rosenbrock_ablation_runs.csv", index=False)
    print("Completed Rosenbrock Ablations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="Best LR from T1")
    args = parser.parse_args()
    run_ablation(best_lr=args.lr)


import torch
import torch.nn as nn
import pandas as pd
import time
import argparse
from pathlib import Path
import math

from fanos_optimizer import FANoS

# -------------------------------------------------------------------
# Rosenbrock 100D
# -------------------------------------------------------------------
class Rosenbrock(nn.Module):
    def __init__(self, dim=100):
        super().__init__()
        self.dim = dim
        # Initialize randomly away from optimum (1,1,...,1)
        # Standard init is often uniform [-2, 2] or N(0,1).
        # We'll use a fixed seed-based init in the runner loop.
        self.theta = nn.Parameter(torch.zeros(dim))

    def forward(self):
        # sum_{i=0}^{N-2} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
        x = self.theta
        term1 = 100.0 * (x[1:] - x[:-1]**2)**2
        term2 = (1.0 - x[:-1])**2
        return term1.sum() + term2.sum()

# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
def run_sweep():
    # Setup
    methods = ["SGD+Mom", "AdamW", "RMSProp", "LBFGS", "FANoS-RMS"]
    lrs = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    seeds = range(10)
    budget = 3000

    results = []
    
    # Ensure artifacts dir exists
    out_dir = Path("artifacts/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_count = 0
    total_runs = len(methods) * len(lrs) * len(seeds)

    for method_name in methods:
        # For FANoS variants, we also run clipped versions of baselines if FANoS uses clip
        # But per spec T1, we need "AdamW+clip" and "RMSProp+clip" IF FANoS uses clipping.
        # Default FANoS uses thermostat clip but we should check grad clip.
        # Spec says: "Also run “AdamW+clip” and “RMSProp+clip” if FANoS uses clipping."
        # FANoS default has grad_clip=1.0. So we should add clipped baselines.
        
        # We will iterate strict method list from spec (+ clipped extra)
        # Spec list: SGD+Mom, AdamW, RMSProp, LBFGS, FANoS-RMS
        pass

    # Extended method list including clipped versions
    # FANoS default grad_clip is 1.0 (from code view).
    full_method_list = list(methods)
    full_method_list.append("AdamW+clip")
    full_method_list.append("RMSProp+clip")

    for method in full_method_list:
        for lr in lrs:
            for seed in seeds:
                run_count += 1
                print(f"[{run_count}/{total_runs}] {method} lr={lr} seed={seed}")

                # Init problem
                torch.manual_seed(seed)
                # Standard init for Rosenbrock often localized. 
                # Let's pick standard N(0, 0.1) or Uniform(-1,1).
                # To be reproducible and stiff, let's use Uniform(-2, 2) which is common.
                model = Rosenbrock(dim=100)
                nn.init.uniform_(model.theta, -2.0, 2.0)
                
                # Setup optimizer
                opt = None
                grad_clip_val = 1.0 # default for FANoS match
                
                # Base params
                p = model.parameters()
                
                if method == "SGD+Mom":
                    opt = torch.optim.SGD(p, lr=lr, momentum=0.9)
                elif method == "AdamW":
                    opt = torch.optim.AdamW(p, lr=lr)
                elif method == "AdamW+clip":
                    opt = torch.optim.AdamW(p, lr=lr)
                elif method == "RMSProp":
                    opt = torch.optim.RMSprop(p, lr=lr)
                elif method == "RMSProp+clip":
                    opt = torch.optim.RMSprop(p, lr=lr)
                elif method == "LBFGS":
                    # LBFGS needs careful closure
                    opt = torch.optim.LBFGS(p, lr=lr)
                elif method == "FANoS-RMS":
                    opt = FANoS(p, lr=lr, identity_mass=False, grad_clip=1.0)
                else:
                    raise ValueError(f"Unknown method {method}")

                # Loop
                evals = 0
                steps = 0
                final_loss = float('inf')
                diverged = False
                start_time = time.time()
                
                # For LBFGS, step() does multiple evals
                if method == "LBFGS":
                    # LBFGS typically runs fewer steps but more evals per step
                    # We bound by evals
                    def closure():
                        nonlocal evals
                        opt.zero_grad()
                        loss = model()
                        loss.backward()
                        evals += 1
                        return loss
                    
                    try:
                        while evals < budget:
                            # If loss is NaN, stop
                            if torch.isnan(model.theta).any():
                                diverged = True
                                break
                            
                            loss_val = opt.step(closure)
                            steps += 1
                            if loss_val is not None:
                                final_loss = float(loss_val.item())
                            
                            # Check NaN after step
                            if not math.isfinite(final_loss):
                                diverged = True
                                break
                    except Exception:
                        diverged = True

                else:
                    # First-order methods
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
                            
                            if "+clip" in method:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                            
                            opt.step()
                            steps += 1
                    except Exception:
                        diverged = True

                duration = time.time() - start_time
                
                # Log
                results.append({
                    "method": method,
                    "lr": lr,
                    "seed": seed,
                    "final_loss": final_loss if not diverged else 1e9, # Sentinel? Or just NaN/Inf? 
                    # Spec says: "Output: per-run final loss + diverged flag"
                    # We will output actual value or fail value.
                    # Validator usually checks finite unless diverged is True.
                    # Let's put 1e9 for diverged for plot sanity (log scale), but flag it.
                    # Actually spec says "raw logs -> standardized -> stats".
                    # Raw should capture reality.
                    "diverged": diverged,
                    "evals": evals,
                    "steps": steps,
                    "time_sec": duration
                })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "raw_rosenbrock100d_runs.csv", index=False)
    print("Completed Rosenbrock 100D sweep.")

if __name__ == "__main__":
    run_sweep()

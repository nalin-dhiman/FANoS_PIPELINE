
from __future__ import annotations

import sys
from pathlib import Path
# Fix import path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Fix import
from fanos_optimizer import FANoS

ROOT = Path(__file__).resolve().parents[0]
RAW = ROOT / "artifacts" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

def rosenbrock_loss(x):
    # x shape (D,)
    term1 = 100.0 * (x[1:] - x[:-1]**2)**2
    term2 = (1.0 - x[:-1])**2
    return term1.sum() + term2.sum()

def compute_T_inst(opt: FANoS) -> float:
    # Recompute T_inst from optimizer state (mean m*v^2).
    temp_sum = 0.0
    temp_count = 0
    for group in opt.param_groups:
        identity_mass = bool(group.get("identity_mass", False))
        eps = float(group.get("eps", 1e-8))
        for p in group["params"]:
            st = opt.state[p]
            if "v" not in st:
                continue
            v = st["v"]
            if identity_mass:
                temp_sum += float((v*v).sum().item())
            else:
                # m = sqrt(s)+eps
                s = st["s"]
                m = s.sqrt().add(eps)
                temp_sum += float((m*v*v).sum().item())
            temp_count += v.numel()
    if temp_count == 0:
        return float("nan")
    return temp_sum / float(temp_count)


def run_regime(seed: int, lr: float, steps: int, regime: str):
    torch.manual_seed(seed)
    # Init same as T1
    x0 = (torch.rand(100)*4 - 2.0)
    x = nn.Parameter(x0.clone().detach())

    opt = FANoS([x], lr=lr, Q=1.0, T0_max=1e-3, tau=20000.0, rho_T=0.9, zeta_clip=10.0, grad_clip=1.0)
    rows = []
    t0 = time.time()
    for k in range(steps):
        opt.zero_grad()
        loss = rosenbrock_loss(x)
        loss.backward()
        opt.step()
        
        # Log stats
        group = opt.param_groups[0]
        zeta = float(group["zeta"])
        T_ema = float(group["T_ema"])
        T0_val = float(opt.schedule(opt._step_count))
        # T_inst we can compute manually or if we trust internal state?
        # Code above had manual compute.
        T_inst = compute_T_inst(opt)
        
        gn = float(torch.norm(x.grad.detach()).item()) if x.grad is not None else float("nan")
        
        rows.append(dict(regime=regime, seed=seed, lr=lr, step=k, loss=float(loss.item()), 
                         zeta=zeta, T_inst=T_inst, T_ema=T_ema, T0=T0_val, grad_norm=gn, wall_time=time.time()-t0))
    return rows


def main():
    OUT = RAW / "raw_thermostat_diagnostics.csv"
    steps = 3000
    # Choose a "good" LR (near best for FANoS-RMS in the paper) and a "bad but stable" LR.
    # T1 usually shows 1e-3 or 3e-3 is good. 1e-2 might be bad/unstable. 3e-2 diverged.
    # Let's pick 3e-3 (Good) and 1e-2 (Bad/Oscillatory).
    rows = []
    print("Running Good Regime...")
    rows += run_regime(seed=0, lr=3e-3, steps=steps, regime="Good")
    print("Running Bad Regime...")
    rows += run_regime(seed=0, lr=1e-2, steps=steps, regime="Bad")
    
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"[diagnostics] wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
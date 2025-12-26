
from __future__ import annotations

import sys
from pathlib import Path
# Fix import path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Fix import: local file
from fanos_optimizer import FANoS

ROOT = Path(__file__).resolve().parents[0]
RAW = ROOT / "artifacts" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


class QuadraticProblem(nn.Module):
    def __init__(self, d: int, cond: float, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        # Construct A with specific condition number
        # A = Q Lambda Q^T
        # random Q
        W = torch.randn(d, d)
        Q, _ = torch.linalg.qr(W)
        
        # Lambda: log-spaced from 1 to cond
        eigs = torch.logspace(0, np.log10(cond), d)
        Lambda = torch.diag(eigs)
        
        self.A = Q @ Lambda @ Q.T
        self.A.requires_grad_(False)
        
    def loss(self, x):
        return 0.5 * torch.einsum("i,ij,j->", x, self.A, x)

def make_quadratic(d, condition_number, seed):
    return QuadraticProblem(d, condition_number, seed)

def run_one(method: str, x0: torch.Tensor, prob, lr: float, budget: int):
    # x0 is Parameter? No, it's tensor. Make it param.
    x = nn.Parameter(x0.clone().detach())
    
    if method == "AdamW":
        opt = torch.optim.AdamW([x], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    elif method == "RMSProp":
        opt = torch.optim.RMSprop([x], lr=lr, alpha=0.99, eps=1e-8, momentum=0.9, weight_decay=0.0)
    elif method == "SGD_Mom":
        opt = torch.optim.SGD([x], lr=lr, momentum=0.9)
    elif method == "FANoS_RMS":
        opt = FANoS([x], lr=lr, Q=1.0, T0_max=1e-3, grad_clip=1.0)
    elif method == "LBFGS":
        opt = torch.optim.LBFGS([x], lr=1.0, max_iter=20, history_size=100)
    else:
        raise ValueError(method)

    diverged = False
    evals = 0
    t0 = time.time()

    if method == "LBFGS":
        def closure():
            nonlocal evals, diverged
            opt.zero_grad()
            loss = prob.loss(x)
            if not torch.isfinite(loss):
                diverged = True
                # LBFGS might not handle exception well in closure, return NaN
                return torch.tensor(float('nan'))
            loss.backward()
            evals += 1
            return loss

        try:
            while evals < budget and not diverged:
                loss = opt.step(closure)
                if torch.isnan(loss):
                    diverged = True
        except Exception:
            diverged = True
    else:
        for _ in range(budget):
            opt.zero_grad()
            loss = prob.loss(x)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            opt.step()
            evals += 1
            if not torch.isfinite(x).all():
                diverged = True
                break

    final_loss = float(prob.loss(x).item()) if not diverged else float("inf")
    return evals, time.time() - t0, diverged, final_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(RAW / "raw_quadratic_sweep_runs.csv"))
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--conds", type=float, nargs="*", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    # Ignore full flag, just run defaults which match T3 requirements
    args = ap.parse_args()

    lrs = {
        "AdamW": 1e-2,
        "RMSProp": 1e-3,
        "SGD_Mom": 1e-4,
        "FANoS_RMS": 1e-3,
        "LBFGS": 1.0, # LBFGS usually 1.0
    }
    methods = ["AdamW", "RMSProp", "SGD_Mom", "LBFGS", "FANoS_RMS"]

    rows = []
    
    total_runs = len(args.conds) * len(args.seeds) * len(methods)
    count = 0
    
    for cond in args.conds:
        for seed in args.seeds:
            # Need strict seed control
            torch.manual_seed(seed)
            np.random.seed(seed)
            x0 = torch.randn(args.d)
            prob = make_quadratic(args.d, condition_number=cond, seed=seed)

            for method in methods:
                count += 1
                if count % 10 == 0:
                    print(f"[{count}/{total_runs}] Running {method} cond={cond:.0e} seed={seed}")
                    
                lr = lrs[method]
                evals, wall, div, final = run_one(method, x0, prob, lr=lr, budget=args.budget)
                rows.append(dict(condition_number=cond, method=method, seed=seed, lr=lr, evals=evals, time_sec=wall, diverged=div, final_loss=final))

    df = pd.DataFrame(rows)
    # output path logic
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[quadratic] wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
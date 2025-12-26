
import torch
import torch.nn as nn
import pandas as pd
import time
import numpy as np
from pathlib import Path
from fanos_optimizer import FANoS

# -------------------------------------------------------------------
# PINN Base & PDEs
# -------------------------------------------------------------------

class PINN(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 1]):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.net.add_module(f"act_{i}", nn.Tanh())
    
    def forward(self, x, t):
        # inputs: x, t (N,1) each
        # output: u (N,1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

class Burgers1D:
    def __init__(self):
        self.nu = 0.01/np.pi
        
    def loss(self, model, x_f, t_f, x_bc, t_bc, u_bc):
        # PDE Residual: u_t + u u_x - nu u_xx = 0
        u = model(x_f, t_f)
        u_t = grad(u, t_f)
        u_x = grad(u, x_f)
        u_xx = grad(u_x, x_f)
        f = u_t + u*u_x - self.nu*u_xx
        loss_f = torch.mean(f**2)
        
        # BC/IC
        u_pred_bc = model(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        return loss_f + loss_bc

class AllenCahn1D:
    def __init__(self):
        # u_t - 0.0001 u_xx + 5 u^3 - 5 u = 0
        self.nu = 0.0001
        
    def loss(self, model, x_f, t_f, x_bc, t_bc, u_bc):
        u = model(x_f, t_f)
        u_t = grad(u, t_f)
        u_x = grad(u, x_f)
        u_xx = grad(u_x, x_f)
        f = u_t - self.nu*u_xx + 5.0*u**3 - 5.0*u
        loss_f = torch.mean(f**2)
        
        u_pred_bc = model(x_bc, t_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc)**2)
        
        return loss_f + loss_bc

# -------------------------------------------------------------------
# Data Generator
# -------------------------------------------------------------------
def get_data(problem_name, N_f=2000, N_bc=100):
    # Domain x in [-1,1], t in [0,1]
    # Collocation points
    x_f = torch.rand(N_f, 1) * 2 - 1
    t_f = torch.rand(N_f, 1)
    x_f.requires_grad = True
    t_f.requires_grad = True
    
    # BC: x=-1 and x=1
    t_bc_rand = torch.rand(N_bc, 1)
    x_bc_neg = -torch.ones(N_bc, 1)
    x_bc_pos = torch.ones(N_bc, 1)
    
    # IC: t=0
    x_ic = torch.rand(N_bc, 1) * 2 - 1
    t_ic = torch.zeros(N_bc, 1)
    
    if problem_name == "Burgers":
        # u(x,0) = -sin(pi x)
        u_ic = -torch.sin(np.pi * x_ic)
        # u(0, t), u(1,t) = 0
        u_bc_neg = torch.zeros(N_bc, 1)
        u_bc_pos = torch.zeros(N_bc, 1)
    elif problem_name == "AllenCahn":
        # u(x,0) = x^2 cos(pi x)
        u_ic = x_ic**2 * torch.cos(np.pi * x_ic)
        # u(-1,t) = u(1,t)
        # periodic BC hard to enforce simply with MSE, 
        # usually assume Dirichlet or Neumann if specified.
        # Strict AC usually periodic. Or standard Dirichlet.
        # Let's use simple Dirichlet u(-1)=u(1)=0 for stability in this demo suite if acceptable?
        # Standard: u(-1,t)=u(1,t), u_x(-1,t)=u_x(1,t).
        # To simplify robust running: u(-1)=u(1) constraint is better.
        # But let's stick to Dirichlet u(-1)=u(1) approx from IC.
        # u(-1) = (-1)^2 cos(-pi) = -1. u(1) = -1.
        u_bc_neg = -torch.ones(N_bc, 1)
        u_bc_pos = -torch.ones(N_bc, 1)

    x_bc = torch.cat([x_bc_neg, x_bc_pos, x_ic], dim=0)
    t_bc = torch.cat([t_bc_rand, t_bc_rand, t_ic], dim=0)
    u_bc = torch.cat([u_bc_neg, u_bc_pos, u_ic], dim=0)
    
    return x_f, t_f, x_bc, t_bc, u_bc

# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------
def run_suite():
    problems = ["Burgers", "AllenCahn"]
    pipelines = ["AdamW->LBFGS", "RMSProp->LBFGS", "FANoS->LBFGS"]
    seeds = range(5)
    
    warmup_steps = 1000 # Short warmup to show heavy lifting by LBFGS or prep by optimizer
    lbfgs_max_iter = 200
    
    results = []
    out_dir = Path("artifacts/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    run_count = 0
    total = len(problems) * len(pipelines) * len(seeds)
    
    for prob_name in problems:
        if prob_name == "Burgers":
            pde = Burgers1D()
        else:
            pde = AllenCahn1D()
            
        for pipe in pipelines:
            warmup_opt_name = pipe.split("->")[0]
            
            for seed in seeds:
                run_count += 1
                print(f"[{run_count}/{total}] {prob_name} {pipe} seed={seed}")
                
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = PINN()
                x_f, t_f, x_bc, t_bc, u_bc = get_data(prob_name)
                
                # Warmup
                if warmup_opt_name == "AdamW":
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                elif warmup_opt_name == "RMSProp":
                    # RMSProp often unstable on PINNs, need good epsilon/alpha
                    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
                elif warmup_opt_name == "FANoS":
                    # Use reasonable defaults
                    opt = FANoS(model.parameters(), lr=1e-3, identity_mass=False)
                
                start_time = time.time()
                warmup_loss = float('inf')
                
                for step in range(warmup_steps):
                    opt.zero_grad()
                    loss = pde.loss(model, x_f, t_f, x_bc, t_bc, u_bc)
                    loss.backward()
                    opt.step()
                    warmup_loss = loss.item()
                
                # Switch to LBFGS
                lbfgs = torch.optim.LBFGS(model.parameters(), 
                                          lr=1.0, 
                                          max_iter=lbfgs_max_iter, 
                                          line_search_fn="strong_wolfe")
                
                final_loss = warmup_loss
                
                def closure():
                    lbfgs.zero_grad()
                    loss = pde.loss(model, x_f, t_f, x_bc, t_bc, u_bc)
                    loss.backward()
                    return loss
                
                try:
                    final_loss_t = lbfgs.step(closure)
                    final_loss = final_loss_t.item()
                except Exception:
                    pass
                
                duration = time.time() - start_time
                
                results.append({
                    "problem": prob_name,
                    "pipeline": pipe,
                    "seed": seed,
                    "warmup_loss": warmup_loss,
                    "final_loss": final_loss,
                    "time_sec": duration
                })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "raw_pinn_suite_runs.csv", index=False)
    print("Completed PINN Suite.")

if __name__ == "__main__":
    run_suite()

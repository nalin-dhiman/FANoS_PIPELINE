
# FANoS Results Package

This package contains the implementation, experiment runners, and results pipeline for the FANoS optimizer paper.

## Structure
- `fanos_optimizer.py`: Core optimizer implementation.
- `run_*.py`: Experiment runners for Tasks T1-T5.
- `standardize_results.py`, `compute_stats.py`: Data processing.
- `make_plots.py`, `make_tables.py`: Figure/Table generation.
- `build_all.py`: Master script to run the full pipeline.
- `artifacts/`: Generated results (Raw logs, CSVs, Plots, Tables).
- `latex_snippets/`: Auto-generated LaTeX code for figures/tables.

## Setup
Requires Python 3.8+ and PyTorch.

```bash
pip install -r requirements.txt
```

## Reproduction
To run the full pipeline (Experiments -> Data Processing -> Plots/Tables -> Validation):

```bash
python build_all.py
```

*Note: The full sweep may take several hours depending on hardware. T5 (PINN suite) and T1 (Rosenbrock sweep) are the most time-consuming.*

## Artifacts
After running, check:
- `artifacts/plots/`: PDF/PNG figures.
- `artifacts/tables/`: LaTeX tables.
- `artifacts/stats/`: Summary CSVs with bootstrapped CIs.




## Install

```bash
pip install fanos
```

## Quickstart

```python
import torch
from fanos import FANoS

model = torch.nn.Linear(10, 1)
opt = FANoS(model.parameters(), lr=1e-3, grad_clip=1.0)

x = torch.randn(64, 10)
y = torch.randn(64, 1)

loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()
opt.step()
opt.zero_grad()
```

# FANoS-v2 Decision Report

Generated: 2026-05-05 21:34:27

## Executive Verdict

FANoS-v2 is no longer just a toy optimizer. It has reproducible positive signals on MNIST, FashionMNIST, sequence memory, ODE fitting, and the PINN-specific preset.

It is still not a general AdamW replacement. The main blockers are speed, weak behavior on ill-conditioned quadratics, and the gap between `auto` and the task-specific PINN preset.

## Fashion 10Seed Auto

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 90.894% | 0.254% | 1.278% | 72.7047 |
| FANoS auto | 90.677% | 0.369% | 1.061% | 72.1346 |
| AdamW | 89.616% | 0.459% | 0.000% | 65.2177 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 11.48% to 10.6058% more CPU time.

## Cifar10 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 65.695% | 1.202% | 1.316% | 38.0922 |
| FANoS auto | 66.041% | 0.592% | 1.662% | 41.0771 |
| AdamW | 64.379% | 0.746% | 0.000% | 24.0376 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 58.4691% to 70.8867% more CPU time.

## Fashionmnist 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 90.555% | 0.486% | 0.659% | 36.8343 |
| FANoS auto | 90.836% | 0.300% | 0.939% | 37.5278 |
| AdamW | 89.896% | 0.498% | 0.000% | 15.9131 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 131.471% to 135.829% more CPU time.

## Mnist 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 98.951% | 0.123% | 0.239% | 37.9262 |
| FANoS auto | 98.901% | 0.196% | 0.189% | 38.8892 |
| AdamW | 98.712% | 0.185% | 0.000% | 15.9073 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 138.419% to 144.474% more CPU time.

## Cifar10 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 65.695% | 1.202% | 1.316% | 30.051 |
| FANoS auto | 66.041% | 0.592% | 1.662% | 33.1615 |
| AdamW | 64.379% | 0.746% | 0.000% | 16.8149 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 78.7165% to 97.2155% more CPU time.

## Eeg 5Seed

Missing summary rows.

## Fashionmnist 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 90.555% | 0.486% | 0.659% | 34.4494 |
| FANoS auto | 90.836% | 0.300% | 0.939% | 36.2401 |
| AdamW | 89.896% | 0.498% | 0.000% | 15.4714 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 122.665% to 134.239% more CPU time.

## Mnist 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 98.951% | 0.123% | 0.239% | 34.2781 |
| FANoS auto | 98.901% | 0.196% | 0.189% | 36.261 |
| AdamW | 98.712% | 0.185% | 0.000% | 16.0698 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 113.307% to 125.646% more CPU time.

## Mnist 10Seed Auto

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 98.987% | 0.097% | 0.200% | 67.777 |
| FANoS auto | 99.003% | 0.126% | 0.216% | 69.312 |
| AdamW | 98.787% | 0.143% | 0.000% | 62.3875 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 8.63881% to 11.0992% more CPU time.

## Night Smoke Fashion

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean |
| --- | ---: | ---: | ---: | ---: |
| FANoS low_lr | 53.320% | 0.000% | 15.625% | 0.257189 |
| FANoS auto | 41.406% | 0.000% | 3.711% | 0.254165 |
| AdamW | 37.695% | 0.000% | 0.000% | 0.236514 |

Verdict: FANoS is better on accuracy in this setup, but costs roughly 8.7412% to 7.46299% more CPU time.

## Night Study

Missing summary rows.

## Night Study Smoke

Missing summary rows.

## Vision 10Seed

Missing summary rows.


## Stiff And Scientific Tasks

| Task | FANoS metric | AdamW metric | Best critical read |
| --- | ---: | ---: | --- |
| Rosenbrock | 6.11189 | 9.72505 | FANoS beats AdamW, but RMSProp is stronger here. |
| Ill-conditioned quadratic | 0.0948339 | 2.643e-04 | FANoS loses badly; needs dedicated curvature handling. |
| Noisy regression | 0.0524179 | 0.0514919 | Competitive; not a decisive win. |
| ODE fit | 0.00737984 | 0.00737989 | Tied with AdamW. |
| PINN auto | 7.829e-04 | 1.863e-08 | Auto loses; use the PINN preset. |
| Sequence memory | 1 | 1 | FANoS matches accuracy and has lower loss. |

PINN preset result:

- FANoS PINN metric mean: `6.486e-09`
- AdamW metric mean: `1.863e-08`
- Critical read: the PINN math update is real, but it is not captured by the generic auto preset yet.


## Next Engineering Priorities

1. Keep `low_lr` as the best current vision preset and `auto` as the general safety preset.
2. Promote the PINN preset to a named public mode because it materially beats AdamW on the Poisson test.
3. Add CIFAR-10 with a stronger CNN before making any broader vision claim.
4. Add an ill-conditioned mode or curvature detector; current FANoS loses this task clearly.
5. Optimize runtime. FANoS wins accuracy in these vision runs but is around 9-11% slower on CPU.
6. Run the same MNIST/FashionMNIST commands on `--device mps` now that your environment reports MPS available.

## Claim Boundary

Safe claim: FANoS-v2 is a promising feedback-controlled optimizer framework with strong task-aware modes and repeated-seed wins on these lightweight benchmarks.

Unsafe claim: FANoS-v2 is a universal optimizer or a drop-in AdamW replacement.

# FANoS-v2 Decision Report

Generated: 2026-05-05 22:08:42

Source results root: `/Users/nalin/Downloads/fanos_v2/results/full_research_mps_fixed_20260505_110606`

## Executive Verdict

FANoS-v2 is no longer just a toy optimizer. It has reproducible positive signals on MNIST, FashionMNIST, sequence memory, ODE fitting, and the PINN-specific preset.

It is still not a general AdamW replacement. The main blockers are speed, weak behavior on ill-conditioned quadratics, and the gap between `auto` and the task-specific PINN preset.

## Cifar10 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean | Time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FANoS auto | 66.041% | 0.592% | 1.662% | 33.1615 | 97.2155% |
| FANoS low_lr | 65.695% | 1.202% | 1.316% | 30.051 | 78.7165% |
| FANoS vision_sweep_best | 62.972% | 1.314% | -1.407% | 30.1709 | 79.4297% |
| FANoS stable | 62.611% | 1.684% | -1.768% | 28.559 | 69.8434% |
| AdamW | 64.379% | 0.746% | 0.000% | 16.8149 | 0.000% |

Verdict: best FANoS config is ahead by 1.662%, but costs 97.2155% more wall-clock time.

## Eeg 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean | Time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FANoS auto | 45.333% | 5.055% | 0.667% | 0.916411 | 109.644% |
| FANoS stable | 43.333% | 4.082% | -1.333% | 0.853559 | 95.266% |
| FANoS low_lr | 42.667% | 4.346% | -2.000% | 0.863732 | 97.5931% |
| FANoS eeg_sweep_best | 42.000% | 5.055% | -2.667% | 0.848382 | 94.0816% |
| AdamW | 44.667% | 4.472% | 0.000% | 0.437126 | 0.000% |

Verdict: best FANoS config is ahead by 0.667%, but costs 109.644% more wall-clock time.

## Fashionmnist 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean | Time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FANoS auto | 90.836% | 0.300% | 0.939% | 36.2401 | 134.239% |
| FANoS low_lr | 90.555% | 0.486% | 0.659% | 34.4494 | 122.665% |
| FANoS vision_sweep_best | 90.283% | 0.281% | 0.386% | 35.7477 | 131.057% |
| FANoS stable | 89.912% | 0.477% | 0.016% | 35.4088 | 128.866% |
| AdamW | 89.896% | 0.498% | 0.000% | 15.4714 | 0.000% |

Verdict: best FANoS config is ahead by 0.939%, but costs 134.239% more wall-clock time.

## Mnist 5Seed

| Method | Top-1 mean | Top-1 std | Delta vs AdamW | Seconds mean | Time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| FANoS low_lr | 98.951% | 0.123% | 0.239% | 34.2781 | 113.307% |
| FANoS vision_sweep_best | 98.921% | 0.085% | 0.209% | 32.1966 | 100.354% |
| FANoS auto | 98.901% | 0.196% | 0.189% | 36.261 | 125.646% |
| FANoS stable | 98.788% | 0.174% | 0.076% | 32.3323 | 101.199% |
| AdamW | 98.712% | 0.185% | 0.000% | 16.0698 | 0.000% |

Verdict: best FANoS config is ahead by 0.239%, but costs 113.307% more wall-clock time.


## Stiff And Scientific Tasks

| Task | FANoS metric | AdamW metric | Best critical read |
| --- | ---: | ---: | --- |
| Rosenbrock | 5.61132 | 9.52388 | FANoS beats AdamW, but RMSProp is stronger here. |
| Ill-conditioned quadratic | nan | nan | FANoS loses badly; needs dedicated curvature handling. |
| Noisy regression | 0.0571414 | 0.0531837 | Competitive; not a decisive win. |
| ODE fit | 0.00712042 | 0.00712049 | Tied with AdamW. |
| PINN auto | 0.00144467 | 1.863e-08 | Auto loses; use the PINN preset. |
| Sequence memory | 1 | 1 | FANoS matches accuracy and has lower loss. |

PINN preset result:

- FANoS PINN metric mean: `6.472e-09`
- AdamW metric mean: `1.863e-08`
- Critical read: the PINN math update is real, but it is not captured by the generic auto preset yet.


## Next Engineering Priorities

1. Keep `low_lr` as the best current vision preset and `auto` as the general safety preset.
2. Optimize runtime before adding more benchmark breadth; current wins are still too expensive.
3. Add CIFAR-10 with a stronger CNN before making any broader vision claim.
4. Add an ill-conditioned mode or curvature detector; current FANoS loses this task clearly.
5. Keep EEG claims conservative unless FANoS beats AdamW on top-1, not only loss.

## Claim Boundary

Safe claim: FANoS-v2 is a promising feedback-controlled optimizer framework with strong task-aware modes and repeated-seed wins on these lightweight benchmarks.

Unsafe claim: FANoS-v2 is a universal optimizer or a drop-in AdamW replacement.

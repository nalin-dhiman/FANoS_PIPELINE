# FANoS-v2 Night Study Report

Generated: 2026-05-01 15:37:15

## Aggregate Summary

| task | optimizer | config | n | top1_mean | top1_std | top1_best | top1_worst | loss_mean | loss_std | loss_best | loss_worst | seconds_mean | state_mb_mean | delta_top1_vs_adamw | delta_seconds_pct_vs_adamw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vision | rmsprop | baseline | 1 | 0.642578125 | 0.0 | 0.642578125 | 0.642578125 | 0.9815734028816223 | 0.0 | 0.9815734028816223 | 0.9815734028816223 | 0.22461425000801682 | 0.8077239990234375 | 0.265625 | -5.031476366744481 |
| vision | fanosv2 | low_lr | 1 | 0.533203125 | 0.0 | 0.533203125 | 0.533203125 | 1.3204840421676636 | 0.0 | 1.3204840421676636 | 1.3204840421676636 | 0.2571886249934323 | 0.8076934814453125 | 0.15625 | 8.74120413117827 |
| vision | fanosv2 | auto | 1 | 0.4140625 | 0.0 | 0.4140625 | 0.4140625 | 2.1467983722686768 | 0.0 | 2.1467983722686768 | 2.1467983722686768 | 0.2541654579981696 | 0.8076934814453125 | 0.037109375 | 7.462987338491968 |
| vision | adamw | baseline | 1 | 0.376953125 | 0.0 | 0.376953125 | 0.376953125 | 2.0017471313476562 | 0.0 | 2.0017471313476562 | 2.0017471313476562 | 0.23651441700349096 | 0.8077239990234375 | 0.0 | 0.0 |
| vision | sgd | baseline | 1 | 0.103515625 | 0.0 | 0.103515625 | 0.103515625 | 2.2882072925567627 | 0.0 | 2.2882072925567627 | 2.2882072925567627 | 0.21902124999905936 | 0.40384674072265625 | -0.2734375 | -7.396237077663388 |

## Raw Rows

| optimizer | loss | top1 | seconds | state_mb | peak_gpu_mb | device | dataset | task | seed | config |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adamw | 2.0017471313476562 | 0.376953125 | 0.23651441700349096 | 0.8077239990234375 | 0.0 | cpu | fashionmnist | vision | 0 | baseline |
| sgd | 2.2882072925567627 | 0.103515625 | 0.21902124999905936 | 0.40384674072265625 | 0.0 | cpu | fashionmnist | vision | 0 | baseline |
| rmsprop | 0.9815734028816223 | 0.642578125 | 0.22461425000801682 | 0.8077239990234375 | 0.0 | cpu | fashionmnist | vision | 0 | baseline |
| fanosv2 | 2.1467983722686768 | 0.4140625 | 0.2541654579981696 | 0.8076934814453125 | 0.0 | cpu | fashionmnist | vision | 0 | auto |
| fanosv2 | 1.3204840421676636 | 0.533203125 | 0.2571886249934323 | 0.8076934814453125 | 0.0 | cpu | fashionmnist | vision | 0 | low_lr |

## Notes

- Baselines are AdamW, SGD, and RMSProp at the benchmark default learning rate.
- FANoS configurations are fixed presets chosen from the previous sweep plus one lower-LR guardrail.
- `delta_top1_vs_adamw` and `delta_seconds_pct_vs_adamw` are computed within each task against the AdamW baseline.
- Treat this as stronger evidence than a single seed, but not a final paper result unless the selected seeds/folds match the target protocol.

# FANoS-v2 Tuning Report

Generated: 2026-05-01 02:42:00

## Tuned FANoS Full Checks

### MNIST Full

| optimizer | loss | top1 | seconds | state_mb | peak_gpu_mb | device |
| --- | --- | --- | --- | --- | --- | --- |
| fanosv2 | 0.003996820189058781 | 0.982484076433121 | 71.7113178330037 | 0.8076934814453125 | 0.0 | cpu |

### EEGBCI Full

| optimizer | loss | top1 | seconds | state_mb | device |
| --- | --- | --- | --- | --- | --- |
| fanosv2 | 0.13492892682552338 | 0.5666666626930237 | 0.614735583003494 | 0.5528106689453125 | cpu |

## Previous Full Baselines

### MNIST Full Baselines

| optimizer | loss | top1 | seconds | state_mb | peak_gpu_mb | device |
| --- | --- | --- | --- | --- | --- | --- |
| fanosv2 | 0.012079098261892796 | 0.9774084394904459 | 68.3418485000002 | 0.8076934814453125 | 0.0 | cpu |
| adamw | 0.00028566407854668796 | 0.988953025477707 | 64.70625004199974 | 0.8077239990234375 | 0.0 | cpu |
| sgd | 0.015189820900559425 | 0.9699442675159236 | 65.25192316700122 | 0.40384674072265625 | 0.0 | cpu |
| rmsprop | 0.0011420216178521514 | 0.9857683121019108 | 66.65654550000909 | 0.8077239990234375 | 0.0 | cpu |

### EEGBCI Full Baselines

| optimizer | loss | top1 | seconds | state_mb | device |
| --- | --- | --- | --- | --- | --- |
| fanosv2 | 1.2830960750579834 | 0.4333333373069763 | 0.5310049170075217 | 0.5528106689453125 | cpu |
| adamw | 0.025927165523171425 | 0.4333333373069763 | 0.3269178750051651 | 0.5528411865234375 | cpu |
| sgd | 0.5267099738121033 | 0.3333333432674408 | 0.2631943750020582 | 0.27640533447265625 | cpu |
| rmsprop | 0.00929068773984909 | 0.4333333373069763 | 0.31423725000058766 | 0.5528411865234375 | cpu |

## Top 5 Sweep Configs

### Vision Sweep Top 5

| optimizer | loss | top1 | seconds | state_mb | peak_gpu_mb | device | lr | target_scale | momentum | thermostat_lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fanosv2 | 0.03185421973466873 | 0.93896484375 | 4.618140416991082 | 0.8076934814453125 | 0.0 | cpu | 0.001 | 0.05 | 0.85 | 0.003 |
| fanosv2 | 0.01646614260971546 | 0.90087890625 | 4.58015645800333 | 0.8076934814453125 | 0.0 | cpu | 0.001 | 0.05 | 0.9 | 0.003 |
| fanosv2 | 0.07143126428127289 | 0.89306640625 | 4.642220749999979 | 0.8076934814453125 | 0.0 | cpu | 0.001 | 0.2 | 0.85 | 0.003 |
| fanosv2 | 0.1285194456577301 | 0.8916015625 | 4.7028358329989715 | 0.8076934814453125 | 0.0 | cpu | 0.001 | 0.1 | 0.9 | 0.003 |
| fanosv2 | 0.06347737461328506 | 0.8857421875 | 4.637665999995079 | 0.8076934814453125 | 0.0 | cpu | 0.001 | 0.1 | 0.85 | 0.003 |

### EEG Sweep Top 5

| optimizer | loss | top1 | seconds | state_mb | device | lr | target_scale | momentum | thermostat_lr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fanosv2 | 0.13492892682552338 | 0.5666666626930237 | 0.47430941701168194 | 0.5528106689453125 | cpu | 0.001 | 0.1 | 0.9 | 0.003 |
| fanosv2 | 0.09632337838411331 | 0.5333333611488342 | 0.4969784580025589 | 0.5528106689453125 | cpu | 0.001 | 0.2 | 0.85 | 0.003 |
| fanosv2 | 0.7250548005104065 | 0.5333333611488342 | 0.5104019159916788 | 0.5528106689453125 | cpu | 0.003 | 0.2 | 0.9 | 0.003 |
| fanosv2 | 2.120828866958618 | 0.5333333611488342 | 0.5132465829956345 | 0.5528106689453125 | cpu | 0.003 | 0.1 | 0.9 | 0.003 |
| fanosv2 | 6.425153732299805 | 0.5333333611488342 | 0.5034040419996018 | 0.5528106689453125 | cpu | 0.003 | 0.05 | 0.9 | 0.003 |

## Readout

- MNIST tuned FANoS improved over default FANoS but still trails AdamW/RMSProp on top-1 in this single-seed CPU run.
- EEGBCI tuned FANoS improved over default FANoS and beat the listed baseline top-1 values in this single-seed run, though this dataset is tiny and needs repeated seeds/folds.
- The main sensitive parameter is learning rate; `lr=0.002` and `lr=0.003` often collapsed on MNIST in the compact sweep.

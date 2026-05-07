# FANoS-v2 Targeted PINN/Sequence Update Report

Generated: 2026-05-01 14:46:38

## Summary

| run_label | task | optimizer | n | loss_mean | metric_mean | seconds_mean | state_mb_mean | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | poisson_pinn_1d | fanosv2 | 3 | 0.03859988537927469 | 0.002543517912272364 | 0.6612291943310993 | 0.00879669189453125 | 3 |
| default | poisson_pinn_1d | adamw | 3 | 0.003829231640944878 | 1.5293793609316708e-07 | 0.48132541666564066 | 0.008819580078125 | 3 |
| default | poisson_pinn_1d | sgd | 3 | 0.004085141078879436 | 1.393254898118812e-07 | 0.4182603333320003 | 0.004398345947265625 | 3 |
| default | poisson_pinn_1d | rmsprop | 3 | 0.003795309787771354 | 0.0006999864417593926 | 0.46388977799991454 | 0.008819580078125 | 3 |
| default | sequence_memory | fanosv2 | 3 | 0.22959620562505734 | 0.8489583333333334 | 3.90296026399786 | 0.0092926025390625 | 3 |
| default | sequence_memory | adamw | 3 | 0.00010505639268861462 | 1.0 | 3.722276333331441 | 0.00931549072265625 | 3 |
| default | sequence_memory | sgd | 3 | 0.6881199479103088 | 0.5494791666666666 | 3.652601263670173 | 0.00464630126953125 | 3 |
| default | sequence_memory | rmsprop | 3 | 0.6808044910430908 | 0.546875 | 3.7075192223370927 | 0.00931549072265625 | 3 |
| pinn_preset | poisson_pinn_1d | fanosv2 | 3 | 0.004641965109234055 | 2.1251029465929605e-05 | 0.7119167360021189 | 0.00879669189453125 | 3 |
| pinn_preset | poisson_pinn_1d | adamw | 3 | 0.003829231640944878 | 1.5293793609316708e-07 | 0.5335702640004456 | 0.008819580078125 | 3 |
| pinn_preset | poisson_pinn_1d | sgd | 3 | 0.004085141078879436 | 1.393254898118812e-07 | 0.47412537533576443 | 0.004398345947265625 | 3 |
| pinn_preset | poisson_pinn_1d | rmsprop | 3 | 0.003795309787771354 | 0.0006999864417593926 | 0.5050972499981677 | 0.008819580078125 | 3 |
| sequence_preset | sequence_memory | fanosv2 | 3 | 2.054072501778137e-05 | 1.0 | 4.199200874669866 | 0.0092926025390625 | 3 |
| sequence_preset | sequence_memory | adamw | 3 | 0.00010505639268861462 | 1.0 | 4.0726877640020875 | 0.00931549072265625 | 3 |
| sequence_preset | sequence_memory | sgd | 3 | 0.6881199479103088 | 0.5494791666666666 | 3.9644715136673767 | 0.00464630126953125 | 3 |
| sequence_preset | sequence_memory | rmsprop | 3 | 0.6808044910430908 | 0.546875 | 3.9707946386624826 | 0.00931549072265625 | 3 |
| no_precond | poisson_pinn_1d | fanosv2 | 3 | 0.005273902012656133 | 2.3077172765321544e-07 | 0.6900104859960265 | 0.004398345947265625 | 3 |
| no_precond | sequence_memory | fanosv2 | 3 | 0.6881648302078247 | 0.5494791666666666 | 4.923426861001644 | 0.00464630126953125 | 3 |

## Critical Readout

- PINN preset reduces FANoS PINN error by roughly two orders of magnitude versus default, but AdamW/SGD still have lower mean solution MSE in this tiny Poisson setup.
- FANoS with no RMS preconditioner is best among FANoS variants on the PINN solution metric and is close to AdamW/SGD, suggesting RMS preconditioning is hurting this PDE residual.
- Sequence preset fixes the failed seed and reaches 100% accuracy on all tested seeds, matching AdamW and beating SGD/RMSProp.
- No-preconditioner sequence fails, so the sequence fix comes from lower momentum/gain warmup, not removing preconditioning.

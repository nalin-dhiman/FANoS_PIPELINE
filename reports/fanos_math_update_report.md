# FANoS-v2 Math Update Report

Generated: 2026-05-01 14:56:55

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
| pinn_alpha0_bad | poisson_pinn_1d | fanosv2 | 3 | 30.82611083984375 | 0.19250279664993286 | 0.7282482916634763 | 0.00879669189453125 | 3 |
| pinn_alpha0_bad | poisson_pinn_1d | adamw | 3 | 0.003829231640944878 | 1.5293793609316708e-07 | 0.5295931386693459 | 0.008819580078125 | 3 |
| pinn_alpha0_bad | poisson_pinn_1d | sgd | 3 | 0.004085141078879436 | 1.393254898118812e-07 | 0.45982962533889804 | 0.004398345947265625 | 3 |
| pinn_alpha0_bad | poisson_pinn_1d | rmsprop | 3 | 0.003795309787771354 | 0.0006999864417593926 | 0.509256125005777 | 0.008819580078125 | 3 |
| pinn_alpha05 | poisson_pinn_1d | fanosv2 | 3 | 0.0022619629744440317 | 6.08940797045913e-08 | 0.6854097913310397 | 0.00879669189453125 | 3 |
| pinn_alpha05 | poisson_pinn_1d | adamw | 3 | 0.003829231640944878 | 1.5293793609316708e-07 | 0.5074896386649925 | 0.008819580078125 | 3 |
| pinn_alpha05 | poisson_pinn_1d | sgd | 3 | 0.004085141078879436 | 1.393254898118812e-07 | 0.44110708333028015 | 0.004398345947265625 | 3 |
| pinn_alpha05 | poisson_pinn_1d | rmsprop | 3 | 0.003795309787771354 | 0.0006999864417593926 | 0.4886387083339893 | 0.008819580078125 | 3 |
| sequence_warmup | sequence_memory | fanosv2 | 3 | 2.0346704938371356e-05 | 1.0 | 3.9439230416707383 | 0.0092926025390625 | 3 |
| sequence_warmup | sequence_memory | adamw | 3 | 0.00010505639268861462 | 1.0 | 3.7144344996728855 | 0.00931549072265625 | 3 |
| sequence_warmup | sequence_memory | sgd | 3 | 0.6881199479103088 | 0.5494791666666666 | 3.5786974863343253 | 0.00464630126953125 | 3 |
| sequence_warmup | sequence_memory | rmsprop | 3 | 0.6808044910430908 | 0.546875 | 3.6351690833301595 | 0.00931549072265625 | 3 |
| no_precond_old | poisson_pinn_1d | fanosv2 | 3 | 0.005273902012656133 | 2.3077172765321544e-07 | 0.6900104859960265 | 0.004398345947265625 | 3 |
| no_precond_old | sequence_memory | fanosv2 | 3 | 0.6881648302078247 | 0.5494791666666666 | 4.923426861001644 | 0.00464630126953125 | 3 |

## Critical Readout

- The new preconditioner exponent alpha matters. Full RMS was weak on PINN; alpha=0.5 makes FANoS competitive with AdamW/SGD on Poisson-1D.
- The first alpha=0 PINN preset was too conservative and underfit badly; raw/no-preconditioner can work, but needs its own LR and should remain an ablation, not the default PINN preset.
- The sequence warmup preset solved all three sequence-memory seeds and matched AdamW accuracy, while SGD/RMSProp stayed near chance.
- FANoS still runs slower than baseline optimizers in these Python-loop tests, so foreach/vectorized implementation remains an engineering priority.

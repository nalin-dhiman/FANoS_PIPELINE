# FANoS-v2 Benchmark Report

Generated: 2026-05-01 02:49:26

Duration: 0.00 minutes

## Configuration

```json
{
  "profile": "smoke",
  "data_root": "/Users/nalin/Downloads/fanos_v2/datasets",
  "results_root": "/Users/nalin/Downloads/fanos_v2/results",
  "report_root": "/Users/nalin/Downloads/fanos_v2/reports",
  "device": "cpu",
  "seed": 0,
  "optimizers": [
    "fanosv2",
    "adamw",
    "sgd",
    "rmsprop"
  ],
  "skip_download": true,
  "dry_run": true,
  "quadratic_steps": 50,
  "quadratic_dim": 64,
  "vision_epochs": 1,
  "vision_train_samples": 512,
  "vision_test_samples": 256,
  "eeg_epochs": 1,
  "eeg_train_subjects": [
    1
  ],
  "eeg_test_subject": 2,
  "eeg_runs": [
    3,
    4
  ]
}
```

## Environment

```json
{
  "python": "3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 11:23:37) [Clang 14.0.6 ]",
  "platform": "macOS-15.7.5-arm64-arm-64bit-Mach-O",
  "torch": "2.8.0",
  "cuda_available": "False",
  "cuda_device_count": "0",
  "mps_built": "True",
  "mps_available": "False",
  "requested_device": "cpu",
  "resolved_device": "cpu"
}
```

## Command Status

```json
{
  "quadratic": 0,
  "vision": 0,
  "eeg": 0
}
```

## Quadratic

| optimizer | loss | seconds | state_mb |
| --- | --- | --- | --- |
| fanosv2 | 0.0 | 0.149624457990285 | 0.015625 |
| adamw | 3.82173633017846e-12 | 0.10073679199558683 | 0.015628814697265625 |
| sgd | 0.32980817556381226 | 0.08463662498979829 | 0.0078125 |
| rmsprop | 0.0005716768791899085 | 0.09675870899809524 | 0.015628814697265625 |

## Vision

| optimizer | loss | top1 | seconds | state_mb | peak_gpu_mb | device |
| --- | --- | --- | --- | --- | --- | --- |
| fanosv2 | 0.012079098261892796 | 0.9774084394904459 | 68.3418485000002 | 0.8076934814453125 | 0.0 | cpu |
| adamw | 0.00028566407854668796 | 0.988953025477707 | 64.70625004199974 | 0.8077239990234375 | 0.0 | cpu |
| sgd | 0.015189820900559425 | 0.9699442675159236 | 65.25192316700122 | 0.40384674072265625 | 0.0 | cpu |
| rmsprop | 0.0011420216178521514 | 0.9857683121019108 | 66.65654550000909 | 0.8077239990234375 | 0.0 | cpu |

## EEGBCI Cross-Subject

| optimizer | loss | top1 | seconds | state_mb | device |
| --- | --- | --- | --- | --- | --- |
| fanosv2 | 1.2830960750579834 | 0.4333333373069763 | 0.5310049170075217 | 0.5528106689453125 | cpu |
| adamw | 0.025927165523171425 | 0.4333333373069763 | 0.3269178750051651 | 0.5528411865234375 | cpu |
| sgd | 0.5267099738121033 | 0.3333333432674408 | 0.2631943750020582 | 0.27640533447265625 | cpu |
| rmsprop | 0.00929068773984909 | 0.4333333373069763 | 0.31423725000058766 | 0.5528411865234375 | cpu |

## Notes

- Data root: `/Users/nalin/Downloads/fanos_v2/datasets`
- Results root: `/Users/nalin/Downloads/fanos_v2/results`
- This report records benchmark outputs; it does not imply FANoS-v2 wins unless the tables show it across meaningful seeds and settings.
- For paper-quality claims, rerun with multiple seeds and hardware profiling.

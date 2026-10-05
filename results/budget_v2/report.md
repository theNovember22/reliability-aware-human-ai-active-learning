# DR5 budget-preserving experiment

All variants were trained and sealed before test evaluation. These previously observed folds remain exploratory.

| Machine | Acquisition | Accuracy mean +/- SD (%) | Macro-F1 | ECE | Development queries |
|---|---|---:|---:|---:|---:|
| frozen | diverse | 60.97 +/- 1.89 | 0.6069 | 0.1201 | 40 |
| frozen | random | 60.43 +/- 2.96 | 0.6012 | 0.1040 | 40 |
| probe | diverse | 59.83 +/- 2.30 | 0.5966 | 0.1499 | 40 |
| probe | random | 58.47 +/- 3.42 | 0.5839 | 0.1322 | 40 |

SD is across fold/seed runs, not a confidence interval. Seeds share test folds.
Each test case also requires an expert response (200 per fold). The 40-query limit is for model development.
The probe uses existing known training diagnoses, never additional expert predictions.
No patient IDs or backbone provenance are available; patient/backbone leakage cannot be ruled out.

# Dataset

`DR-5/` is tracked in this repository. It contains five pairs of cached
train/test CSVs, not original retinal images.

Required columns: `Logits`, `True Label`, `Human Label`, `Feat`.
Despite its name, `Logits` contains five normalized probabilities. `Feat`
contains 512 comma-separated floats. Labels are integers 0 through 4.
Each train file has 800 rows and each test file 200 rows, with balanced classes.

The existing JSON files are retained but not required by the corrected runner.
CSV files have no sample/patient IDs, so patient-disjointness and backbone
training provenance cannot be verified from them alone.

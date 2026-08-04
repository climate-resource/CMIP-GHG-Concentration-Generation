# Notes made in 2026

## Satellite data (CO2 and CH4)

Added support for including scaled satellite data (OBS4MIPs/C3S) on top of the
ground-based observational network, for CO2 and CH4. Off by default, so
existing runs are unaffected unless you opt in.

Toggle it (and pick a fit) via env vars alongside the usual `GAS`/`RUN_ID`:

```bash
# CO2 without satellite data (default, same as before)
GAS="co2" RUN_ID="dev-test-run" bash scripts/run-helper.sh

# CO2 with satellite data, linear fit
SAT_GAS="True" SAT_FIT="LINEAR_FIT" GAS="co2" RUN_ID="dev-test-run" bash scripts/run-helper.sh

# CH4 with satellite data, linear fit
SAT_GAS="True" SAT_FIT="LINEAR_FIT" GAS="ch4" RUN_ID="dev-test-run" bash scripts/run-helper.sh
```

`SAT_FIT` must match one of the fit suffixes in the raw satellite file names
under `data/raw/scaled_sat/` (e.g. `LINEAR_FIT`, `ML_FIT`, `NONLINEAR_LAT_FIT`).
`SAT_GAS="True"` with `GAS="all"` turns satellite data on for both CO2 and CH4.

This script is not designed to be run multiple times, so what you need to do if you want to produce multiple outputs for different fits for the same gas is: delete the file doit_dev-test-run.log

## Data archival on zenodo

neem and epica data did not appear in the Zenodo bundle for some reason.
Not sure what that bug is, but it shows that our testing isn't good enough
(probably we need to test that you can download the zenodo bundle
then run offline and still reproduce everything).

TODO was too depressing to look at.
We should return to that at some point.
Not today.

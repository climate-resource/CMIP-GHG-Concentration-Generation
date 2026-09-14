# Train/test split evaluation

An out-of-sample check on the ground-based observational network: does the
pipeline's **final** gridded/input4MIPs result reproduce ground-based
observations it never saw?

## Methodology

1. **Split.** For CO2 and CH4 independently, a target fraction
   (`test_split_fraction`, `0.2` in `scripts/write-eval-split-config.py`) of
   NOAA surface-flask rows (one row = one station-month observation) is
   randomly held out before binning. NOAA in-situ rows - and for CH4 also
   AGAGE/GAGE/ALE rows - are **never** eligible for holdout; they always
   stay in training. This happens in
   `1100_ch4_bin-observational-network.py`/`1200_co2_bin-observational-network.py`,
   right before binning (`local.binning.calculate_bin_averages`), so the
   held-out rows never influence binning, spatial interpolation, or anything
   downstream.

   The split is stratified by `local.train_test_split.stratified_test_split`
   at the (year, month, lat_bin, lon_bin) level - one group is exactly one
   spatial bin's contribution to one month, i.e. one input point to
   `local.binned_data_interpolation.interpolate`'s `scipy.interpolate.griddata`
   call. Within a group, eligible (flask) rows can only be held out down to
   fully emptying the group if a non-eligible ("backbone": in-situ/AGAGE/GAGE/ALE)
   row is also present there; otherwise at least 1 eligible row always stays
   in training. This guarantees the split can never *depopulate* a spatial
   bin that had data - only reduce how many stations get averaged together
   within it - so it can't shrink a month's spatial coverage/convex hull
   below what the full dataset already achieves.

   **Why this matters, and why it isn't a true 80/20 split in practice:** an
   earlier, simpler version of this split (plain random rows, then
   row-count-per-month floors) reliably broke the pipeline - `griddata`
   failed to fill the grid for scattered months, those months got dropped
   from `observational_network_interpolated_file` entirely, and enough of
   them landing in the same year made that year's spacing non-uniform, which
   crashes `1202_co2_.../1102_ch4_...`'s seasonality decomposition
   (`local/mean_preserving_interpolation/lai_kaplan.py`:
   `NotImplementedError: Non-uniform spacing in x`). The
   never-depopulate-a-bin rule above fixes that, but most 15x60 degree
   spatial bins only have 1-4 contributing stations in a given month, so in
   practice only a small fraction of rows can safely be held out without
   emptying a bin - **for CO2, the actual achieved fraction was ~1-2%, not
   20%**, even after also restricting the eligible pool to flask-only rows
   (which didn't help much either - flask and in-situ rows rarely share the
   same spatial bin, so protecting in-situ mostly just removes rows from the
   eligible pool without unlocking many bins to give up all their flask
   data). This is a genuine property of the ground network's spatial
   sparsity, not a bug - treat the true "20%" in `test_split_fraction` as a
   target/upper bound, and check the actual achieved count printed by
   `1100`/`1200` (or count `held_out_test_data_file`'s rows) for the real
   number.

2. **Train.** The pipeline is run as normal, all the way through
   `crunch_grids` and `write_input4mips` (unlike an earlier version of this
   evaluation, which stopped at the intermediate
   `observational_network_interpolated_file` to save time) - but the
   ground-based observational network only ever contains the training rows.
   Satellite data is off by default; see "Comparing satellite fits" below
   for turning it on.

3. **Test.** The held-out rows are written straight to
   `held_out_test_data_file` (see
   `src/local/config/calculate_{ch4,co2}_monthly_15_degree.py`) and are not
   touched again by the pipeline.

4. **Reproducibility.** The split is seeded
   (`test_split_seed`, fixed to `42` in `scripts/write-eval-split-config.py`)
   and independent of satellite data (the split happens before satellite
   data is combined with the ground network - see
   `1101_ch4_.../1201_co2_..._interpolate-observational-network.py`), so
   re-running the config-writing script (with any `SAT_GAS`/`SAT_FIT`
   setting) and the pipeline reproduces the exact same held-out test set.
   That's what makes it usable as a fixed benchmark across a no-satellite
   baseline run and several satellite-fit runs.

This mechanism is opt-in and off by default (`test_split_fraction=None`) -
the main `dev-config.yaml`/`ci-config.yaml` pipelines are unaffected and
always use 100% of the observational network.

## Producing a run to evaluate

```bash
# 1. Write the eval-split config (CO2 + CH4, no satellite by default, seed 42)
pixi run python scripts/write-eval-split-config.py

# 2. Reuse the main dev run's raw/interim data and dependency database, so
#    retrieval/processing of NOAA/AGAGE/ice-core data isn't redone (these
#    steps don't depend on the split - it only changes what `1100`/`1200` do
#    with already-processed per-station data). Requires an existing
#    `output-bundles/dev-test-run/` from a normal
#    `pixi run python scripts/write-config.py` + `doit` run.
mkdir -p output-bundles/dev-test-run-eval-split/data/processed
cp -r output-bundles/dev-test-run/data/raw output-bundles/dev-test-run-eval-split/data/
cp -r output-bundles/dev-test-run/data/interim output-bundles/dev-test-run-eval-split/data/
cp output-bundles/dev-test-run/data/processed/dependencies.db output-bundles/dev-test-run-eval-split/data/processed/
# Do NOT copy data/processed/esgf-ready - that's write_input4mips's own
# output directory and this run needs to write its own, separate from the
# full-data run's.

# 3. Run doit against the eval-split config, reusing the main run's doit DB
#    so shared upstream tasks are recognised as already done.
DOIT_CONFIGURATION_FILE=eval-split-config-absolute.yaml \
DOIT_RUN_ID=dev-test-run-eval-split \
DOIT_DB_BACKEND=json \
DOIT_DB_FILE=doit-db-dev.json \
pixi run doit run --verbosity=2 \
  "${PWD}/output-bundles/dev-test-run-eval-split/data/processed/esgf-ready/co2_input4MIPs_esgf-ready.complete" \
  "${PWD}/output-bundles/dev-test-run-eval-split/data/processed/esgf-ready/ch4_input4MIPs_esgf-ready.complete"
```

Targeting the two `*_esgf-ready.complete` files runs everything needed for
both gases end to end (doit resolves the whole dependency chain); `doit
list` shows individual task names if you'd rather target one gas or one
step at a time. Each gas takes roughly 1-5 minutes (CH4 is slower - it also
smooths the Law Dome ice-core record with a 250-draw bootstrap).

### Comparing satellite fits

Because the held-out test set doesn't depend on satellite data (see
"Reproducibility" above), the same `run_id` can accumulate a no-satellite
baseline plus several satellite-fit variants side by side, and
`evaluate_co2_satellite_fits.py` will automatically discover and compare
whichever variants are available
(`local.diagnostics_reporting.discover_gridded_files`). To add a fit:

```bash
SAT_GAS=True SAT_FIT=LINEAR_STD_WEIGHT_FIT pixi run python scripts/write-eval-split-config.py
# then re-run doit as in step 3 above
```

`write_input4mips`'s output filenames carry a `SAT_<fit>` suffix (or none,
for the baseline) - but **the write itself happens at the *plain*,
unsuffixed path first, and the suffix is added by renaming afterwards**
(see `4010_write-input4mips-files.py`). This means the *first* write into a
fresh day's version folder after a baseline run physically overwrites the
baseline's file before renaming its own content into place - the baseline
is not preserved automatically. Once a fit has been written (and renamed to
its own suffixed name), later fits no longer collide with it, since the
plain path is empty again going into the next run.

**Practical consequence: write every satellite fit before (re-)writing the
no-satellite baseline**, and if you need the baseline to still exist
afterwards, run `pixi run python scripts/write-eval-split-config.py` (no
`SAT_GAS`) one more time at the end, after all fits are done, so it lands
safely on the now-empty plain path without clobbering anything. This is
exactly how the CO2 baseline + 5 fits currently on disk were produced: 5
fits first, baseline regenerated last.

## Notebooks

- `evaluate_co2_train_test_split.py` / `evaluate_ch4_train_test_split.py`
- `evaluate_co2_satellite_fits.py` - compares the no-satellite baseline
  against all five `STD_WEIGHT` satellite fits (`LINEAR`, `LINEAR_LAT`,
  `LINEAR_SEASONAL`, `LINEAR_SEASONAL_LAT`, `NONLINEAR_LAT`) against the
  same held-out test set as the baseline notebooks, using
  `discover_gridded_files` to find every variant written under the
  `run_id` in one pass - see "Comparing satellite fits" above for how
  those runs are produced. No CH4 equivalent yet.

All notebooks here are hand-maintained (not auto-generated) and not
registered with `doit` - open and run manually. Each loads the
`dev-test-run-eval-split` config.

The two `evaluate_{co2,ch4}_train_test_split.py` notebooks produce:

- **Held-out test data station locations**: a world-map scatter of the
  unique (latitude, longitude) points in the held-out rows
  (`held_out_test_data_file`), using the same `geopandas`/Natural Earth
  world-outline pattern as the network-overview notebooks (e.g.
  `notebooks/001y_process-noaa-data/0019_noaa-network-overview.py`). Since
  only NOAA surface-flask rows are ever held out, this should look like a
  scattered subsample of the flask network's station locations
  specifically, not concentrated in any particular region.
- **Two pipeline results**, compared against the held-out data throughout:
  - **A. Final gridded product** - the actual final input4MIPs output (the
    `gnz`, 15-degree-latitude zonal-mean grid written by
    `40yy_write-input4mips`), loaded via
    `local.diagnostics_reporting.discover_gridded_files`/`load_concatenated_gridded`
    (the same loader
    `notebooks/diagnostics/sat_period/CO2/compare_nosat_vs_sat_period_nosat.py`
    uses for its own "Final gridded output diff" section) - the pipeline's
    real deliverable.
  - **B. Interpolated ground-network grid** - the intermediate
    `observational_network_interpolated_file` (post `1201`/`1101`, pre
    `1202`/`1102` downstream processing) - lets you see whether the extra
    downstream steps (seasonality/latitudinal-gradient reconstruction,
    gridding) help or hurt agreement with held-out data, relative to the
    raw interpolated network. Unlike `1202_co2_.../1102_ch4_...` (which
    drops any *year* containing an incomplete month, since its output feeds
    into steps that need complete years), this keeps every month that has
    *any* spatial coverage.
- **Monthly trend plots** (one per pipeline result, both kept side by
  side): each pipeline result plotted against the held-out test data's
  monthly trend, markers only (no connecting line), the pipeline result
  drawn on top (higher `zorder`). The test trend is the held-out rows,
  binned the same way the pipeline bins its training data
  (`local.binning.calculate_bin_averages` - stations equally weighted
  within a spatial bin), then cos(latitude)-weighted across whichever bins
  happen to have test data in a given month. This is deliberately *not*
  run through spatial interpolation (`griddata`) - the test rows are a
  sparse subset that often won't meet the pipeline's own
  minimum-points-per-month threshold for interpolation, and reusing that
  machinery on held-out data would blur the "did we reproduce points we
  never saw" comparison this evaluation is for.
- **Agreement (global mean)**: pipeline and test monthly values matched by
  (year, month) (`local.evaluation_metrics.match_monthly`/
  `summarise_agreement`), with residuals plotted over time and summarised
  as bias, MAE, RMSE, R² and correlation, for both pipeline results side by
  side. **Read this with the sampling-geometry caveat below in mind.**

This is the number(s) to compare across a no-satellite baseline run and
satellite-fit runs that share the same held-out test set - see
`evaluate_co2_satellite_fits.py` and "Comparing satellite fits" above.

A location-matched variant of this agreement table also exists
(`local.evaluation_metrics.match_pointwise` - for each held-out bin, look
up the pipeline's own value at that *same* location/month instead of
comparing two differently-sampled "global means") but isn't wired into the
current notebooks. It's worth knowing it's there: see the caveat directly
below for why it can matter a lot, and reintroduce it (call `match_pointwise`
the same way `match_monthly` is called, then diff the two tables) if the
global-mean numbers ever look suspicious.

## Known limitations / caveats

- **The achieved test fraction is much smaller than `test_split_fraction`
  and varies by gas/network density.** See "Split" above - this is the
  most important caveat, not a minor footnote. Always check the actual
  held-out row count for the run you're looking at, don't assume "20%".
- **The global-mean agreement table can show a large, misleading "bias"
  that isn't real pipeline error.** The held-out sample's spatial coverage
  is sparse and can be geographically skewed (e.g. CH4's held-out rows, in
  one run, were concentrated in the tropical Pacific/East Asia/North
  America with no Southern Hemisphere or high-latitude points at all). For
  a gas with strong spatial gradients (CH4 much more than CO2), averaging
  only over that skewed sample and comparing it to the pipeline's *true*
  global mean manufactures a large apparent bias with no real pipeline
  error behind it - in one run this inflated CH4's apparent bias from a
  genuine ~+4 ppb (location-matched, see above) to a spurious ~-32 ppb
  (global-mean). If a bias looks surprisingly large, compute the
  location-matched version (`local.evaluation_metrics.match_pointwise`,
  not currently wired into the notebooks - see "Notebooks" above) before
  concluding the pipeline itself is at fault.
- **The two pipeline results aren't computed identically.** Pipeline A has
  been through spatial interpolation (filling gaps) and everything
  downstream of it (seasonality/latitudinal-gradient reconstruction,
  gridding); pipeline B has only been through spatial interpolation; the
  test-data trend has been through neither (see above). This is
  intentional - reproducing the pipeline's *actual output* isn't the same
  test as reproducing raw held-out observations - but it means none of the
  three are perfectly apples-to-apples with each other, especially in
  months/years where test data coverage is very sparse.

# EO-update run

The EO-update run produces CO2 and CH4 files that include scaled satellite
data. It runs in parallel to the baseline run and writes to its own folder, so
the baseline is never affected.

| | Baseline | EO-update |
|---|---|---|
| Config | `dev-config.yaml` (from `scripts/write-config.py`) | `eo-update-config.yaml` (from `scripts/write-eo-update-config.py`) |
| Run ID / output | `output-bundles/dev-test-run/` | `output-bundles/eo-update/` |
| Satellite data | off (unless `SAT_GAS=True`) | on, weighted by inverse retrieval uncertainty |
| CO2 fit | - | `LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT` |
| CH4 fit | - | `NONLINEAR_LAT_STD_WEIGHT_FIT` |
| CV source / `source_id` | `CR-CMIP-testing` (remote) | `CR-CMIP-EO-update` (generated into `output-bundles/eo-update/cvs/`) |

## Run the baseline

```bash
GAS="co2" RUN_ID="dev-test-run" bash scripts/run-helper.sh   # one gas
GAS="all" RUN_ID="dev-test-run" bash scripts/run-helper.sh   # all gases
```

`run-helper.sh` regenerates `dev-config-absolute.yaml` and uses
`doit-db-dev.json`.

## Run the EO-update

1. **Set the metadata** in the "Metadata" block of
   `scripts/write-eo-update-config.py` (`DOI`, `COMMENT`, `SOURCE_ID_ENTRY` etc.).
   Contact and further-info URL are set in
   `SOURCE_ID_ENTRY` in the same block. The CV folder is regenerated (needs internet) each time the script runs.

2. **Write the config:**

   ```bash
   pixi run python scripts/write-eo-update-config.py
   ```

   This writes `eo-update-config.yaml` and `eo-update-config-absolute.yaml`.
   Re-run it after any metadata change.

3. **Reuse the baseline's upstream work** (one-off), so raw downloads and
   NOAA/AGAGE/ice-core processing aren't redone:

   ```bash
   mkdir -p output-bundles/eo-update/data
   cp -r output-bundles/dev-test-run/data/raw output-bundles/dev-test-run/data/interim \
       output-bundles/eo-update/data/
   cp doit-db-dev.json doit-db-eo-update.json
   ```

   Also merge the baseline's dependency sources into the EO-update DB,
   otherwise the write step fails with `AssertionError: ['HadCRUT5']`:

   ```bash
   python3 - <<'EOF'
   import sqlite3
   c = sqlite3.connect("output-bundles/eo-update/data/processed/dependencies.db")
   c.execute("attach 'output-bundles/dev-test-run/data/processed/dependencies.db' as dev")
   c.execute("insert or ignore into source select * from dev.source")
   c.execute("insert or ignore into dependencies select * from dev.dependencies")
   c.commit()
   EOF
   ```

4. **Run doit** (a separate DB file keeps the baseline's doit state safe):

   ```bash
   DOIT_CONFIGURATION_FILE=eo-update-config-absolute.yaml DOIT_RUN_ID=eo-update \
       DOIT_DB_BACKEND=json DOIT_DB_FILE=doit-db-eo-update.json \
       pixi run doit --verbosity=2 -n 4 \
       "${PWD}/output-bundles/eo-update/data/processed/esgf-ready/co2_input4MIPs_esgf-ready.complete"
   ```

   Repeat with `ch4_...` for methane.

Output files appear under
`output-bundles/eo-update/data/processed/esgf-ready/input4MIPs/.../CR-CMIP-EO-update/`.
Check that `satellite_data_source`, `satellite_data_reference`, `satellite_fit`, `satellite_weighting` and your metadata are
present in the attributes.

## Rerunning after metadata changes

`doit` does not track the CV contents (`SOURCE_ID_ENTRY`, `ACTIVITY_ID_ENTRY`),
so after changing only those, delete
`output-bundles/eo-update/data/processed/esgf-ready/<gas>_input4MIPs_esgf-ready.complete`
before rerunning step 4 to force the files to be rewritten.

## Status

CO2 and CH4 have both been run end to end (2026-10-08) and the files carry the
expected metadata.

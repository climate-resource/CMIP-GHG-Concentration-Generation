# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CO2 - check gridded output across all satellite fits
#
# Cleaned-up successor to `check_CO2_output_sat.py`: instead of a hardcoded,
# manually-updated `v<date>` path, this discovers the most recent (or a
# chosen) version folder under the input4MIPs output tree itself, then diffs
# every `_SAT_<FIT>.nc` output it finds there against the no-satellite
# baseline output in that same folder.
#
# This needs the `40yy_write-input4mips` step to have actually been run (for
# the fits you want to compare) - it's the final gridding/writing stage,
# downstream of everything the `compare_co2_<FIT>.py` notebooks in this
# folder look at. If nothing shows up below, that's why.
#
# Note: the `gnz` product used here is a **zonal mean** (dims `time, lat` -
# no `lon`), and each fit is written as several files chunked by time range
# (e.g. `000101-099912`, `100001-174912`, `175001-202212`), not one file per
# fit. This notebook concatenates those chunks back into one continuous
# timeseries per fit before diffing.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import local.diagnostics_reporting as report

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
run_id: str = "dev-test-run"
gas: str = "co2"
version: str | None = None  # None picks the most recently created v* folder found
output_bundles_root: str = "../../../output-bundles"

# %% [markdown]
# ## Find the version folder to check

# %%
esgf_ready_dir = report.get_esgf_ready_gas_dir(Path(output_bundles_root), run_id, gas)
esgf_ready_dir

# %%
version_dir = report.find_version_dir(esgf_ready_dir, version)
version_dir


# %% [markdown]
# ## Discover available fits
#
# Looks for `{gas}_input4MIPs_..._SAT_<FIT>.nc` files in the chosen version
# folder and infers `<FIT>` (and the time-range chunk boundaries) from the
# filename - nothing here is hardcoded, so this picks up new fits (and any
# change in how files get chunked) automatically. Discovery/loading logic
# lives in `local.diagnostics_reporting` so it stays in sync with the
# "Final gridded output diff" section of the `compare_co2_<FIT>.py`
# notebooks in this folder, which reuses the exact same functions.

# %%
baseline_chunks, fit_chunks = report.discover_gridded_files(version_dir, gas)
print(f"Baseline (no satellite) chunks: {[p.name for p in baseline_chunks]}")
print(f"Found {len(fit_chunks)} fit(s): {sorted(fit_chunks)}")

baseline_da = report.load_concatenated_gridded(baseline_chunks, gas)

# %% [markdown]
# ## Baseline
#
# **Derived in:** `40yy_write-input4mips` (the final gridding/ESGF-ready
# output step, downstream of everything the `compare_co2_<FIT>.py` notebooks
# in this folder look at).
#
# The no-satellite-data output, for reference: a Hovmöller-style view (time
# on the x-axis, latitude on the y-axis) of the full record.

# %%
if baseline_da is not None:
    fig, ax = plt.subplots(figsize=(10, 3))
    baseline_da.plot(ax=ax, x="time", y="lat")
    ax.set_title(f"{gas.upper()} - no satellite data")
    plt.show()
else:
    print("No baseline (no-satellite) output found in this version folder yet.")

# %% [markdown]
# ## Per-fit diffs over time
#
# **Derived in:** `40yy_write-input4mips`.
#
# `<fit> minus baseline`, full record. Satellite data only covers
# 2003-2023, so a real effect should be concentrated there (plus a much
# smaller amount bleeding backwards through the historical-extension
# harmonisation steps) - a fit with a diff pattern that *doesn't* look like
# that is worth a closer look.


# %%
diffs = {}
if baseline_da is not None:
    for fit, chunks in fit_chunks.items():
        fit_da = report.load_concatenated_gridded(chunks, gas)
        if fit_da is not None:
            diffs[fit] = report.gridded_diff_from_baseline(baseline_da, fit_da)

for fit, diff in diffs.items():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    diff.plot(ax=ax, x="time", y="lat", cmap="RdBu_r")
    ax.set_title(fit)
    plt.show()

# %% [markdown]
# ## All fits together
#
# **Derived in:** `40yy_write-input4mips`.
#
# Time-mean diff (over the full record) by latitude, one line per fit, so
# the *relative* size and shape of each fit's effect is directly comparable
# in one view.

# %%
if diffs:
    fig, ax = plt.subplots(figsize=(8, 5))

    for fit, diff in diffs.items():
        diff.mean("time").plot(ax=ax, label=fit)

    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_title(f"{gas.upper()} - time-mean diff from baseline, by latitude")
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary
#
# Synthesised from the per-fit diffs above (not derived from a single
# pipeline step). Max/mean absolute difference from baseline over the full record, per fit -
# a quick way to rank fits by how much they change the final product without
# eyeballing every plot above.

# %%
summary = pd.DataFrame(
    [
        {
            "fit": fit,
            "max_abs_diff": float(np.abs(diff).max()),
            "mean_abs_diff": float(np.abs(diff).mean()),
        }
        for fit, diff in diffs.items()
    ],
    columns=["fit", "max_abs_diff", "mean_abs_diff"],
).set_index("fit")
summary.sort_values("max_abs_diff", ascending=False)

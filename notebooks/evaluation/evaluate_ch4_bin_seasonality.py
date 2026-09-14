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
# # CH4: bin-level satellite coverage and seasonality
#
# Zooms in from the grid-wide coverage picture (`plot_spatial_bin_coverage`)
# to individual 15x60 degree bins: which specific bins does satellite data
# actually rescue from being under-observed, and how does the *seasonal
# cycle itself* (not just how well it's observed) compare in those bins vs.
# the no-satellite baseline?
#
# Two bin groups are compared throughout:
#
# - **Filled by satellite**: bins where `bin_fraction_months_populated_satellite_period`
#   rises by more than `threshold` once satellite data is switched on - these
#   are the bins satellite data is doing real work in.
# - **Already covered without satellite**: bins that were already above
#   `threshold` on the ground network alone - these can't have moved much
#   regardless of satellite data, so they're a natural control group: any
#   seasonal difference seen here isn't attributable to satellite coverage
#   filling gaps.
#
# Needs `1101_ch4_interpolate-observational-network` to have been run for
# both configurations under the same `run_id` (see `diagnostics_root`
# below) - that's what saves both the coverage-fraction maps and the
# per-bin `bin_monthly_climatology_satellite_period` used here.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import local.diagnostics_reporting as report

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
run_id: str = "dev-test-run"
gas: str = "ch4"
satellite_fit: str = "NONLINEAR_LAT_STD_WEIGHT_FIT"
diagnostics_root: str = "../../output-bundles/{run_id}/data/diagnostics"
threshold: float = 0.8

# %%
diagnostics_root_path = Path(diagnostics_root.format(run_id=run_id))
satellite_suffix = f"SAT_{satellite_fit}"
suffixes = ("nosat", satellite_suffix)
labels = {"nosat": "No satellite data", satellite_suffix: f"Satellite data\n({satellite_fit})"}
colors = {"nosat": "tab:blue", satellite_suffix: "tab:orange"}
diagnostics_root_path

# %% [markdown]
# ## 1. Grid-wide coverage
#
# **Derived in:** `1101_ch4_interpolate-observational-network`
#
# Same picture as in the `compare_ch4_*.py` diagnostics notebooks: fraction
# of satellite-period months each bin has a real observation in, per
# configuration, plus the difference map. The bin selections below are
# drawn directly from this difference map (and the baseline panel).

# %%
report.coverage_table(diagnostics_root_path, gas, suffixes, labels)

# %%
report.plot_spatial_bin_coverage(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 2. Selecting bins
#
# `select_bins_by_satellite_fill` returns `None` if either configuration's
# coverage map is missing (e.g. `{satellite_fit}` hasn't been run yet under
# this `run_id`).

# %%
bin_selection = report.select_bins_by_satellite_fill(
    diagnostics_root_path, gas, suffixes, threshold=threshold
)
mask_filled_by_satellite, mask_already_covered = bin_selection if bin_selection is not None else (None, None)
if bin_selection is None:
    print(f"Missing coverage diagnostics for one or both of {suffixes} - nothing to select.")

# %% [markdown]
# ### 2a. Bins satellite data fills in
#
# Bins where the satellite-period coverage fraction rises by more than
# `threshold` once satellite data is switched on.

# %%
bins_filled_by_satellite: list[tuple[float, float]] = (
    report.plot_bin_selection_map(
        mask_filled_by_satellite,
        f"{gas.upper()} - bins where satellite data raises coverage by >{threshold} ({satellite_fit})",
    )
    if mask_filled_by_satellite is not None
    else []
)

# %% [markdown]
# ### 2b. Bins already covered without satellite data
#
# Bins where the no-satellite baseline's own coverage fraction is already
# above `threshold` - the control group.

# %%
bins_already_covered: list[tuple[float, float]] = (
    report.plot_bin_selection_map(
        mask_already_covered,
        f"{gas.upper()} - bins already >{threshold} covered without satellite data",
    )
    if mask_already_covered is not None
    else []
)

# %% [markdown]
# ## 3. Seasonality in the selected bins
#
# **Derived in:** `1101_ch4_interpolate-observational-network`
#
# Mean-by-calendar-month value of the actual (non-interpolated) ground/satellite
# observations in each selected bin, restricted to the satellite-period
# years - i.e. the real seasonal cycle behind the coverage numbers above, not
# a fitted/extrapolated one.

# %% [markdown]
# ### 3a. Bins filled by satellite data

# %%
report.plot_bin_seasonality(
    diagnostics_root_path,
    gas,
    suffixes,
    labels,
    colors,
    bins_filled_by_satellite,
    "bins filled by satellite data",
)

# %% [markdown]
# ### 3b. Bins already covered without satellite data

# %%
report.plot_bin_seasonality(
    diagnostics_root_path,
    gas,
    suffixes,
    labels,
    colors,
    bins_already_covered,
    "bins already covered without satellite data",
)

# %% [markdown]
# ## 4. Seasonality after interpolation
#
# **Derived in:** `1101_ch4_interpolate-observational-network`
#
# Same comparison as section 3, but from `bin_monthly_climatology_satellite_period_interpolated`
# - computed *after* `griddata` spatial interpolation has filled every bin,
# rather than straight from the raw ground/satellite observations. No gaps
# (interpolation guesses every bin/month it covers) and no
# ground/satellite observation-count boxes, since those describe the raw
# data behind the curve, not `griddata`'s guesses.

# %% [markdown]
# ### 4a. Bins filled by satellite data

# %%
report.plot_interpolated_bin_seasonality(
    diagnostics_root_path,
    gas,
    suffixes,
    labels,
    colors,
    bins_filled_by_satellite,
    "bins filled by satellite data",
)

# %% [markdown]
# ### 4b. Bins already covered without satellite data

# %%
report.plot_interpolated_bin_seasonality(
    diagnostics_root_path,
    gas,
    suffixes,
    labels,
    colors,
    bins_already_covered,
    "bins already covered without satellite data",
)

# %% [markdown]
# ## 5. Global vs. satellite-filled-bin seasonality
#
# **Derived in:** `1101_ch4_interpolate-observational-network`
#
# One configuration only (satellite data on): the cos(latitude)-weighted
# global-mean seasonal cycle (across every grid bin) against the same
# weighted-mean seasonal cycle restricted to just the bins satellite data
# fills in (section 2a) - does that specific region's seasonal cycle look
# like a scaled-down version of the global one, or genuinely different in
# shape/timing?

# %%
report.plot_global_vs_bin_group_seasonality(
    diagnostics_root_path,
    gas,
    satellite_suffix,
    labels[satellite_suffix],
    bins_filled_by_satellite,
    "bins filled by satellite data",
)

# %% [markdown]
# todo: get rmse between with and without satellite data, and plot it in the global map

# %%

# %%

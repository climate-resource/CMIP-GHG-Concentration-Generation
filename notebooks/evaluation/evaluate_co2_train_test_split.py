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
# # CO2: train/test split evaluation
#
# The goal of this notebook is to set a baseline evaluation of the
# output of the CMIP-GHG-Concentration-Generation pipeline against a
# subset of the ground based data used.
#
# This is to allow to quantify future results of evaluation of impact of adding
# satellite data to said pipeline.
#
# The data is split in 80% train data and 20% test data, the latter being
# taken completely from flask data, since taking it from monthly
# in-situ observation can remove enough months from the ground data to
# cause a crash.
#
# In this notebook we compare the test data with the gridded and interpolated
# data and with the final product of the pipeline, in monthly means.
#
# See `README.md` in this folder for more info

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cftime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.binning
import local.diagnostics_reporting as report
import local.evaluation_metrics as em
import local.xarray_space
from local.config import load_config_from_file

# %%
# `load_config_from_file` needs `ppm`/`ppb` registered in the pint
# application registry to deserialise `smooth_law_dome_data`'s noise
# settings, even though nothing else in this notebook uses pint directly.
pint.set_application_registry(openscm_units.unit_registry)


# %%
def plot_pipeline_vs_test(pipeline_da: xr.DataArray, pipeline_label: str, title_suffix: str) -> None:
    """Plot one pipeline result against the held-out test data"""
    _fig, ax = plt.subplots(figsize=(12, 5))
    # Plotted first so xarray/nc-time-axis registers a cftime unit
    # converter on the x-axis before the plain `ax.plot` call below (which
    # also uses cftime values) needs it - `zorder` (not call order)
    # controls what ends up drawn on top.
    pipeline_da.plot(  # type: ignore
        ax=ax,
        color="tab:blue",
        marker="o",
        linestyle="none",
        markersize=5,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label=pipeline_label,
        zorder=3,
    )
    ax.plot(
        test_monthly["time"],
        test_monthly["value"],
        color="tab:orange",
        marker="o",
        linestyle="none",
        markersize=5,
        alpha=0.6,
        label="Held-out test data",
        zorder=2,
    )
    ax.set_title(f"{gas.upper()} - monthly, global-mean trend: {title_suffix}")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{gas.upper()} [{test_unit}]")
    ax.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
gas: str = "co2"
step: str = "calculate_co2_monthly_fifteen_degree_pieces"
config_file: str = "../../eval-split-config-absolute.yaml"
step_config_id: str = "only"
run_id: str = "dev-test-run-eval-split"
output_bundles_root: str = "../../output-bundles"

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)
config_retrieve_misc = get_config_for_step_id(config=config, step="retrieve_misc_data", step_config_id="only")
output_bundles_root_path = Path(output_bundles_root)

if config_step.held_out_test_data_file is None:
    msg = (
        "`held_out_test_data_file` is not set for this config - "
        "was `test_split_fraction` set when this config was written? "
        "See `scripts/write-eval-split-config.py`."
    )
    raise ValueError(msg)


# %% [markdown]
# ## Test data

# %% [markdown]
# Randomly held out from NOAA flask data (target: `test_split_fraction` of
# all eligible flask rows, subject to a per-bin safety cap - see
# `local.train_test_split.stratified_test_split`).

# %%
test_data = pd.read_csv(config_step.held_out_test_data_file)
test_split_stats = pd.read_csv(
    config_step.held_out_test_data_file.with_name(
        f"{config_step.gas}_observational-network_test-holdout-stats.csv"
    )
).iloc[0]
print(
    f"{len(test_data)} held-out rows: "
    f"{len(test_data) / test_split_stats['n_eligible_flask_total']:.1%} of eligible flask data, "
    f"{len(test_data) / test_split_stats['n_all_total']:.1%} of all ground-network datapoints"
)
test_data

# %%
countries = gpd.read_file(
    config_retrieve_misc.natural_earth.raw_dir / config_retrieve_misc.natural_earth.countries_shape_file_name
)

fig, ax = plt.subplots(figsize=(10, 5))
countries.plot(color="lightgray", ax=ax)
test_data[["longitude", "latitude"]].drop_duplicates().plot(
    x="longitude",
    y="latitude",
    kind="scatter",
    ax=ax,
    color="tab:orange",
    zorder=3,
)
ax.set_xlim((-180.0, 180.0))
ax.set_ylim((-90.0, 90.0))
ax.set_title(f"{gas.upper()} - test data station locations")
plt.tight_layout()
plt.show()

# %%
# Training locations are only available at grid-bin resolution (the
# production pipeline only saves the bin-averaged training data, not raw
# per-station rows), so this is coarser than the station-level test-data
# scatter above - it shows which 15x60 degree bins have training data, not
# exact station coordinates.
train_bin_averages = pd.read_csv(config_step.processed_bin_averages_file)

fig, ax = plt.subplots(figsize=(10, 5))
countries.plot(color="lightgray", ax=ax)
train_bin_averages[["lon_bin", "lat_bin"]].drop_duplicates().plot(
    x="lon_bin",
    y="lat_bin",
    kind="scatter",
    ax=ax,
    color="tab:blue",
    label="Training data (grid bins)",
    zorder=2,
)
test_data[["longitude", "latitude"]].drop_duplicates().plot(
    x="longitude",
    y="latitude",
    kind="scatter",
    ax=ax,
    color="tab:orange",
    label="Test data (stations)",
    zorder=3,
)
ax.set_xlim((-180.0, 180.0))
ax.set_ylim((-90.0, 90.0))
ax.set_title(f"{gas.upper()} - test vs. train data station locations")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Comparison with interpolated binned data

# %%
interpolated_da: xr.DataArray = xr.load_dataarray(  # type: ignore
    config_step.observational_network_interpolated_file, use_cftime=True
)
pipeline_interpolated_allyears = local.xarray_space.calculate_global_mean_from_lon_mean(
    interpolated_da.mean(dim="lon")
)

# %%
# Weighted average of test data
test_data_with_bins = local.binning.add_lat_lon_bin_columns(test_data)
test_bin_averages = local.binning.calculate_bin_averages(test_data_with_bins)
test_bin_averages

# %%
test_unit = test_bin_averages["unit"].unique()
if len(test_unit) != 1:
    msg = f"Expected a single unit, got {test_unit=}"
    raise AssertionError(msg)
test_unit = test_unit[0]

test_lat_mean = test_bin_averages.groupby(["year", "month", "lat_bin"])["value"].mean().reset_index()
test_lat_mean["cos_lat"] = np.cos(np.deg2rad(test_lat_mean["lat_bin"]))
test_lat_mean["weighted_value"] = test_lat_mean["value"] * test_lat_mean["cos_lat"]

test_monthly = test_lat_mean.groupby(["year", "month"])[["weighted_value", "cos_lat"]].sum()
test_monthly["value"] = test_monthly["weighted_value"] / test_monthly["cos_lat"]
test_monthly = test_monthly.reset_index()
test_monthly["time"] = [
    cftime.datetime(int(y), int(m), 15) for y, m in zip(test_monthly["year"], test_monthly["month"])
]
test_monthly

# %%
# Select time period covered by ground obs
test_period = slice(
    cftime.datetime(int(test_monthly["year"].min()), 1, 1),
    cftime.datetime(int(test_monthly["year"].max()), 12, 31),
)

pipeline_interpolated = pipeline_interpolated_allyears.sel(time=test_period)

# %%
# Plot test data against interpolated binned train ground data
plot_pipeline_vs_test(
    pipeline_interpolated, "Pipeline result (interpolated grid, pre-1202)", "interpolated ground-network grid"
)

# %% [markdown]
# ## Compare with final product monthly mean

# %%
esgf_ready_gas_dir = report.get_esgf_ready_gas_dir(output_bundles_root_path, run_id, gas)
baseline_chunks, available_fits = report.discover_gridded_files(esgf_ready_gas_dir, gas)

pipeline_final_zonal_allyears = report.load_concatenated_gridded(baseline_chunks, gas)
pipeline_final_allyears = local.xarray_space.calculate_global_mean_from_lon_mean(
    pipeline_final_zonal_allyears
)
pipeline_final_allyears

# %%
pipeline_final = pipeline_final_allyears.sel(time=test_period)

# %%
plot_pipeline_vs_test(pipeline_final, "Pipeline result (final gridded product)", "final gridded product")

# %% [markdown]
# ## Evaluation of distance

# %%
matched_final = em.match_monthly(pipeline_final, test_monthly)
matched_interpolated = em.match_monthly(pipeline_interpolated, test_monthly)

agreement_summary = pd.DataFrame(
    [
        {"pipeline_result": "A. Final gridded product", **em.summarise_agreement(matched_final)},
        {
            "pipeline_result": "B. Interpolated grid (pre-1202)",
            **em.summarise_agreement(matched_interpolated),
        },
    ]
).set_index("pipeline_result")
agreement_summary

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.axhline(0, color="grey", linewidth=1)
for matched, label, color in [
    (matched_final, "A. Final gridded product", "tab:blue"),
    (matched_interpolated, "B. Interpolated grid (pre-1202)", "tab:purple"),
]:
    ax.plot(
        [cftime.datetime(int(y), int(m), 15) for y, m in zip(matched["year"], matched["month"])],
        matched["residual"],
        marker="o",
        linestyle="none",
        markersize=4,
        alpha=0.7,
        color=color,
        label=label,
    )
ax.set_title(f"{gas.upper()} - residual (pipeline - held-out test data)")
ax.set_xlabel("Time")
ax.set_ylabel(f"Residual [{test_unit}]")
ax.legend()
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, matched, label in [
    (axes[0], matched_final, "A. Final gridded product"),
    (axes[1], matched_interpolated, "B. Interpolated grid (pre-1202)"),
]:
    ax.scatter(matched["test_value"], matched["pipeline_value"], alpha=0.5, s=20, color="tab:blue")
    lims = [
        min(matched["test_value"].min(), matched["pipeline_value"].min()),
        max(matched["test_value"].max(), matched["pipeline_value"].max()),
    ]
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_title(label)
    ax.set_xlabel(f"Held-out test data [{test_unit}]")
    ax.legend()
axes[0].set_ylabel(f"Pipeline result [{test_unit}]")
plt.tight_layout()
plt.show()

# %%

# %%

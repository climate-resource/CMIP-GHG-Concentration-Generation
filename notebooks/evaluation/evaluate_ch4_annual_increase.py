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
# # CH4: annual global increase during the satellite period
#
# For each year `x` in the satellite period, the year-on-year increase is
# `global_annual_mean[x] - global_annual_mean[x - 1]`. This notebook computes
# that increase from six different sources and shows each as a histogram,
# so you can compare how much year-to-year variability each source implies
# - and whether the pipeline's final product (with or without satellite
# data) looks like a plausible blend of the raw inputs feeding it:
#
# - **Satellite data (scaled)**: the bin-averaged, scaled satellite product
#   itself (`1100a_ch4_bin_satellite_data`'s output) - what actually gets
#   fed into the pipeline when satellite data is switched on.
# - **Ground-based data, per source**: NOAA in-situ, NOAA surface-flask, and
#   AGAGE - each source's own raw station data, independently
#   spatially-binned and globally averaged here (not the pipeline's combined
#   observational network).
# - **Final pipeline output, no satellite data**: the actual final
#   input4MIPs gridded product from the no-satellite baseline run.
# - **Final pipeline output, with satellite data (`satellite_fit`)**: same,
#   for the chosen satellite fit.
#
# All sources are restricted to the satellite period (from
# `local.diagnostics.get_satellite_data_year_range`, e.g. 2003-2023) and use
# a cos(latitude)-weighted global mean, equally weighted across
# months/years - a simple, consistent aggregation chosen for comparability
# across sources, not necessarily identical to what any individual pipeline
# step does internally.
#
# Needs `1100`/`1100a_ch4_bin_satellite_data` to have been run for
# `satellite_fit` (to get `scaled_data_path`/`interim_data_path`), and
# `40yy_write-input4mips` to have been run for both the no-satellite
# baseline and `satellite_fit` under `run_id`.
#
# **Gotcha:** unlike the other pieces used here, `interim_data_path` (the
# section 1 input) is *not* suffixed per satellite configuration - it's
# just overwritten by whichever `GAS`/`SAT_FIT`/`SAT_GAS` was active the
# last time `scripts/write-config.py` + `doit` ran `1100a_ch4_bin_satellite_data`
# under this `run_id`, and goes empty entirely if CH4's satellite data was
# switched off in that run. If section 1 comes back empty/wrong, re-run
# just that one (cheap) step for the fit you want, e.g.:
# `GAS=ch4 SAT_GAS=True SAT_FIT=<satellite_fit> pixi run python scripts/write-config.py`,
# then `doit run` targeting `1100a_ch4_bin_satellite_data:only` (see `doit list --all`
# for the exact target string doit expects).

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

import local.binning
import local.diagnostics
import local.diagnostics_reporting as report
import local.xarray_space
from local.config import load_config_from_file

# %%
# `load_config_from_file` needs `ppm`/`ppb` registered in the pint
# application registry to deserialise `smooth_law_dome_data`'s noise
# settings, even though nothing else in this notebook uses pint directly.
pint.set_application_registry(openscm_units.unit_registry)

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
gas: str = "ch4"
config_file: str = "../../dev-config-absolute.yaml"
run_id: str = "dev-test-run"
output_bundles_root: str = "../../output-bundles"
satellite_fit: str = "NONLINEAR_LAT_STD_WEIGHT_FIT"

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(Path(config_file))
config_process_noaa_surface_flask_data = get_config_for_step_id(
    config=config, step="process_noaa_surface_flask_data", step_config_id=gas
)
config_process_noaa_in_situ_data = get_config_for_step_id(
    config=config, step="process_noaa_in_situ_data", step_config_id=gas
)
config_process_agage_data = get_config_for_step_id(
    config=config, step="retrieve_and_extract_agage_data", step_config_id=f"{gas}_gc-md_monthly"
)
config_process_scaled_sat_data = get_config_for_step_id(
    config=config, step="process_scaled_sat_data", step_config_id=gas
)
output_bundles_root_path = Path(output_bundles_root)

satellite_period_start_year, satellite_period_end_year = local.diagnostics.get_satellite_data_year_range(
    config_process_scaled_sat_data.scaled_data_path
)
print(f"Satellite period: {satellite_period_start_year}-{satellite_period_end_year}")

# %% [markdown]
# ## Helpers
#
# `compute_global_annual_mean` takes any dataframe already resolved to
# `(year, month, lat_bin)` and computes a cos(latitude)-weighted global
# mean per month, then an equally-weighted mean across each year's months.
# `station_data_to_global_annual_mean` does the same starting from raw
# per-station rows (`latitude`/`longitude`, not yet binned).


# %%
def compute_global_annual_mean(monthly_binned: pd.DataFrame) -> pd.Series:
    """Cos(latitude)-weighted global annual mean from a `(year, month, lat_bin)`-resolved dataframe"""
    lat_mean = monthly_binned.groupby(["year", "month", "lat_bin"])["value"].mean().reset_index()
    lat_mean["cos_lat"] = np.cos(np.deg2rad(lat_mean["lat_bin"]))
    lat_mean["weighted_value"] = lat_mean["value"] * lat_mean["cos_lat"]
    monthly = lat_mean.groupby(["year", "month"])[["weighted_value", "cos_lat"]].sum()
    monthly["value"] = monthly["weighted_value"] / monthly["cos_lat"]
    return monthly.reset_index().groupby("year")["value"].mean()


def station_data_to_global_annual_mean(station_df: pd.DataFrame) -> pd.Series:
    """Bin raw per-station monthly rows onto the 15x60 degree grid, then compute the global annual mean"""
    binned = local.binning.add_lat_lon_bin_columns(station_df)
    bin_averages = local.binning.calculate_bin_averages(binned)
    return compute_global_annual_mean(bin_averages)


def year_over_year_increase(annual_mean: pd.Series, start_year: int, end_year: int) -> pd.Series:
    """`annual_mean[year] - annual_mean[year - 1]`, restricted to `year` in `[start_year, end_year]`"""
    diffs = annual_mean.sort_index().diff()
    return diffs[(diffs.index >= start_year) & (diffs.index <= end_year)]


def plot_increase_by_year(increases: dict[str, pd.Series], unit: str) -> None:
    """One bar chart per source: year on the x-axis, that year's increase on the y-axis"""
    n_cols = 3
    n_rows = -(-len(increases) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (label, raw_values) in zip(axes_flat, increases.items()):
        values = raw_values.dropna()
        if values.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9)
        else:
            ax.bar(values.index, values, color="tab:blue", edgecolor="black", alpha=0.8)
            ax.axhline(0, color="k", linestyle=":", linewidth=1)
            ax.axhline(values.mean(), color="tab:red", linestyle="--", linewidth=1.5)
        ax.set_title(f"{label}\nmean={values.mean():.2f}, std={values.std():.2f} [{unit}/yr]", fontsize=9)
        ax.set_xlabel("Year")

    for ax in axes_flat[len(increases) :]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel(f"Year-on-year increase [{unit}/yr]")
    fig.suptitle(
        f"{gas.upper()} - annual global increase, satellite period "
        f"({satellite_period_start_year}-{satellite_period_end_year})"
    )
    plt.tight_layout()
    plt.show()


def plot_final_output_comparison(
    nosat_increase: pd.Series, sat_increase: pd.Series, sat_label: str, unit: str
) -> None:
    """Plot a grouped bar chart comparing the two final-output increases directly, year by year"""
    years = sorted(set(nosat_increase.dropna().index) | set(sat_increase.dropna().index))
    width = 0.4

    _fig, ax = plt.subplots(figsize=(max(7, 0.25 * len(years)), 8))
    ax.bar(
        np.array(years) - width / 2,
        nosat_increase.reindex(years),
        width=width,
        color="tab:blue",
        label="No satellite data",
    )
    ax.bar(
        np.array(years) + width / 2,
        sat_increase.reindex(years),
        width=width,
        color="tab:orange",
        label=sat_label,
    )
    ax.axhline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Year-on-year increase [{unit}/yr]")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.set_title(f"{gas.upper()} - final output annual increase: no satellite vs. satellite data")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## 1. Satellite data (scaled)
#
# **Derived in:** `1100a_ch4_bin_satellite_data`
#
# Already binned onto the 15x60 degree grid (`gas, unit, year, month,
# lat_bin, lon_bin, value`), so this goes straight to
# `compute_global_annual_mean` - no station-level binning needed.

# %%
satellite_bin_averages = pd.read_csv(config_process_scaled_sat_data.interim_data_path)
satellite_unit = satellite_bin_averages["unit"].iloc[0]
satellite_annual_mean = compute_global_annual_mean(satellite_bin_averages)
satellite_annual_mean

# %% [markdown]
# ## 2. Ground-based data, per source
#
# **Derived in:** `001y_process-noaa-data`, `002y_process-agage-data`
#
# Raw per-station monthly data, independently binned and globally averaged
# here per source - not the pipeline's own combined observational network.

# %%
noaa_in_situ = pd.read_csv(config_process_noaa_in_situ_data.processed_monthly_data_with_loc_file)
noaa_flask = pd.read_csv(config_process_noaa_surface_flask_data.processed_monthly_data_with_loc_file)
agage = pd.read_csv(config_process_agage_data.processed_monthly_data_with_loc_file)

ground_unit = noaa_in_situ["unit"].iloc[0]

noaa_in_situ_annual_mean = station_data_to_global_annual_mean(noaa_in_situ)
noaa_flask_annual_mean = station_data_to_global_annual_mean(noaa_flask)
agage_annual_mean = station_data_to_global_annual_mean(agage)

pd.DataFrame(
    {
        "NOAA in-situ": noaa_in_situ_annual_mean,
        "NOAA flask": noaa_flask_annual_mean,
        "AGAGE": agage_annual_mean,
    }
)

# %% [markdown]
# ## 3. Final pipeline output
#
# **Derived in:** `40yy_write-input4mips`
#
# The actual final, gridded input4MIPs product - a zonal (lat-only,
# longitude already averaged out) mean - for the no-satellite baseline and
# for `satellite_fit`, both under `run_id`.

# %%
esgf_ready_gas_dir = report.get_esgf_ready_gas_dir(output_bundles_root_path, run_id, gas)
baseline_chunks, fit_chunks = report.discover_gridded_files(esgf_ready_gas_dir, gas)

if not baseline_chunks:
    msg = f"No no-satellite baseline gridded output found for {gas!r} under {esgf_ready_gas_dir}."
    raise ValueError(msg)
if satellite_fit not in fit_chunks:
    msg = f"No gridded output found for {gas!r}/{satellite_fit!r} under {esgf_ready_gas_dir}."
    raise ValueError(msg)


def gridded_to_global_annual_mean(chunks: list[Path]) -> pd.Series:
    """Load a variant's final gridded (zonal-mean) output and reduce it to a global annual mean"""
    zonal_da = report.load_concatenated_gridded(chunks, gas)
    global_monthly_da = local.xarray_space.calculate_global_mean_from_lon_mean(zonal_da)
    global_annual_da = global_monthly_da.groupby("time.year").mean()
    return global_annual_da.to_pandas()


pipeline_nosat_annual_mean = gridded_to_global_annual_mean(baseline_chunks)
pipeline_sat_annual_mean = gridded_to_global_annual_mean(fit_chunks[satellite_fit])

pd.DataFrame({"No satellite data": pipeline_nosat_annual_mean, satellite_fit: pipeline_sat_annual_mean})

# %% [markdown]
# ## 4. Year-on-year increase, satellite period only

# %%
increases = {
    "Satellite data (scaled)": year_over_year_increase(
        satellite_annual_mean, satellite_period_start_year, satellite_period_end_year
    ),
    "NOAA in-situ": year_over_year_increase(
        noaa_in_situ_annual_mean, satellite_period_start_year, satellite_period_end_year
    ),
    "NOAA flask": year_over_year_increase(
        noaa_flask_annual_mean, satellite_period_start_year, satellite_period_end_year
    ),
    "AGAGE": year_over_year_increase(
        agage_annual_mean, satellite_period_start_year, satellite_period_end_year
    ),
    "Final output, no satellite": year_over_year_increase(
        pipeline_nosat_annual_mean, satellite_period_start_year, satellite_period_end_year
    ),
    f"Final output, satellite ({satellite_fit})": year_over_year_increase(
        pipeline_sat_annual_mean, satellite_period_start_year, satellite_period_end_year
    ),
}
pd.DataFrame(increases)

# %% [markdown]
# ## 5. Annual increase by year, per source

# %%
plot_increase_by_year(increases, ground_unit)

# %% [markdown]
# ## 6. Final output: no satellite vs. satellite data, side by side

# %%
plot_final_output_comparison(
    increases["Final output, no satellite"],
    increases[f"Final output, satellite ({satellite_fit})"],
    f"Satellite data\n({satellite_fit})",
    ground_unit,
)

# %%

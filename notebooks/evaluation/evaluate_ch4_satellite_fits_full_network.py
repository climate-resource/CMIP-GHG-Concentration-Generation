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
# # CH4: satellite fit comparison against the full observational network
#
# Same question as `evaluate_ch4_satellite_fits.py` - does adding satellite
# data (and which fit) change how well the pipeline's final gridded product
# agrees with the ground-based observations - but **without** a held-out
# train/test split: the pipeline's own combined observational network
# already used every ground station when fitting, so this isn't an
# out-of-sample check. It answers a different, complementary question:
# does the final smoothed/gridded product still track the actual station
# network's own aggregate, and does satellite data change that?
#
# Only the `STD_WEIGHT` fits are compared here - `LINEAR_STD_WEIGHT_FIT`,
# `LINEAR_LAT_STD_WEIGHT_FIT`, `LINEAR_SEASONAL_STD_WEIGHT_FIT`,
# `LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT`, `NONLINEAR_LAT_STD_WEIGHT_FIT` -
# whichever of these have gridded (`40yy_write-input4mips`) output under
# `run_id` are discovered automatically.
#
# Every comparison below is shown twice: once over the ground-based
# observational network's own full record, and once restricted to the
# satellite period (2003 onwards) - satellite data structurally cannot
# affect anything outside that window, so restricting to it isolates its
# real effect. The seasonal latitude-gradient breakdown at the end is only
# shown for the satellite period.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import cftime
import matplotlib.pyplot as plt
import numpy as np
import openscm_units
import pandas as pd
import pint
import xarray as xr
from pydoit_nb.config_handling import get_config_for_step_id

import local.diagnostics_reporting as report
import local.evaluation_metrics as em
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
step: str = "calculate_ch4_monthly_fifteen_degree_pieces"
config_file: str = "../../dev-config-absolute.yaml"
step_config_id: str = "only"
run_id: str = "dev-test-run"
output_bundles_root: str = "../../output-bundles"
satellite_period_start_year: int = 2003

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)
output_bundles_root_path = Path(output_bundles_root)

# %% [markdown]
# ## Observations
#
# The pipeline's full ground-based observational network - already
# bin-averaged by `1100_ch4_bin-observational-network` - not a held-out
# subset, so this isn't a true out-of-sample check (see the note at the
# top).

# %%
obs_bin_averages = pd.read_csv(config_step.processed_bin_averages_file)

obs_unit = obs_bin_averages["unit"].unique()
if len(obs_unit) != 1:
    msg = f"Expected a single unit, got {obs_unit=}"
    raise AssertionError(msg)
obs_unit = obs_unit[0]

obs_lat_mean = obs_bin_averages.groupby(["year", "month", "lat_bin"])["value"].mean().reset_index()
obs_lat_mean["cos_lat"] = np.cos(np.deg2rad(obs_lat_mean["lat_bin"]))
obs_lat_mean["weighted_value"] = obs_lat_mean["value"] * obs_lat_mean["cos_lat"]

obs_monthly = obs_lat_mean.groupby(["year", "month"])[["weighted_value", "cos_lat"]].sum()
obs_monthly["value"] = obs_monthly["weighted_value"] / obs_monthly["cos_lat"]
obs_monthly = obs_monthly.reset_index()
obs_monthly["time"] = [
    cftime.datetime(int(y), int(m), 15) for y, m in zip(obs_monthly["year"], obs_monthly["month"])
]

ground_based_period = slice(
    cftime.datetime(int(obs_monthly["year"].min()), 1, 1),
    cftime.datetime(int(obs_monthly["year"].max()), 12, 31),
)
obs_monthly

# %% [markdown]
# ## Discover every variant on disk
#
# The no-satellite baseline, plus whichever `STD_WEIGHT` fits have been run
# and written under this `run_id` - they all coexist side by side, no need
# to load a separate config per fit.

# %%
esgf_ready_gas_dir = report.get_esgf_ready_gas_dir(output_bundles_root_path, run_id, gas)
baseline_chunks, fit_chunks = report.discover_gridded_files(esgf_ready_gas_dir, gas)

if not baseline_chunks:
    msg = f"No no-satellite baseline gridded output found for {gas!r} under {esgf_ready_gas_dir}."
    raise ValueError(msg)

std_weight_fits = sorted(fit for fit in fit_chunks if fit.endswith("STD_WEIGHT_FIT"))
variants = {"No satellite (baseline)": baseline_chunks, **{fit: fit_chunks[fit] for fit in std_weight_fits}}
print(f"Found {len(variants)} variants: {list(variants)}")

# %% [markdown]
# ## Monthly trend per variant, cropped to the ground-based observational network's record
#
# `pipeline_zonal` keeps the `lat` dimension (no longitude - the final
# product is a zonal mean) for the latitude-profile plots below;
# `pipeline_trends` is its cos(latitude)-weighted global mean, used
# everywhere else.

# %%
pipeline_zonal = {}
pipeline_trends = {}
for label, chunks in variants.items():
    gridded_da = report.load_concatenated_gridded(chunks, gas)
    pipeline_zonal[label] = gridded_da.sel(time=ground_based_period)
    pipeline_trends[label] = local.xarray_space.calculate_global_mean_from_lon_mean(pipeline_zonal[label])

# %% [markdown]
# ## Restricting to the satellite period
#
# `obs_monthly_satellite_period`/`pipeline_trends_satellite_period` are the
# same shape as `obs_monthly`/`pipeline_trends`, just restricted to
# `satellite_period_start_year` (2003) onwards - the only months where
# satellite data can possibly affect the result. Every comparison below is
# run once against each pair.

# %%
satellite_period = slice(cftime.datetime(satellite_period_start_year, 1, 1), ground_based_period.stop)

obs_monthly_satellite_period = obs_monthly[obs_monthly["year"] >= satellite_period_start_year].reset_index(
    drop=True
)
obs_lat_mean_satellite_period = obs_lat_mean[obs_lat_mean["year"] >= satellite_period_start_year]
pipeline_trends_satellite_period = {
    label: trend.sel(time=satellite_period) for label, trend in pipeline_trends.items()
}
pipeline_zonal_satellite_period = {
    label: zonal.sel(time=satellite_period) for label, zonal in pipeline_zonal.items()
}

print(f"'ground based period': {len(obs_monthly)} months")
print(f"'satellite period': {len(obs_monthly_satellite_period)} months")


# %% [markdown]
# ## Plotting/summary helpers
#
# Defined once, called once per period label below, so every comparison is
# shown for both the ground-based-period and the satellite-period without
# duplicating the plotting code itself.


# %%
SATELLITE_START_YEAR = 2003
GOSAT_START_YEAR = 2009


def add_reference_lines(ax: plt.Axes) -> None:
    """Mark when satellite data starts (2003) and GOSAT was added (2009) with grey dashed lines"""
    ax.axvline(
        cftime.datetime(SATELLITE_START_YEAR, 1, 1),
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="satellite data starts (2003)",
    )
    ax.axvline(
        cftime.datetime(GOSAT_START_YEAR, 1, 1),
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="GOSAT added (2009)",
    )


# %%
def plot_combined_overlay(
    trends: dict[str, xr.DataArray], obs_monthly_df: pd.DataFrame, period_label: str
) -> None:
    """Overlay every variant's trend plus the ground-based observations on one axis"""
    colors = plt.get_cmap("tab10").colors
    _fig, ax = plt.subplots(figsize=(13, 6))
    # Pipeline lines plotted first so xarray/nc-time-axis registers a
    # cftime unit converter on the x-axis before the plain `ax.plot` call
    # below (which also uses cftime values) needs it - `zorder` (not call
    # order) controls what ends up drawn on top.
    for i, (label, trend) in enumerate(trends.items()):
        trend.plot(  # type: ignore
            ax=ax,
            color=colors[i % len(colors)],
            marker="o",
            linestyle="none",
            markersize=4,
            label=label,
            zorder=2,
        )
    ax.plot(
        obs_monthly_df["time"],
        obs_monthly_df["value"],
        color="black",
        marker="o",
        linestyle="none",
        markersize=4,
        alpha=0.35,
        label="Observations (ground network)",
        zorder=1,
    )
    add_reference_lines(ax)
    ax.set_title(f"{gas.upper()} - monthly, global-mean trend: baseline vs. satellite fits ({period_label})")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{gas.upper()} [{obs_unit}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_individual_variants(
    trends: dict[str, xr.DataArray], obs_monthly_df: pd.DataFrame, period_label: str
) -> None:
    """One plot per *fit* variant, each vs. the no-satellite baseline and the ground-based observations"""
    baseline_label = next(iter(trends))
    baseline_trend = trends[baseline_label]
    for label, trend in trends.items():
        if label == baseline_label:
            continue
        _fig, ax = plt.subplots(figsize=(12, 5))
        trend.plot(  # type: ignore
            ax=ax,
            color="tab:blue",
            marker="o",
            linestyle="none",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=label,
            zorder=3,
        )
        ax.plot(
            obs_monthly_df["time"],
            obs_monthly_df["value"],
            color="tab:orange",
            marker="o",
            linestyle="none",
            markersize=5,
            alpha=0.6,
            label="Observations (ground network)",
            zorder=1,
        )
        ax.plot(
            baseline_trend["time"],
            baseline_trend.values,
            color="tab:green",
            marker="o",
            linestyle="none",
            markersize=1.25,
            alpha=0.6,
            label=baseline_label,
            zorder=4,
        )
        add_reference_lines(ax)
        ax.set_title(f"{gas.upper()} - monthly, global-mean trend: {label} vs. baseline ({period_label})")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{gas.upper()} [{obs_unit}]")
        ax.legend()
        plt.tight_layout()
        plt.show()


def compute_agreement(
    trends: dict[str, xr.DataArray], obs_monthly_df: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Match every variant against the ground-based observations and summarise agreement"""
    matched_per_variant = {label: em.match_monthly(trend, obs_monthly_df) for label, trend in trends.items()}
    summary = pd.DataFrame(
        [
            {"variant": label, **em.summarise_agreement(matched)}
            for label, matched in matched_per_variant.items()
        ]
    ).set_index("variant")
    return matched_per_variant, summary


def plot_scatter_grid(matched_per_variant: dict[str, pd.DataFrame], period_label: str) -> None:
    """One pipeline-vs-observations scatter (with a 1:1 line) per variant, in a grid"""
    n_variants = len(matched_per_variant)
    n_cols = 3
    n_rows = -(-n_variants // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, (label, matched) in zip(axes, matched_per_variant.items()):
        ax.scatter(matched["test_value"], matched["pipeline_value"], alpha=0.5, s=15, color="tab:blue")
        lims = [
            min(matched["test_value"].min(), matched["pipeline_value"].min()),
            max(matched["test_value"].max(), matched["pipeline_value"].max()),
        ]
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)

    for ax in axes[:n_variants]:
        ax.set_xlabel(f"Observations [{obs_unit}]")
    for ax in axes[n_variants:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n_variants]):
        if i % n_cols == 0:
            ax.set_ylabel(f"Pipeline result [{obs_unit}]")

    fig.suptitle(f"{gas.upper()} - pipeline vs. ground-based observations ({period_label})")
    plt.tight_layout()
    plt.show()


def obs_latitude_profile(obs_lat_mean_df: pd.DataFrame) -> pd.Series:
    """Time-averaged (equal-weight over year/month) observed value per latitude bin"""
    return obs_lat_mean_df.groupby("lat_bin")["value"].mean()


def plot_latitude_profile(
    zonal: dict[str, xr.DataArray], obs_lat_profile: pd.Series, period_label: str
) -> None:
    """Each variant's time-averaged latitude profile vs. the ground-based observations'"""
    colors = plt.get_cmap("tab10").colors
    _fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, da) in enumerate(zonal.items()):
        time_mean = da.mean(dim="time")
        ax.plot(
            time_mean["lat"],
            time_mean.values,
            marker="o",
            markersize=5,
            color=colors[i % len(colors)],
            label=label,
            zorder=2,
        )
    ax.plot(
        obs_lat_profile.index,
        obs_lat_profile.values,
        marker="s",
        markersize=9,
        linestyle="none",
        color="black",
        label="Observations (ground network)",
        zorder=3,
    )
    ax.set_title(f"{gas.upper()} - latitude profile, time-averaged ({period_label})")
    ax.set_xlabel("Latitude")
    ax.set_ylabel(f"{gas.upper()} [{obs_unit}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


DJF_MONTHS = (12, 1, 2)
JJA_MONTHS = (6, 7, 8)


def filter_season(
    zonal: dict[str, xr.DataArray], obs_lat_mean_df: pd.DataFrame, months: tuple[int, ...]
) -> tuple[dict[str, xr.DataArray], pd.DataFrame]:
    """Restrict `zonal` and `obs_lat_mean_df` to the given calendar months (e.g. DJF/JJA)"""
    zonal_season = {label: da.isel(time=da["time"].dt.month.isin(months)) for label, da in zonal.items()}
    obs_lat_mean_season = obs_lat_mean_df[obs_lat_mean_df["month"].isin(months)]
    return zonal_season, obs_lat_mean_season


def plot_latitude_profile_seasonal_breakdown(
    zonal: dict[str, xr.DataArray], obs_lat_mean_df: pd.DataFrame, period_label: str
) -> None:
    """Latitude profile for all months, then restricted to DJF (winter) and JJA (summer) separately"""
    for season_label, months in (("all months", None), ("DJF", DJF_MONTHS), ("JJA", JJA_MONTHS)):
        if months is None:
            zonal_season, obs_lat_mean_season = zonal, obs_lat_mean_df
        else:
            zonal_season, obs_lat_mean_season = filter_season(zonal, obs_lat_mean_df, months)
        plot_latitude_profile(
            zonal_season,
            obs_latitude_profile(obs_lat_mean_season),
            f"{period_label}, {season_label}",
        )


def plot_agreement_bars(agreement_summary: pd.DataFrame, period_label: str) -> None:
    """RMSE/bias/MAE/R²/correlation bar charts, one bar per variant"""
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    order = agreement_summary.index

    axes[0, 0].bar(order, agreement_summary["rmse"], color="tab:blue")
    axes[0, 0].set_title("RMSE")
    axes[0, 0].set_ylabel(f"RMSE [{obs_unit}]")

    axes[0, 1].axhline(0, color="grey", linewidth=1)
    axes[0, 1].bar(order, agreement_summary["bias"], color="tab:orange")
    axes[0, 1].set_title("Bias (pipeline - observations)")
    axes[0, 1].set_ylabel(f"Bias [{obs_unit}]")

    axes[0, 2].bar(order, agreement_summary["mae"], color="tab:green")
    axes[0, 2].set_title("MAE")
    axes[0, 2].set_ylabel(f"MAE [{obs_unit}]")

    axes[1, 0].bar(order, agreement_summary["r_squared"], color="tab:purple")
    axes[1, 0].set_title("R²")
    axes[1, 0].set_ylabel("R²")

    axes[1, 1].bar(order, agreement_summary["correlation"], color="tab:red")
    axes[1, 1].set_title("Correlation")
    axes[1, 1].set_ylabel("Pearson correlation")

    axes[1, 2].set_visible(False)

    for ax in axes.flatten():
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.suptitle(f"{gas.upper()} - agreement metrics by variant ({period_label})")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## 1. Every variant vs. ground-based observations
#
# The no-satellite baseline is shown on every plot here, directly answering
# "the final product without satellite data, compared against
# observations" - the fits are then overlaid on top of it.
#
# ### Ground based period

# %%
plot_combined_overlay(pipeline_trends, obs_monthly, "ground based period")

# %% [markdown]
# ### Satellite period

# %%
plot_combined_overlay(pipeline_trends_satellite_period, obs_monthly_satellite_period, "satellite period")

# %% [markdown]
# ## 2. Individual comparisons: each fit vs. baseline and observations
#
# The combined plots above get crowded with several series on top of each
# other - one plot per *fit* variant (the no-satellite baseline gets no
# plot of its own here, since it's shown on every other plot instead)
# makes it easier to see how closely any single fit tracks the
# observations, and whether it actually moves away from the baseline.
#
# ### Ground based period

# %%
plot_individual_variants(pipeline_trends, obs_monthly, "ground based period")

# %% [markdown]
# ### Satellite period

# %%
plot_individual_variants(pipeline_trends_satellite_period, obs_monthly_satellite_period, "satellite period")

# %% [markdown]
# ## 3. Agreement per variant
#
# Matches each variant against the ground-based observations by (year,
# month) and summarises the residuals (pipeline minus observations): bias,
# MAE, RMSE, R² and correlation.
#
# ### Ground based period

# %%
matched_per_variant, agreement_summary = compute_agreement(pipeline_trends, obs_monthly)
agreement_summary

# %% [markdown]
# ### Satellite period

# %%
matched_per_variant_satellite_period, agreement_summary_satellite_period = compute_agreement(
    pipeline_trends_satellite_period, obs_monthly_satellite_period
)
agreement_summary_satellite_period

# %% [markdown]
# ## 4. Scatter: pipeline vs. observations, with a 1:1 reference line
#
# A systematic offset from the dashed 1:1 line is bias; scatter around a
# line parallel to it (but offset) is still well-correlated but biased;
# scatter that doesn't track the line at all is poor agreement regardless
# of correlation.
#
# ### Ground based period

# %%
plot_scatter_grid(matched_per_variant, "ground based period")

# %% [markdown]
# ### Satellite period

# %%
plot_scatter_grid(matched_per_variant_satellite_period, "satellite period")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 5. RMSE, bias, MAE, R² and correlation by variant
#
# The same numbers as the agreement tables above (section 3), plotted so
# every variant - the no-satellite baseline and each `STD_WEIGHT` fit - can
# be compared at a glance.
#
# ### Ground based period

# %%
plot_agreement_bars(agreement_summary, "ground based period")

# %% [markdown]
# ### Satellite period

# %%
plot_agreement_bars(agreement_summary_satellite_period, "satellite period")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## 6. Seasonal breakdown of the latitudinal gradient (satellite period only)
#
# The trend plots above collapse `lat` into a single cos(latitude)-weighted
# global mean, which can hide fit-to-fit differences (several of the fits
# explicitly model a latitude-dependent correction - `LINEAR_LAT`,
# `LINEAR_SEASONAL_LAT`, `NONLINEAR_LAT`). This instead averages over
# *time* and keeps `lat`, shown for all months, then DJF (winter) and JJA
# (summer) separately - a seasonally-varying fit effect could otherwise be
# hidden by averaging over the full year. Only shown for the satellite
# period, since that's the only period a fit-to-fit difference can be
# attributed to satellite data at all.

# %%
plot_latitude_profile_seasonal_breakdown(
    pipeline_zonal_satellite_period, obs_lat_mean_satellite_period, "satellite period"
)

# %%

# %%

# %%

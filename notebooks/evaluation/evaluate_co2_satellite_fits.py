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
# # CO2: satellite fit comparison
#
# Builds on `evaluate_co2_train_test_split.py`'s baseline: does adding
# satellite data - and which fit - change how well the pipeline's final
# gridded product reproduces the same held-out ground observations?
#
# The held-out test set doesn't depend on satellite data (the train/test
# split happens before satellite data is combined with the ground network),
# so the no-satellite baseline and every satellite-fit run share the exact
# same held-out points - the comparison below is apples-to-apples.
#
# All five `STD_WEIGHT` fits are compared here: `LINEAR`, `LINEAR_LAT`,
# `LINEAR_SEASONAL`, `LINEAR_SEASONAL_LAT`, `NONLINEAR_LAT`.
#
# See `README.md` in this folder for how each fit's run was produced
# (`SAT_GAS=True SAT_FIT=<FIT> pixi run python scripts/write-eval-split-config.py`,
# then `doit`, once per fit, all writing into the same
# `dev-test-run-eval-split` output tree).
#
# **Every comparison below is shown twice: once over the full held-out
# period, and once restricted to the satellite period (2003 onwards).**
# Satellite data doesn't exist before 2003, so roughly half of the
# held-out months are identical (or nearly so - see below) across every
# variant, which dilutes the "all months" comparison. Even in the
# satellite period, fit-to-fit differences in the final product are
# typically only ~0.1-0.2 ppm - an order of magnitude smaller than the
# ~2 ppm RMSE against held-out ground truth, which is dominated by other
# noise sources (a single flask observation's synoptic-scale variability,
# the held-out sample's sparse/skewed spatial coverage). So don't expect
# even the satellite-period-only comparison to show dramatic separation
# between fits - it's a real, physical, but genuinely small effect.

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

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
gas: str = "co2"
step: str = "calculate_co2_monthly_fifteen_degree_pieces"
config_file: str = "../../eval-split-config-absolute.yaml"
step_config_id: str = "only"
run_id: str = "dev-test-run-eval-split"
output_bundles_root: str = "../../output-bundles"
satellite_period_start_year: int = 2003

# %% [markdown]
# ## Load config
#
# The held-out test set's location (`held_out_test_data_file`) doesn't
# depend on satellite data, so it doesn't matter which fit's config
# happens to be on disk as `eval-split-config-absolute.yaml` right now -
# every fit's run used the exact same split.

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)
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
#
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
test_data[:3]

# %%
test_data_with_bins = local.binning.add_lat_lon_bin_columns(test_data)
test_bin_averages = local.binning.calculate_bin_averages(test_data_with_bins)

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

test_period = slice(
    cftime.datetime(int(test_monthly["year"].min()), 1, 1),
    cftime.datetime(int(test_monthly["year"].max()), 12, 31),
)
test_monthly

# %% [markdown]
# ## Discover every variant on disk
#
# One `discover_gridded_files` call finds the no-satellite baseline *and*
# every satellite-fit variant that's been written under this `run_id` -
# they all coexist side by side (see `README.md`), no need to load a
# separate config per fit.

# %%
esgf_ready_gas_dir = report.get_esgf_ready_gas_dir(output_bundles_root_path, run_id, gas)
baseline_chunks, fit_chunks = report.discover_gridded_files(esgf_ready_gas_dir, gas)

if not baseline_chunks:
    msg = f"No no-satellite baseline gridded output found for {gas!r} under {esgf_ready_gas_dir}."
    raise ValueError(msg)

variants = {"No satellite (baseline)": baseline_chunks, **fit_chunks}
print(f"Found {len(variants)} variants: {list(variants)}")

# %% [markdown]
# ## Monthly trend per variant, cropped to the held-out test period
#
# `pipeline_zonal` keeps the `lat` dimension (no longitude - the final
# product is a zonal mean, see `evaluate_co2_train_test_split.py`) for the
# latitude-profile plots below; `pipeline_trends` is its cos(latitude)-weighted
# global mean, used everywhere else.

# %%
pipeline_zonal = {}
pipeline_trends = {}
for label, chunks in variants.items():
    gridded_da = report.load_concatenated_gridded(chunks, gas)
    pipeline_zonal[label] = gridded_da.sel(time=test_period)
    pipeline_trends[label] = local.xarray_space.calculate_global_mean_from_lon_mean(pipeline_zonal[label])

# %% [markdown]
# ## Restricting to the satellite period
#
# `test_monthly_satellite_period`/`pipeline_trends_satellite_period` are
# the same shape as `test_monthly`/`pipeline_trends`, just restricted to
# `satellite_period_start_year` (2003) onwards - the only months where
# satellite data can possibly affect the result. Every comparison below is
# run once against each pair.

# %%
satellite_period = slice(cftime.datetime(satellite_period_start_year, 1, 1), test_period.stop)

test_monthly_satellite_period = test_monthly[test_monthly["year"] >= satellite_period_start_year].reset_index(
    drop=True
)
pipeline_trends_satellite_period = {
    label: trend.sel(time=satellite_period) for label, trend in pipeline_trends.items()
}
pipeline_zonal_satellite_period = {
    label: zonal.sel(time=satellite_period) for label, zonal in pipeline_zonal.items()
}

print(f"'all months': {len(test_monthly)} held-out months")
print(f"'satellite period only': {len(test_monthly_satellite_period)} held-out months")


# %% [markdown]
# ## Plotting/summary helpers
#
# Defined once, called once per entry in `periods` in each section below,
# so every comparison is shown for both the full period and the
# satellite-only period without duplicating the plotting code itself.


# %%
def plot_combined_overlay(
    trends: dict[str, xr.DataArray], test_monthly_df: pd.DataFrame, period_label: str
) -> None:
    """Overlay every variant's trend plus the held-out test data on one axis"""
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
        test_monthly_df["time"],
        test_monthly_df["value"],
        color="black",
        marker="o",
        linestyle="none",
        markersize=4,
        alpha=0.35,
        label="Held-out test data",
        zorder=1,
    )
    ax.set_title(f"{gas.upper()} - monthly, global-mean trend: baseline vs. satellite fits ({period_label})")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{gas.upper()} [{test_unit}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_individual_variants(
    trends: dict[str, xr.DataArray], test_monthly_df: pd.DataFrame, period_label: str
) -> None:
    """One plot per *fit* variant, each vs. the no-satellite baseline and the held-out test data"""
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
            test_monthly_df["time"],
            test_monthly_df["value"],
            color="tab:orange",
            marker="o",
            linestyle="none",
            markersize=5,
            alpha=0.6,
            label="Held-out test data",
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
        ax.set_title(f"{gas.upper()} - monthly, global-mean trend: {label} vs. baseline ({period_label})")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{gas.upper()} [{test_unit}]")
        ax.legend()
        plt.tight_layout()
        plt.show()


def compute_agreement(
    trends: dict[str, xr.DataArray], test_monthly_df: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Match every variant against the held-out test data and summarise agreement"""
    matched_per_variant = {label: em.match_monthly(trend, test_monthly_df) for label, trend in trends.items()}
    summary = pd.DataFrame(
        [
            {"variant": label, **em.summarise_agreement(matched)}
            for label, matched in matched_per_variant.items()
        ]
    ).set_index("variant")
    return matched_per_variant, summary


def plot_scatter_grid(matched_per_variant: dict[str, pd.DataFrame], period_label: str) -> None:
    """One pipeline-vs-test scatter (with a 1:1 line) per variant, in a grid"""
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
        ax.set_xlabel(f"Held-out test data [{test_unit}]")
    for ax in axes[n_variants:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n_variants]):
        if i % n_cols == 0:
            ax.set_ylabel(f"Pipeline result [{test_unit}]")

    fig.suptitle(f"{gas.upper()} - pipeline vs. held-out test data ({period_label})")
    plt.tight_layout()
    plt.show()


def test_latitude_profile(test_lat_mean_df: pd.DataFrame) -> pd.Series:
    """Time-averaged (equal-weight over year/month) held-out test value per latitude bin"""
    return test_lat_mean_df.groupby("lat_bin")["value"].mean()


def plot_latitude_profile(
    zonal: dict[str, xr.DataArray], test_lat_profile: pd.Series, period_label: str
) -> None:
    """Each variant's time-averaged latitude profile vs. the held-out test data's"""
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
        test_lat_profile.index,
        test_lat_profile.values,
        marker="s",
        markersize=9,
        linestyle="none",
        color="black",
        label="Held-out test data",
        zorder=3,
    )
    ax.set_title(f"{gas.upper()} - latitude profile, time-averaged ({period_label})")
    ax.set_xlabel("Latitude")
    ax.set_ylabel(f"{gas.upper()} [{test_unit}]")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


DJF_MONTHS = (12, 1, 2)
JJA_MONTHS = (6, 7, 8)


def filter_season(
    zonal: dict[str, xr.DataArray], test_lat_mean_df: pd.DataFrame, months: tuple[int, ...]
) -> tuple[dict[str, xr.DataArray], pd.DataFrame]:
    """Restrict `zonal` and `test_lat_mean_df` to the given calendar months (e.g. DJF/JJA)"""
    zonal_season = {label: da.isel(time=da["time"].dt.month.isin(months)) for label, da in zonal.items()}
    test_lat_mean_season = test_lat_mean_df[test_lat_mean_df["month"].isin(months)]
    return zonal_season, test_lat_mean_season


def filter_hemisphere(
    zonal: dict[str, xr.DataArray], test_lat_mean_df: pd.DataFrame, hemisphere: str
) -> tuple[dict[str, xr.DataArray], pd.DataFrame]:
    """Restrict `zonal` and `test_lat_mean_df` to one hemisphere (``'north'`` or ``'south'``)"""
    if hemisphere == "north":
        zonal_hemi = {label: da.isel(lat=da["lat"] >= 0) for label, da in zonal.items()}
        test_lat_mean_hemi = test_lat_mean_df[test_lat_mean_df["lat_bin"] >= 0]
    else:
        zonal_hemi = {label: da.isel(lat=da["lat"] < 0) for label, da in zonal.items()}
        test_lat_mean_hemi = test_lat_mean_df[test_lat_mean_df["lat_bin"] < 0]
    return zonal_hemi, test_lat_mean_hemi


def plot_latitude_profile_seasonal_breakdown(
    zonal: dict[str, xr.DataArray], test_lat_mean_df: pd.DataFrame, period_label: str
) -> None:
    """DJF/JJA latitude profiles, then each of those split further by hemisphere"""
    for season_label, months in (("DJF", DJF_MONTHS), ("JJA", JJA_MONTHS)):
        zonal_season, test_lat_mean_season = filter_season(zonal, test_lat_mean_df, months)
        plot_latitude_profile(
            zonal_season,
            test_latitude_profile(test_lat_mean_season),
            f"{period_label}, {season_label}",
        )

    for season_label, months in (("DJF", DJF_MONTHS), ("JJA", JJA_MONTHS)):
        zonal_season, test_lat_mean_season = filter_season(zonal, test_lat_mean_df, months)
        for hemisphere_label, hemisphere in (
            ("northern hemisphere", "north"),
            ("southern hemisphere", "south"),
        ):
            zonal_hemi, test_lat_mean_hemi = filter_hemisphere(zonal_season, test_lat_mean_season, hemisphere)
            plot_latitude_profile(
                zonal_hemi,
                test_latitude_profile(test_lat_mean_hemi),
                f"{period_label}, {season_label}, {hemisphere_label}",
            )


def plot_agreement_bars(agreement_summary: pd.DataFrame, period_label: str) -> None:
    """RMSE/bias/MAE/R² bar charts, one bar per variant"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    order = agreement_summary.index

    axes[0, 0].bar(order, agreement_summary["rmse"], color="tab:blue")
    axes[0, 0].set_title("RMSE")
    axes[0, 0].set_ylabel(f"RMSE [{test_unit}]")

    axes[0, 1].axhline(0, color="grey", linewidth=1)
    axes[0, 1].bar(order, agreement_summary["bias"], color="tab:orange")
    axes[0, 1].set_title("Bias (pipeline - test)")
    axes[0, 1].set_ylabel(f"Bias [{test_unit}]")

    axes[1, 0].bar(order, agreement_summary["mae"], color="tab:green")
    axes[1, 0].set_title("MAE")
    axes[1, 0].set_ylabel(f"MAE [{test_unit}]")

    axes[1, 1].bar(order, agreement_summary["r_squared"], color="tab:purple")
    axes[1, 1].set_title("R²")
    axes[1, 1].set_ylabel("R²")

    for ax in axes.flatten():
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.suptitle(f"{gas.upper()} - agreement metrics by variant ({period_label})")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Plot: every variant vs. held-out test data
#
# ### All months

# %%
plot_combined_overlay(pipeline_trends, test_monthly, "all months")

# %% [markdown]
# ### Satellite period only

# %%
plot_combined_overlay(
    pipeline_trends_satellite_period, test_monthly_satellite_period, "satellite period only"
)

# %% [markdown]
# ## Individual comparisons: each variant vs. held-out test data
#
# The combined plots above get crowded with 6 series on top of each other -
# one plot per *fit* variant (the no-satellite baseline gets no plot of its
# own here, since it's shown on every other plot instead) makes it easier to
# see how closely any single fit tracks the held-out points, and whether it
# actually moves away from the baseline, the same way
# `evaluate_co2_train_test_split.py` compares its pipeline results.
#
# ### All months

# %%
plot_individual_variants(pipeline_trends, test_monthly, "all months")

# %% [markdown]
# ### Satellite period only

# %%
plot_individual_variants(
    pipeline_trends_satellite_period, test_monthly_satellite_period, "satellite period only"
)

# %% [markdown]
# ## Latitude profile: time-averaged effect on latitude
#
# The plots above collapse `lat` into a single cos(latitude)-weighted
# global mean, which is exactly the kind of aggregation that can hide
# fit-to-fit differences (several of the fits explicitly model a
# latitude-dependent correction - `LINEAR_LAT`, `LINEAR_SEASONAL_LAT`,
# `NONLINEAR_LAT`). This instead averages over *time* and keeps `lat`, so
# any latitudinal structure the fits disagree about should show up
# directly as a shape difference between the lines. The held-out test
# data's profile is the time-average, per latitude bin, of whichever
# bins happened to have held-out points (see `README.md` for why this
# sample is sparse and not evenly spread across latitude) - treat it as a
# rough reference, not ground truth at every latitude.
#
# ### All months

# %%
test_lat_profile = test_latitude_profile(test_lat_mean)
plot_latitude_profile(pipeline_zonal, test_lat_profile, "all months")

# %% [markdown]
# #### Seasonal breakdown (all months)
#
# The same profile, restricted to DJF (winter) and JJA (summer) separately,
# then each of those split further into northern/southern hemisphere - the
# global profile above averages over a full year, which can hide a
# seasonally- or hemispherically-varying fit effect the same way the
# global-mean trend plots hide latitudinal structure.

# %%
plot_latitude_profile_seasonal_breakdown(pipeline_zonal, test_lat_mean, "all months")

# %% [markdown]
# ### Satellite period only

# %%
test_lat_mean_satellite_period = test_lat_mean[test_lat_mean["year"] >= satellite_period_start_year]
test_lat_profile_satellite_period = test_latitude_profile(test_lat_mean_satellite_period)
plot_latitude_profile(
    pipeline_zonal_satellite_period, test_lat_profile_satellite_period, "satellite period only"
)

# %% [markdown]
# #### Seasonal breakdown (satellite period only)

# %%
plot_latitude_profile_seasonal_breakdown(
    pipeline_zonal_satellite_period, test_lat_mean_satellite_period, "satellite period only"
)

# %% [markdown]
# ## Agreement per variant
#
# Matches each variant against the held-out test data by (year, month) and
# summarises the residuals (pipeline minus test): bias, MAE, RMSE, R² and
# correlation - see `evaluate_co2_train_test_split.py`/`README.md` for the
# full definitions and the sampling-geometry caveat that applies to this
# global-mean comparison.
#
# ### All months

# %%
matched_per_variant, agreement_summary = compute_agreement(pipeline_trends, test_monthly)
agreement_summary

# %% [markdown]
# ### Satellite period only

# %%
matched_per_variant_satellite_period, agreement_summary_satellite_period = compute_agreement(
    pipeline_trends_satellite_period, test_monthly_satellite_period
)
agreement_summary_satellite_period

# %% [markdown]
# ## Scatter: pipeline vs. test, with a 1:1 reference line
#
# A systematic offset from the dashed 1:1 line is bias; scatter around a
# line parallel to it (but offset) is still well-correlated but biased;
# scatter that doesn't track the line at all is poor agreement regardless
# of correlation.
#
# ### All months

# %%
plot_scatter_grid(matched_per_variant, "all months")

# %% [markdown]
# ### Satellite period only

# %%
plot_scatter_grid(matched_per_variant_satellite_period, "satellite period only")

# %% [markdown]
# ### RMSE, bias, MAE and R² by variant - all months

# %%
plot_agreement_bars(agreement_summary, "all months")

# %% [markdown]
# ### RMSE, bias, MAE and R² by variant - satellite period only

# %%
plot_agreement_bars(agreement_summary_satellite_period, "satellite period only")

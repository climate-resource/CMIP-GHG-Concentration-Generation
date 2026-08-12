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
# # Compare satellite-data diagnostics
#
# This notebook compares the diagnostics saved by the `12yy_co2-monthly-15-degree`
# and `11yy_ch4-monthly-15-degree` pipelines across different satellite-data
# configurations (currently: no satellite data at all, vs. satellite data using
# the `LINEAR_FIT` fit). Each section explains, in a couple of sentences, what
# the diagnostic actually is and what a "big" difference between configurations
# would mean.
#
# Unlike `check_CO2_output_sat.py`/`check_CH4_output_sat.py` (which compare the
# *final* gridded input4MIPs output by hand, with paths that need updating for
# every new run), this notebook reads the small diagnostics files that the
# `12yy`/`11yy` notebooks now save automatically at several points *inside* the
# pipeline - so it shows not just "did the final answer change" but roughly
# *where in the pipeline* that change comes from.
#
# Not covered here (out of scope for this first pass, see the conversation this
# notebook came out of): full spatial diff maps of the interpolated field, and
# a capstone comparison at the final input4MIPs level (that needs the `40yy`
# step to have been run for both configurations too).

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
run_id: str = "dev-test-run"
diagnostics_root: str = "../../output-bundles/{run_id}/data/diagnostics"
gases: tuple[str, ...] = ("co2", "ch4")
# Order matters for consistent colouring below.
satellite_configs: tuple[str, ...] = ("nosat", "SAT_LINEAR_FIT")
satellite_config_labels: dict[str, str] = {
    "nosat": "No satellite data",
    "SAT_LINEAR_FIT": "Satellite data (linear fit)",
}
satellite_config_colors: dict[str, str] = {
    "nosat": "tab:blue",
    "SAT_LINEAR_FIT": "tab:orange",
}

# %%
diagnostics_root_path = Path(diagnostics_root.format(run_id=run_id))
diagnostics_root_path


# %% [markdown]
# ## Helpers


# %%
def diagnostics_path(gas: str, piece: str, suffix: str, ext: str) -> Path:
    """Path to a diagnostics file, matching `local.diagnostics.diagnostics_file_stem`"""
    return diagnostics_root_path / gas / f"{gas}_diagnostics_{piece}__{suffix}{ext}"


def load_yaml_diagnostics(gas: str, piece: str, suffix: str) -> dict | None:
    """Load a scalar/small diagnostics YAML file, if it exists"""
    path = diagnostics_path(gas, piece, suffix, ".yaml")
    if not path.exists():
        return None

    with open(path) as fh:
        return yaml.safe_load(fh)


def load_nc_diagnostics(gas: str, piece: str, suffix: str) -> xr.Dataset | None:
    """Load an array-valued diagnostics NetCDF file, if it exists"""
    path = diagnostics_path(gas, piece, suffix, ".nc")
    if not path.exists():
        return None

    return xr.load_dataset(path)


def check_to_row(gas: str, suffix: str, check: dict | None) -> dict:
    """Flatten a `local.diagnostics.record_check` dictionary into a table row"""
    if check is None:
        return {
            "gas": gas,
            "satellite_config": satellite_config_labels[suffix],
            "check": None,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "n_points": None,
        }

    return {
        "gas": gas,
        "satellite_config": satellite_config_labels[suffix],
        "check": check["name"],
        "max_abs_diff": check["max_abs_diff"],
        "max_rel_diff": check["max_rel_diff"],
        "n_points": check["n_points"],
    }


# %% [markdown]
# ## 1. Input coverage
#
# From `1201_co2_interpolate-observational-network` / `1101_ch4_...`. This is
# the most "upstream" diagnostic: how many extra data points does satellite
# data actually add to the spatial interpolation each month, and how many
# months does that save from being dropped for having too few points (fewer
# than 4) or from producing NaNs after interpolation? Everything downstream is
# a consequence of this changing.

# %%
coverage_rows = []
for gas in gases:
    for suffix in satellite_configs:
        d = load_yaml_diagnostics(gas, "interpolation", suffix)
        if d is None:
            continue
        coverage_rows.append(
            {
                "gas": gas,
                "satellite_config": satellite_config_labels[suffix],
                "ground_network_bin_rows": d["n_ground_network_bin_rows"],
                "satellite_bin_rows": d["n_satellite_bin_rows"],
                "year_months_kept": d["n_year_months_kept"],
                "year_months_dropped_insufficient_points": d["n_year_months_dropped_insufficient_points"],
                "year_months_dropped_nan": d["n_year_months_dropped_nan_after_interpolation"],
                "points_per_year_month_mean": round(d["points_per_year_month_mean"], 1),
                "points_per_year_month_min": d["points_per_year_month_min"],
            }
        )

coverage_df = pd.DataFrame(coverage_rows).set_index(["gas", "satellite_config"])
coverage_df

# %% [markdown]
# ## 2. Latitudinal gradient EOFs (observational-network period)
#
# From `1202_co2_..._latitudinal-gradient-seasonality` / `1102_ch4_...`. The
# latitudinal gradient is decomposed into Empirical Orthogonal Functions (EOFs,
# spatial north-south patterns) and Principal Components (PCs, how strongly each
# pattern is expressed each year). Satellite data feeds directly into this
# decomposition (it only covers the observational-network period), so this is
# where its effect on the pipeline's *statistics* (not just its inputs) first
# shows up.
#
# We plot the first two EOF spatial patterns and their explained-variance ratio
# for each configuration. A EOF pattern that visibly changes shape (not just
# scale) between configurations means satellite data is picking up a genuinely
# different spatial structure, not just adding noise.

# %%
for gas in gases:
    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    fig.suptitle(f"{gas.upper()} - latitudinal gradient EOFs (observational network)")

    for suffix in satellite_configs:
        ds = load_nc_diagnostics(gas, "obs-network", suffix)
        if ds is None:
            continue
        color = satellite_config_colors[suffix]
        label = satellite_config_labels[suffix]

        for eof in range(min(2, ds.sizes["lat_gradient_eof"])):
            axes[eof].plot(
                ds["lat_gradient_eofs_full"].sel(lat_gradient_eof=eof),
                ds["lat"],
                color=color,
                label=label if eof == 0 else None,
            )
            axes[eof].set_title(f"EOF {eof}")
            axes[eof].set_xlabel(f"{gas.upper()} anomaly")
            axes[eof].set_ylabel("Latitude")

        n_show = min(6, ds.sizes["lat_gradient_eof"])
        axes[2].plot(
            range(n_show),
            ds["lat_gradient_explained_variance_ratio"].isel(lat_gradient_eof=slice(0, n_show)),
            "o-",
            color=color,
            label=label,
        )
        axes[2].set_title("Explained variance ratio")
        axes[2].set_xlabel("EOF index")
        axes[2].set_ylabel("Fraction of variance")

    axes[0].legend()
    axes[2].legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Latitudinal gradient principal components (observational-network years)
#
# The PC0/PC1 timeseries these EOFs are scaled by, over the years the
# observational network actually covers. A shift here in the satellite-covered
# years (roughly 2003 onwards) is the direct effect of adding satellite data;
# a shift in *earlier* years as well would be a sign that something unexpected
# is going on, since satellite data shouldn't be able to affect years before it
# was collected at this stage of the pipeline.

# %%
for gas in gases:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharex=True)
    fig.suptitle(f"{gas.upper()} - latitudinal gradient PCs (observational network)")

    for suffix in satellite_configs:
        ds = load_nc_diagnostics(gas, "obs-network", suffix)
        if ds is None:
            continue
        color = satellite_config_colors[suffix]
        label = satellite_config_labels[suffix]

        for eof in range(min(2, ds.sizes["lat_gradient_eof"])):
            ds["lat_gradient_pcs_full"].sel(lat_gradient_eof=eof).plot(ax=axes[eof], color=color, label=label)
            axes[eof].set_title(f"PC{eof}")

    axes[0].legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. Seasonality (observational-network period)
#
# CO2 and CH4 handle seasonality differently (see the discussion this notebook
# came out of): CO2 splits it into a fixed climatological shape *plus* a
# separately EOF-decomposed "seasonality change" term; CH4 uses a single fixed
# *relative* (fractional) seasonal cycle with no EOF decomposition at all.

# %% [markdown]
# ### CO2: seasonality-change EOF (leading pattern) and explained variance

# %%
if "co2" in gases:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle("CO2 - seasonality-change EOF0 (observational network)")

    for suffix in satellite_configs:
        ds = load_nc_diagnostics("co2", "obs-network", suffix)
        if ds is None:
            continue
        color = satellite_config_colors[suffix]
        label = satellite_config_labels[suffix]

        ds["seasonality_change_eofs_full"].sel(seasonality_change_eof=0).plot.line(
            ax=axes[0], hue="lat", add_legend=False, color=color, alpha=0.6
        )

        n_show = min(6, ds.sizes["seasonality_change_eof"])
        axes[1].plot(
            range(n_show),
            ds["seasonality_change_explained_variance_ratio"].isel(seasonality_change_eof=slice(0, n_show)),
            "o-",
            color=color,
            label=label,
        )

    axes[0].set_title("EOF0 by latitude and month (one line per latitude)")
    axes[1].set_title("Explained variance ratio")
    axes[1].set_xlabel("EOF index")
    axes[1].legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### CH4: relative (fractional) seasonality field
#
# No EOF here - just the field itself. Plotted as the seasonal cycle at a few
# representative latitudes.

# %%
if "ch4" in gases:
    fig, ax = plt.subplots(figsize=(8, 5))

    example_lats = [-82.5, -37.5, 7.5, 52.5, 82.5]
    linestyles = ["-", "--"]

    for suffix, ls in zip(satellite_configs, linestyles):
        ds = load_nc_diagnostics("ch4", "obs-network", suffix)
        if ds is None:
            continue
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, lat in enumerate(example_lats):
            da = ds["relative_seasonality_obs_network"].sel(lat=lat, method="nearest")
            ax.plot(
                da["month"],
                da,
                linestyle=ls,
                color=color_cycle[i % len(color_cycle)],
                label=f"lat={lat} ({satellite_config_labels[suffix]})",
            )

    ax.set_xlabel("Month")
    ax.set_ylabel("Relative seasonal anomaly")
    ax.set_title("CH4 relative seasonality: solid = no satellite, dashed = satellite (linear fit)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Global-, annual-mean (observational-network period)
#
# The area-weighted global mean, before any historical extension. There's no
# EOF to compare here - it's just a scalar-per-year timeseries - so we plot it
# directly, plus the difference between configurations.

# %%
for gas in gases:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle(f"{gas.upper()} - global-, annual-mean (observational network)")

    series = {}
    for suffix in satellite_configs:
        ds = load_nc_diagnostics(gas, "obs-network", suffix)
        if ds is None:
            continue
        da = ds["global_annual_mean_obs_network"]
        series[suffix] = da
        da.plot(ax=axes[0], color=satellite_config_colors[suffix], label=satellite_config_labels[suffix])

    axes[0].set_title("Global-, annual-mean")
    axes[0].legend()

    if set(satellite_configs).issubset(series):
        common_years = np.intersect1d(
            series[satellite_configs[0]]["year"], series[satellite_configs[1]]["year"]
        )
        diff = series[satellite_configs[1]].sel(year=common_years) - series[satellite_configs[0]].sel(
            year=common_years
        )
        diff.plot(ax=axes[1])
        axes[1].axhline(0, color="k", linewidth=0.7)
        axes[1].set_title(f"{satellite_configs[1]} minus {satellite_configs[0]}")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Latitudinal gradient extension (all years)
#
# From `1203_co2_extend-lat-gradient-pcs` / `1103_ch4_extend-pcs`. The PCs
# above, extended back to year 1 using a regression against PRIMAP fossil
# emissions (both gases), plus - CH4 only - a joint optimisation against the
# NEEM and Law Dome ice cores for the gap closest to the observational network.

# %% [markdown]
# ### PC0/PC1, all years

# %%
for gas in gases:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharex=True)
    fig.suptitle(f"{gas.upper()} - latitudinal gradient PCs, all years")

    for suffix in satellite_configs:
        ds = load_nc_diagnostics(gas, "lat-gradient-extend", suffix)
        if ds is None:
            continue
        color = satellite_config_colors[suffix]
        label = satellite_config_labels[suffix]

        for eof in range(min(2, ds.sizes["eof"])):
            ds["principal-components"].sel(eof=eof).plot(ax=axes[eof], color=color, label=label)
            axes[eof].set_title(f"PC{eof}")

    axes[0].legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### PC0-vs-emissions regression quality
#
# The linear regression of PC0 against PRIMAP fossil emissions that fills the
# gap between the ice-core-derived period and the observational network. `r2`
# close to 1 means PC0 is well explained by emissions alone in the years both
# are observed; a large change in `r2` between configurations would suggest
# satellite data made this relationship noisier (or cleaner).

# %%
regression_rows = []
for gas in gases:
    for suffix in satellite_configs:
        d = load_yaml_diagnostics(gas, "lat-gradient-extend", suffix)
        if d is None:
            continue
        regression_rows.append(
            {
                "gas": gas,
                "satellite_config": satellite_config_labels[suffix],
                "m": d["pc0_emissions_regression_m"],
                "c": d["pc0_emissions_regression_c"],
                "r2": d["pc0_emissions_regression_r2"],
                "n_years_in_regression": d["pc0_emissions_regression_n_years"],
                "n_years_filled_with_regression": d["n_years_filled_with_regression"],
            }
        )

pd.DataFrame(regression_rows).set_index(["gas", "satellite_config"])

# %% [markdown]
# ### CH4: NEEM/Law Dome joint-optimisation residual
#
# CH4-only. For the years between the observational network and the
# ice-core-optimised period, a single (global-mean, PC0) pair is fit per year
# to simultaneously match both the NEEM and Law Dome ice cores. The residual
# left behind by that fit (area-weighted RMS difference, in the gas's usual
# unit) is otherwise discarded once the optimiser converges - plotted here so
# you can see whether satellite data (which shifts the EOFs this optimisation
# uses) makes that joint fit harder to achieve.

# %%
if "ch4" in gases:
    fig, ax = plt.subplots(figsize=(8, 4))

    for suffix in satellite_configs:
        ds = load_nc_diagnostics("ch4", "lat-gradient-extend", suffix)
        if ds is None or "neem_law_dome_optimisation_residual" not in ds:
            continue
        ds["neem_law_dome_optimisation_residual"].plot(
            ax=ax, color=satellite_config_colors[suffix], label=satellite_config_labels[suffix]
        )

    ax.set_title("CH4 - NEEM/Law Dome joint-optimisation residual by year")
    ax.set_ylabel(f"RMS residual ({ds['neem_law_dome_optimisation_residual'].attrs.get('units', '')})")
    ax.legend()
    plt.tight_layout()
    plt.show()

    optim_rows = []
    for suffix in satellite_configs:
        d = load_yaml_diagnostics("ch4", "lat-gradient-extend", suffix)
        if d is None:
            continue
        optim_rows.append(
            {
                "satellite_config": satellite_config_labels[suffix],
                "n_years_optimised": d["n_years_optimised_against_ice_cores"],
                "residual_mean": d["optimisation_residual_mean"],
                "residual_max": d["optimisation_residual_max"],
            }
        )
    pd.DataFrame(optim_rows).set_index("satellite_config")

# %% [markdown]
# ## 6. Global-, annual-mean extension (all years) and correctness checks
#
# From `1204_co2_extend-global-annual-mean` / `1104_ch4_...`. This is where the
# ice-core-derived history gets stitched onto the observational-network period,
# and where the pipeline's own internal correctness checks live (NEEM, Law
# Dome, EPICA, Menking et al., and a "does the field still decompose into
# global-mean + latitudinal-gradient" self-consistency check). Normally these
# only ever raise an error if they fail; here we print the actual numbers
# behind them so a *passing* check that got noticeably worse is still visible.

# %% [markdown]
# ### All-years global-, annual-mean

# %%
for gas in gases:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle(f"{gas.upper()} - global-, annual-mean, all years")

    series = {}
    for suffix in satellite_configs:
        ds = load_nc_diagnostics(gas, "global-mean-extend", suffix)
        if ds is None:
            continue
        da = ds["global_annual_mean_allyears"]
        series[suffix] = da
        da.plot(ax=axes[0], color=satellite_config_colors[suffix], label=satellite_config_labels[suffix])

    axes[0].set_title("All years")
    axes[0].legend()

    if set(satellite_configs).issubset(series):
        common_years = np.intersect1d(
            series[satellite_configs[0]]["year"], series[satellite_configs[1]]["year"]
        )
        diff = series[satellite_configs[1]].sel(year=common_years) - series[satellite_configs[0]].sel(
            year=common_years
        )
        diff.plot(ax=axes[1])
        axes[1].axhline(0, color="k", linewidth=0.7)
        axes[1].set_title(f"{satellite_configs[1]} minus {satellite_configs[0]}, all years")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Correctness checks
#
# - `max_abs_diff` / `max_rel_diff`: the largest absolute/relative difference
#   between the pipeline's reconstruction and the independent reference data
#   (an ice core, or another gas-specific dataset) the check compares against.
# - `field_decomposition`: not a comparison to external data - it checks that
#   `full_field - global_annual_mean` exactly reproduces the latitudinal
#   gradient computed earlier in the pipeline. Should be ~machine precision;
#   if this gets noticeably worse, something is wrong with the pipeline itself,
#   not just with the data.

# %%
check_rows = []
for gas in gases:
    for suffix in satellite_configs:
        d = load_yaml_diagnostics(gas, "global-mean-extend", suffix)
        if d is None:
            continue
        for check_name in (
            "neem_check",
            "law_dome_check",
            "epica_check",
            "menking_check",
            "field_decomposition_check",
        ):
            if check_name in d:
                check_rows.append(check_to_row(gas, suffix, d[check_name]))

checks_df = pd.DataFrame(check_rows).set_index(["gas", "satellite_config", "check"])
checks_df

# %% [markdown]
# ### Harmonisation seam size
#
# How big a jump the global-, annual-mean takes right at each point where one
# data source is stitched onto another (e.g. Mauna Loa onto the observational
# network for CO2, Law Dome onto the observational network for CH4), compared
# to a typical year-on-year step elsewhere. Satellite data only affects the
# observational-network side of these joins, so a growing seam jump would mean
# it's pulling the two sides further apart.

# %%
seam_rows = []
for gas in gases:
    for suffix in satellite_configs:
        d = load_yaml_diagnostics(gas, "global-mean-extend", suffix)
        if d is None:
            continue
        row = {
            "gas": gas,
            "satellite_config": satellite_config_labels[suffix],
            "typical_year_on_year_step": d["typical_year_on_year_step"],
        }
        if gas == "co2":
            row["mauna_loa_seam_jump"] = d["mauna_loa_seam_jump"]
            row["menking_seam_jump"] = d["menking_seam_jump"]
        else:
            row["law_dome_seam_jump"] = d["law_dome_seam_jump"]
        seam_rows.append(row)

pd.DataFrame(seam_rows).set_index(["gas", "satellite_config"])

# %% [markdown]
# ## 7. CO2: seasonality-change extension (all years)
#
# From `1205_co2_extend-seasonality-change-pcs`. CO2-only - see section 3.

# %%
if "co2" in gases:
    fig, ax = plt.subplots(figsize=(7, 4))

    for suffix in satellite_configs:
        ds = load_nc_diagnostics("co2", "seasonality-extend", suffix)
        if ds is None:
            continue
        ds["principal-components"].sel(eof=0).plot(
            ax=ax, color=satellite_config_colors[suffix], label=satellite_config_labels[suffix]
        )

    ax.set_title("CO2 - seasonality-change PC0, all years")
    ax.legend()
    plt.tight_layout()
    plt.show()

    seasonality_regression_rows = []
    for suffix in satellite_configs:
        d = load_yaml_diagnostics("co2", "seasonality-extend", suffix)
        if d is None:
            continue
        seasonality_regression_rows.append(
            {
                "satellite_config": satellite_config_labels[suffix],
                "m": d["pc0_composite_regression_m"],
                "c": d["pc0_composite_regression_c"],
                "r2": d["pc0_composite_regression_r2"],
                "n_years_in_regression": d["pc0_composite_regression_n_years"],
            }
        )
    pd.DataFrame(seasonality_regression_rows).set_index("satellite_config")

# %% [markdown]
# ## 8. Summary
#
# The handful of numbers most likely to matter for deciding whether a given
# satellite fit is worth using: how much input coverage changed, how much the
# final (well, final-within-`12yy`/`11yy`) global-, annual-mean shifted in the
# years satellite data actually covers, and whether any correctness check got
# meaningfully worse.

# %%
summary_rows = []
for gas in gases:
    cov = {s: load_yaml_diagnostics(gas, "interpolation", s) for s in satellite_configs}
    gm = {s: load_nc_diagnostics(gas, "global-mean-extend", s) for s in satellite_configs}

    if all(cov.values()):
        extra_points = (
            cov[satellite_configs[1]]["n_satellite_bin_rows"] if satellite_configs[1] in cov else None
        )
    else:
        extra_points = None

    if all(gm.values()):
        common_years = np.intersect1d(
            gm[satellite_configs[0]]["global_annual_mean_allyears"]["year"],
            gm[satellite_configs[1]]["global_annual_mean_allyears"]["year"],
        )
        # Focus on the satellite-covered period (2003 onwards) specifically.
        recent_years = common_years[common_years >= 2003]
        diff = gm[satellite_configs[1]]["global_annual_mean_allyears"].sel(year=recent_years) - gm[
            satellite_configs[0]
        ]["global_annual_mean_allyears"].sel(year=recent_years)
        max_abs_shift_2003_on = float(np.abs(diff).max())
    else:
        max_abs_shift_2003_on = None

    summary_rows.append(
        {
            "gas": gas,
            "satellite_bin_rows_added": extra_points,
            "max_abs_global_mean_shift_since_2003": max_abs_shift_2003_on,
        }
    )

pd.DataFrame(summary_rows).set_index("gas")

# %%

# %%

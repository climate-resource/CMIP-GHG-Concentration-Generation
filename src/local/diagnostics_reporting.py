"""
Reporting/plotting helpers for comparing satellite-data diagnostics across pipeline runs

This module is deliberately separate from :mod:`local.diagnostics`:
:mod:`local.diagnostics` is imported by the production `12yy`/`11yy` notebooks
(via `doit`) to *save* diagnostics; this module is only ever imported by
notebooks under `notebooks/diagnostics/` to *load and plot* them. Keeping the
split means the production notebooks never need matplotlib/pandas/cartopy as
a dependency of the pipeline itself.

Each function takes the resolved diagnostics root directory, the gas, and a
pair of satellite-data "suffixes" to compare (see
`local.diagnostics.get_satellite_suffix` - typically `"nosat"` vs
`"SAT_{fit}"`). Missing diagnostics files (e.g. because that fit hasn't been
run yet) are handled gracefully - the corresponding line/table entry is just
left out rather than raising.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml


def diagnostics_path(root: Path, gas: str, piece: str, suffix: str, ext: str) -> Path:
    """Path to a diagnostics file, matching `local.diagnostics.diagnostics_file_stem`"""
    return root / gas / f"{gas}_diagnostics_{piece}__{suffix}{ext}"


def load_yaml_diagnostics(root: Path, gas: str, piece: str, suffix: str) -> dict | None:
    """Load a scalar/small diagnostics YAML file, if it exists"""
    path = diagnostics_path(root, gas, piece, suffix, ".yaml")
    if not path.exists():
        return None

    with open(path) as fh:
        return yaml.safe_load(fh)


def load_nc_diagnostics(root: Path, gas: str, piece: str, suffix: str) -> xr.Dataset | None:
    """Load an array-valued diagnostics NetCDF file, if it exists"""
    path = diagnostics_path(root, gas, piece, suffix, ".nc")
    if not path.exists():
        return None

    return xr.load_dataset(path)


def _round_or_none(value: float | None, ndigits: int = 1) -> float | None:
    """`round`, but passes `None` through instead of raising"""
    return None if value is None else round(value, ndigits)


def _check_to_row(gas: str, label: str, check: dict | None) -> dict[str, Any] | None:
    """Flatten a `local.diagnostics.record_check` dictionary into a table row"""
    if check is None:
        return None

    return {
        "gas": gas,
        "satellite_config": label,
        "check": check["name"],
        "max_abs_diff": check["max_abs_diff"],
        "max_rel_diff": check["max_rel_diff"],
        "n_points": check["n_points"],
    }


def _get_satellite_period_years(root: Path, gas: str, suffixes: tuple[str, str]) -> tuple[int, int] | None:
    """Get the (start, end) years satellite data covers, from whichever config's diagnostics has it"""
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, gas, "interpolation", suffix)
        if d is not None and "satellite_period_start_year" in d:
            return d["satellite_period_start_year"], d["satellite_period_end_year"]
    return None


def get_satellite_period_start_year(root: Path, gas: str, suffixes: tuple[str, str]) -> int | None:
    """Get the first year satellite data covers, from whichever config's diagnostics has it"""
    period = _get_satellite_period_years(root, gas, suffixes)
    return None if period is None else period[0]


def _add_satellite_start_line(ax: plt.Axes, start_year: int | None, show_legend: bool = True) -> None:
    """Add a dashed vertical line marking the first year satellite data covers, and (re-)draw the legend"""
    if start_year is None:
        return
    ax.axvline(
        start_year,
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="satellite data starts" if show_legend else None,
    )
    if show_legend:
        ax.legend(fontsize=8)


def coverage_table(root: Path, gas: str, suffixes: tuple[str, str], labels: dict[str, str]) -> pd.DataFrame:
    """
    Input spatial-coverage diagnostics table (from `1201`/`1101`)

    `bins_populated_per_year_month_mean`/`_min` (out of `spatial_bins_total`,
    72 on our 15x60 degree grid) is the number of *distinct* spatial bins
    with a real (ground or satellite) value each month - the direct measure
    of how much of the grid is actually observed vs left for `griddata` to
    interpolate.

    Shown at two scopes, side by side: the *full record* (every year the
    ground network covers, back to the 1960s/70s), and the *satellite period*
    (only the years the satellite product itself covers, e.g. 2003-2023).
    Satellite data can only ever affect the satellite-period rows - it is
    structurally impossible for it to change anything in earlier years. The
    full-record numbers mix in ~5-6 decades where satellite data can't have
    done anything, which dilutes its real effect, so judge the effect from
    the satellite-period rows.
    """
    rows = []
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, gas, "interpolation", suffix)
        if d is None:
            continue

        if "satellite_period_start_year" in d:
            scope_label = (
                f"satellite period ({d['satellite_period_start_year']}-{d['satellite_period_end_year']})"
            )
            rows.append(
                {
                    "gas": gas,
                    "satellite_config": labels[suffix],
                    "scope": scope_label,
                    "bins_populated_per_year_month_mean": _round_or_none(
                        d.get("bins_populated_per_year_month_mean_satellite_period")
                    ),
                    "bins_populated_per_year_month_min": d.get(
                        "bins_populated_per_year_month_min_satellite_period"
                    ),
                    "spatial_bins_total": d.get("n_spatial_bins_total"),
                }
            )

        rows.append(
            {
                "gas": gas,
                "satellite_config": labels[suffix],
                "scope": "full record",
                "bins_populated_per_year_month_mean": _round_or_none(
                    d.get("bins_populated_per_year_month_mean")
                ),
                "bins_populated_per_year_month_min": d.get("bins_populated_per_year_month_min"),
                "spatial_bins_total": d.get("n_spatial_bins_total"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index(["gas", "scope", "satellite_config"]).sort_index()


def plot_spatial_bin_coverage(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
) -> None:
    """
    Plot which of the grid's spatial bins have real data vs. are left fully interpolated (from `1201`/`1101`)

    For each configuration: the fraction of satellite-period months each of
    the 72 (15x60 degree) bins actually had a real (ground or satellite)
    value in, plus - directly answering "does satellite data fill in gaps
    left by the ground network" - a difference map (`other` minus
    `baseline`). Positive values there are bins that go from
    partly/never-observed to observed more often once satellite data is
    switched on; bins that were already always observed by the ground
    network alone can't move (they're already at 1.0). Coastlines are shown
    for geographic orientation only - our grid is far coarser (15x60 degree
    bins) than the coastline detail suggests.
    """
    baseline, other = suffixes

    maps = {}
    for suffix in suffixes:
        ds = load_nc_diagnostics(root, gas, "interpolation", suffix)
        if ds is not None and "bin_fraction_months_populated_satellite_period" in ds:
            maps[suffix] = ds["bin_fraction_months_populated_satellite_period"]

    if not maps:
        print("No spatial coverage diagnostics found.")
        return

    n_panels = len(maps) + (1 if set(suffixes).issubset(maps) else 0)
    fig, axes = plt.subplots(
        ncols=n_panels, figsize=(5 * n_panels, 4), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes = np.atleast_1d(axes)

    for ax, suffix in zip(axes, suffixes):
        if suffix not in maps:
            continue
        maps[suffix].plot(
            ax=ax,
            x="lon",
            y="lat",
            vmin=0,
            vmax=1,
            cmap="viridis",
            add_colorbar=True,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines()
        ax.set_title(labels[suffix])

    if set(suffixes).issubset(maps):
        diff = maps[other] - maps[baseline]
        diff.plot(
            ax=axes[-1], x="lon", y="lat", cmap="RdBu_r", add_colorbar=True, transform=ccrs.PlateCarree()
        )
        axes[-1].coastlines()
        axes[-1].set_title(f"{labels[other]} minus {labels[baseline]}")

    fig.suptitle(f"{gas.upper()} - fraction of satellite-period months each bin is directly observed")
    plt.tight_layout()
    plt.show()

    period = _get_satellite_period_years(root, gas, suffixes)
    period_str = f"{period[0]}-{period[1]}" if period is not None else "the satellite data period"
    print(
        f"Fraction of months within {period_str} that have an observation in that bin: "
        "without satellite data, with satellite data, and a diff of the two. A value of 1 "
        f"means all months between {period_str} have an observation in that bin, a value of 0 "
        "means no months have an observation in that bin."
    )


def plot_lat_gradient_eofs(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """
    Plot the leading latitudinal-gradient EOF spatial patterns and explained variance

    (observational-network period, from `1202`/`1102`). A EOF pattern that
    visibly changes shape (not just scale) between configurations means
    satellite data is picking up a genuinely different spatial structure,
    not just adding noise.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(15, 4))
    fig.suptitle(f"{gas.upper()} - latitudinal gradient EOFs (observational network)")

    for suffix in suffixes:
        ds = load_nc_diagnostics(root, gas, "obs-network", suffix)
        if ds is None:
            continue
        color = colors[suffix]
        label = labels[suffix]

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


def plot_lat_gradient_pcs_obs_network(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """
    Plot PC0/PC1 over the observational-network years (from `1202`/`1102`)

    A shift here in the satellite-covered years is the direct effect of
    adding satellite data; a shift in *earlier* years too would be a sign
    something unexpected is going on. The dashed vertical line marks the
    first year satellite data covers.
    """
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), sharex=True)
    fig.suptitle(f"{gas.upper()} - latitudinal gradient PCs (observational network)")

    for suffix in suffixes:
        ds = load_nc_diagnostics(root, gas, "obs-network", suffix)
        if ds is None:
            continue
        color = colors[suffix]
        label = labels[suffix]

        for eof in range(min(2, ds.sizes["lat_gradient_eof"])):
            ds["lat_gradient_pcs_full"].sel(lat_gradient_eof=eof).plot(ax=axes[eof], color=color, label=label)
            axes[eof].set_title(f"PC{eof}")

    start_year = get_satellite_period_start_year(root, gas, suffixes)
    for ax in axes:
        _add_satellite_start_line(ax, start_year)
    axes[0].legend()
    plt.tight_layout()
    plt.show()


def plot_seasonality(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """
    Plot the seasonality diagnostics (from `1202`/`1102`)

    CO2 and CH4 handle seasonality differently: CO2 splits it into a fixed
    climatological shape plus a separately EOF-decomposed "seasonality
    change" term; CH4 uses a single fixed *relative* (fractional) seasonal
    cycle with no EOF decomposition at all. This dispatches accordingly.
    Neither has a time/year dimension (both vary by month), so there's no
    satellite-start line to draw here.
    """
    if gas == "co2":
        _plot_seasonality_change_eof_co2(root, suffixes, labels, colors)
    elif gas == "ch4":
        _plot_relative_seasonality_ch4(root, suffixes, labels, colors)
    else:
        raise NotImplementedError(gas)


def _plot_seasonality_change_eof_co2(
    root: Path, suffixes: tuple[str, str], labels: dict[str, str], colors: dict[str, str]
) -> None:
    """
    Left column: PC0, from 1850 onwards (from `1205` - shown from 1850, not
    year 1, since the composite regression's own reference period starts
    there and the driver is assumed constant before it, so nothing earlier
    is informative). Right column: EOF0 by month (from `1202`). Both broken
    down the same way, one row per latitude, for direct row-by-row
    comparison - even though PC0 itself doesn't vary by latitude (it's the
    same panel repeated down the column), only the EOFs do. Bottom row:
    explained variance ratio, spanning the full width, drawn larger since
    it's a single summary rather than one of a per-latitude set.
    """
    datasets = {
        suffix: ds
        for suffix in suffixes
        if (ds := load_nc_diagnostics(root, "co2", "obs-network", suffix)) is not None
    }
    if not datasets:
        print("No CO2 seasonality-change diagnostics found.")
        return

    pc_datasets = {
        suffix: ds
        for suffix in suffixes
        if (ds := load_nc_diagnostics(root, "co2", "seasonality-extend", suffix)) is not None
    }

    lats = next(iter(datasets.values()))["lat"].values
    n_lats = len(lats)

    fig = plt.figure(figsize=(11, 2.3 * n_lats + 2))
    gs = fig.add_gridspec(
        n_lats + 1,
        2,
        height_ratios=[1] * n_lats + [2.5],
        hspace=1.1,
        wspace=0.3,
        top=0.95,
        bottom=0.02,
        left=0.08,
        right=0.97,
    )

    start_year = get_satellite_period_start_year(root, "co2", suffixes)

    ax_pc0_first = None
    ax_eof_first = None
    for i, lat in enumerate(lats):
        # Left column: PC0, all years - repeated every row (it doesn't vary
        # by latitude, unlike the EOFs), so each row can be compared directly.
        ax_pc0 = fig.add_subplot(gs[i, 0], sharex=ax_pc0_first)
        ax_pc0_first = ax_pc0_first or ax_pc0
        for suffix, ds in pc_datasets.items():
            ds["principal-components"].sel(eof=0, year=slice(1850, None)).plot(
                ax=ax_pc0, color=colors[suffix], label=labels[suffix]
            )
        if not pc_datasets:
            ax_pc0.text(0.5, 0.5, "No 1205 diagnostics found", ha="center", va="center", fontsize=8)
        ax_pc0.set_title(f"PC0, lat={lat}", fontsize=9)
        ax_pc0.set_xlabel("")
        ax_pc0.set_ylabel("")
        _add_satellite_start_line(ax_pc0, start_year, show_legend=(i == 0))
        if i == 0:
            ax_pc0.legend(fontsize=7)  # re-draw at the smaller fontsize used elsewhere in this figure
        if i == n_lats - 1:
            ax_pc0.set_xlabel("Year")

        # Right column: EOF0 by month, one row per latitude.
        ax_eof = fig.add_subplot(gs[i, 1], sharex=ax_eof_first)
        ax_eof_first = ax_eof_first or ax_eof

        for suffix, ds in datasets.items():
            ds["seasonality_change_eofs_full"].sel(seasonality_change_eof=0).sel(
                lat=lat, method="nearest"
            ).plot(ax=ax_eof, color=colors[suffix], label=labels[suffix])
        ax_eof.axhline(0, color="k", linestyle=":", linewidth=1)
        ax_eof.set_title(f"EOF0, lat={lat}", fontsize=9)
        ax_eof.set_xlabel("")
        ax_eof.set_ylabel("ppm", fontsize=8)
        if i == 0:
            ax_eof.legend(fontsize=7)
        if i == n_lats - 1:
            ax_eof.set_xlabel("Month")

    # Bottom row: explained variance ratio, spanning both columns, bigger.
    ax_var = fig.add_subplot(gs[n_lats, :])
    n_show = min(6, next(iter(datasets.values())).sizes["seasonality_change_eof"])
    for suffix, ds in datasets.items():
        ax_var.plot(
            range(n_show),
            ds["seasonality_change_explained_variance_ratio"].isel(seasonality_change_eof=slice(0, n_show)),
            "o-",
            color=colors[suffix],
            label=labels[suffix],
            markersize=9,
            linewidth=2,
        )
    ax_var.set_title("Explained variance ratio", fontsize=14)
    ax_var.set_xlabel("EOF index")
    ax_var.legend(fontsize=10)

    fig.suptitle("CO2 - seasonality-change: PC0, EOF0 by latitude, and explained variance ratio")
    plt.show()


def _plot_relative_seasonality_ch4(
    root: Path, suffixes: tuple[str, str], labels: dict[str, str], colors: dict[str, str]
) -> None:
    """5 subplots (one row), one per representative latitude, shared y-axis, dotted zero line."""
    example_lats = [-82.5, -37.5, 7.5, 52.5, 82.5]

    datasets = {
        suffix: ds
        for suffix in suffixes
        if (ds := load_nc_diagnostics(root, "ch4", "obs-network", suffix)) is not None
    }
    if not datasets:
        print("No CH4 seasonality diagnostics found.")
        return

    fig, axes = plt.subplots(ncols=len(example_lats), figsize=(4 * len(example_lats), 4), sharey=True)

    for ax, lat in zip(axes, example_lats):
        for suffix, ds in datasets.items():
            da = ds["relative_seasonality_obs_network"].sel(lat=lat, method="nearest")
            ax.plot(da["month"], da, color=colors[suffix], label=labels[suffix])
        ax.axhline(0, color="k", linestyle=":", linewidth=1)
        ax.set_title(f"lat={lat}")
        ax.set_xlabel("Month")

    axes[0].set_ylabel("Relative seasonal anomaly")
    axes[0].legend(fontsize=8)
    fig.suptitle("CH4 relative seasonality by latitude (shared y-axis)")
    plt.tight_layout()
    plt.show()


def _diff_plot(
    series: dict[str, xr.DataArray],
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
    title: str,
    satellite_start_year: int | None = None,
) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle(title)

    for suffix in suffixes:
        if suffix not in series:
            continue
        series[suffix].plot(ax=axes[0], color=colors[suffix], label=labels[suffix])

    axes[0].legend()

    if set(suffixes).issubset(series):
        baseline, other = suffixes
        common_years = np.intersect1d(series[baseline]["year"], series[other]["year"])
        diff = series[other].sel(year=common_years) - series[baseline].sel(year=common_years)
        diff.plot(ax=axes[1])
        axes[1].axhline(0, color="k", linewidth=0.7)
        axes[1].set_title(f"{labels[other]} minus {labels[baseline]}")

    for ax in axes:
        _add_satellite_start_line(ax, satellite_start_year)

    plt.tight_layout()
    plt.show()


def plot_global_mean_obs_network(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """
    Global-, annual-mean over the observational-network period (from `1202`/`1102`)

    There's no EOF to compare here - it's a scalar-per-year timeseries - so
    we plot it directly, plus the difference between configurations. The
    dashed vertical line marks the first year satellite data covers.
    """
    series = {}
    for suffix in suffixes:
        ds = load_nc_diagnostics(root, gas, "obs-network", suffix)
        if ds is not None:
            series[suffix] = ds["global_annual_mean_obs_network"]

    start_year = get_satellite_period_start_year(root, gas, suffixes)
    _diff_plot(
        series,
        suffixes,
        labels,
        colors,
        f"{gas.upper()} - global-, annual-mean (observational network)",
        satellite_start_year=start_year,
    )


def plot_lat_gradient_extend_pcs(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """
    PC0/PC1 extended to all years (from `1203`/`1103`), plus EOF0/EOF1
    underneath, each shown twice - once over the full record and once
    zoomed in from 1850 onwards.

    Extended using a regression against PRIMAP fossil emissions (both gases),
    plus - CH4 only - a joint optimisation against the NEEM and Law Dome ice
    cores for the gap closest to the observational network.

    EOF0/EOF1 only vary by latitude, not year - this extension step doesn't
    recompute them, it just carries the observational-network fit's spatial
    patterns forward unchanged - so the EOF row is identical between the two
    zoom levels; it's repeated anyway to keep a direct visual pairing with
    the PC0/PC1 row alongside it in each block. The dashed vertical line
    marks the first year satellite data covers.
    """
    datasets = {
        suffix: ds
        for suffix in suffixes
        if (ds := load_nc_diagnostics(root, gas, "lat-gradient-extend", suffix)) is not None
    }
    if not datasets:
        print(f"No {gas.upper()} lat-gradient-extend diagnostics found.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle(f"{gas.upper()} - latitudinal gradient PCs (extended) and EOFs")

    start_year = get_satellite_period_start_year(root, gas, suffixes)

    def plot_pc_row(row: int, year_slice: tuple[int, None] | None, zoom_label: str) -> None:
        for eof in range(2):
            ax = axes[row, eof]
            for suffix, ds in datasets.items():
                da = ds["principal-components"].sel(eof=eof)
                if year_slice is not None:
                    da = da.sel(year=slice(*year_slice))
                da.plot(ax=ax, color=colors[suffix], label=labels[suffix])
            ax.set_title(f"PC{eof}, {zoom_label}")
            _add_satellite_start_line(ax, start_year, show_legend=(row == 0 and eof == 0))

    def plot_eof_row(row: int) -> None:
        for eof in range(2):
            ax = axes[row, eof]
            for suffix, ds in datasets.items():
                ax.plot(ds["eofs"].sel(eof=eof), ds["lat"], color=colors[suffix], label=labels[suffix])
            ax.set_title(f"EOF{eof} (static - doesn't vary by year)")
            ax.set_xlabel(f"{gas.upper()} anomaly")
            ax.set_ylabel("Latitude")

    plot_pc_row(0, None, "full record")
    plot_eof_row(1)
    plot_pc_row(2, (1850, None), "from 1850")
    plot_eof_row(3)

    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def regression_table(root: Path, gas: str, suffixes: tuple[str, str], labels: dict[str, str]) -> pd.DataFrame:
    """
    PC0-vs-emissions regression quality (from `1203`/`1103`)

    The linear regression of PC0 against PRIMAP fossil emissions that fills
    the gap between the ice-core-derived period and the observational
    network. `r2` close to 1 means PC0 is well explained by emissions alone;
    a large change between configurations means satellite data changed how
    well that relationship holds.
    """
    rows = []
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, gas, "lat-gradient-extend", suffix)
        if d is None:
            continue
        rows.append(
            {
                "gas": gas,
                "satellite_config": labels[suffix],
                "m": d["pc0_emissions_regression_m"],
                "c": d["pc0_emissions_regression_c"],
                "r2": d["pc0_emissions_regression_r2"],
                "n_years_in_regression": d["pc0_emissions_regression_n_years"],
                "n_years_filled_with_regression": d["n_years_filled_with_regression"],
            }
        )

    return pd.DataFrame(rows).set_index(["gas", "satellite_config"])


def plot_global_mean_allyears(
    root: Path,
    gas: str,
    suffixes: tuple[str, str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """
    All-years extended global-, annual-mean (from `1204`/`1104`), plus the difference.

    The dashed vertical line marks the first year satellite data covers.
    """
    series = {}
    for suffix in suffixes:
        ds = load_nc_diagnostics(root, gas, "global-mean-extend", suffix)
        if ds is not None:
            series[suffix] = ds["global_annual_mean_allyears"]

    start_year = get_satellite_period_start_year(root, gas, suffixes)
    _diff_plot(
        series,
        suffixes,
        labels,
        colors,
        f"{gas.upper()} - global-, annual-mean, all years",
        satellite_start_year=start_year,
    )


def neem_tolerance_check_table(root: Path, suffixes: tuple[str, str], labels: dict[str, str]) -> pd.DataFrame:
    """
    CH4-only: the NEEM check's actual value against the tolerance it's judged by (from `1104`)

    Does the fully reconstructed, all-years field still match the raw NEEM
    ice-core measurements, at NEEM's own years/latitude? It's checked with
    `np.testing.assert_allclose(..., rtol=neem_rtol)`, and `neem_rtol` is
    deliberately widened from 1e-3 to 2e-3 when satellite data is on (see the
    comment above `neem_rtol` in `1104_ch4_extend-global-annual-mean.py`),
    because satellite data shifts the EOFs this check depends on. This table
    makes that trade-off directly visible: a `neem_max_rel_diff` between the
    two columns means the run would fail the *unwidened* tolerance but is
    accepted under the one actually used.
    """
    tight_rtol = 1e-3
    widened_rtol = 2e-3

    rows = []
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, "ch4", "global-mean-extend", suffix)
        if d is None or "neem_check" not in d:
            continue

        max_rel_diff = d["neem_check"]["max_rel_diff"]
        rows.append(
            {
                "satellite_config": labels[suffix],
                "neem_max_rel_diff": max_rel_diff,
                "neem_rtol_used": d.get("neem_rtol_used"),
                "passes_at_tight_rtol_1e-3": None if max_rel_diff is None else max_rel_diff < tight_rtol,
                "passes_at_widened_rtol_2e-3": None if max_rel_diff is None else max_rel_diff < widened_rtol,
            }
        )

    return pd.DataFrame(rows).set_index("satellite_config")


def checks_table(root: Path, gas: str, suffixes: tuple[str, str], labels: dict[str, str]) -> pd.DataFrame:
    """
    Correctness-check residuals (from `1204`/`1104`)

    `max_abs_diff`/`max_rel_diff`: the largest difference between the
    pipeline's reconstruction and independent reference data (an ice core,
    or another gas-specific dataset). `field_decomposition` isn't a
    comparison to external data - it checks that
    `full_field - global_annual_mean` exactly reproduces the latitudinal
    gradient computed earlier; should stay at ~machine precision regardless
    of satellite configuration.
    """
    rows = []
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, gas, "global-mean-extend", suffix)
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
                row = _check_to_row(gas, labels[suffix], d[check_name])
                if row is not None:
                    rows.append(row)

    return pd.DataFrame(rows).set_index(["gas", "satellite_config", "check"])


def seam_table(root: Path, gas: str, suffixes: tuple[str, str], labels: dict[str, str]) -> pd.DataFrame:
    """
    Harmonisation seam sizes (from `1204`/`1104`)

    How big a jump the global-, annual-mean takes right at each point where
    one data source is stitched onto another, compared to a typical
    year-on-year step elsewhere. Satellite data only affects the
    observational-network side of these joins.
    """
    rows = []
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, gas, "global-mean-extend", suffix)
        if d is None:
            continue
        row = {
            "gas": gas,
            "satellite_config": labels[suffix],
            "typical_year_on_year_step": d["typical_year_on_year_step"],
        }
        if gas == "co2":
            row["mauna_loa_seam_jump"] = d["mauna_loa_seam_jump"]
            row["menking_seam_jump"] = d["menking_seam_jump"]
        else:
            row["law_dome_seam_jump"] = d["law_dome_seam_jump"]
        rows.append(row)

    return pd.DataFrame(rows).set_index(["gas", "satellite_config"])


def plot_co2_seasonality_extend(
    root: Path, suffixes: tuple[str, str], labels: dict[str, str], colors: dict[str, str]
) -> None:
    """
    CO2-only: seasonality-change PC0 and EOF0, extended to all years (from `1205`). No CH4 equivalent.

    Left column: PC0, shown twice - full record, and zoomed in from 1850
    onwards (the composite regression's own reference period starts there -
    see `CO2SeasonalityChangeRegression` - and the driver is assumed
    constant before it, so nothing earlier is informative). Right column:
    EOF0 by month, one line per latitude - doesn't vary by year, so it's the
    same panel in both rows, kept for direct visual pairing with PC0
    alongside it. The dashed vertical line marks the first year satellite
    data covers.
    """
    datasets = {
        suffix: ds
        for suffix in suffixes
        if (ds := load_nc_diagnostics(root, "co2", "seasonality-extend", suffix)) is not None
    }
    if not datasets:
        print("No CO2 seasonality-change (1205) diagnostics found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CO2 - seasonality-change PC0 and EOF0")

    start_year = get_satellite_period_start_year(root, "co2", suffixes)

    for row, (year_slice, zoom_label) in enumerate([(None, "full record"), ((1850, None), "from 1850")]):
        ax_pc, ax_eof = axes[row, 0], axes[row, 1]

        for suffix, ds in datasets.items():
            da = ds["principal-components"].sel(eof=0)
            if year_slice is not None:
                da = da.sel(year=slice(*year_slice))
            da.plot(ax=ax_pc, color=colors[suffix], label=labels[suffix])
        ax_pc.set_title(f"PC0, {zoom_label}")
        _add_satellite_start_line(ax_pc, start_year, show_legend=(row == 0))

        for suffix, ds in datasets.items():
            ds["eofs"].sel(eof=0).plot.line(
                ax=ax_eof, hue="lat", add_legend=False, color=colors[suffix], alpha=0.6
            )
        ax_eof.set_title(f"EOF0 by month, one line per latitude ({zoom_label}, static)")

    plt.tight_layout()
    plt.show()


def co2_seasonality_extend_regression_table(
    root: Path, suffixes: tuple[str, str], labels: dict[str, str]
) -> pd.DataFrame:
    """CO2-only: PC0-vs-composite-timeseries regression quality (from `1205`)"""
    rows = []
    for suffix in suffixes:
        d = load_yaml_diagnostics(root, "co2", "seasonality-extend", suffix)
        if d is None:
            continue
        rows.append(
            {
                "satellite_config": labels[suffix],
                "m": d["pc0_composite_regression_m"],
                "c": d["pc0_composite_regression_c"],
                "r2": d["pc0_composite_regression_r2"],
                "n_years_in_regression": d["pc0_composite_regression_n_years"],
            }
        )
    return pd.DataFrame(rows).set_index("satellite_config")


# --- Gridded (input4MIPs, `40yy_write-input4mips`) output diffing ---
#
# Shared between `check_{gas}_output_all_fits.py` (loops over every fit found)
# and the last section of each `compare_{gas}_{FIT}.py` notebook (just its own
# fit). All of this needs the `40yy` gridding/writing step to have actually
# been run for the fit(s) in question - functions here return/print an
# explanation and do nothing rather than raising if it hasn't.

_GRIDDED_FILENAME_DATE_RANGE_RE = r"\d+-\d+"


def get_esgf_ready_gas_dir(output_bundles_root: Path, run_id: str, gas: str) -> Path:
    """Path to a gas's `gnz` (zonal-mean) input4MIPs output directory, written by `40yy_write-input4mips`"""
    return (
        output_bundles_root
        / run_id
        / "data/processed/esgf-ready/input4MIPs/CMIP6Plus/CMIP/CR/CR-CMIP-testing/atmos/mon"
        / gas
        / "gnz"
    )


def find_version_dir(esgf_ready_gas_dir: Path, version: str | None = None) -> Path | None:
    """Find the version folder to use: `version` if given and it exists, else the most recently created `v*` folder"""
    if version is not None:
        version_dir = esgf_ready_gas_dir / version
        return version_dir if version_dir.exists() else None

    if not esgf_ready_gas_dir.exists():
        return None

    version_dirs = sorted((p for p in esgf_ready_gas_dir.glob("v*") if p.is_dir()), key=lambda p: p.name)
    return version_dirs[-1] if version_dirs else None


def discover_gridded_files(version_dir: Path | None, gas: str) -> tuple[list[Path], dict[str, list[Path]]]:
    """Find the no-satellite baseline chunks and the per-fit chunks in a version folder"""
    if version_dir is None or not version_dir.exists():
        return [], {}

    prefix = f"{gas}_input4MIPs_GHGConcentrations_CMIP_CR-CMIP-testing_gnz_"
    fit_re = re.compile(rf"^{re.escape(prefix)}{_GRIDDED_FILENAME_DATE_RANGE_RE}_SAT_(?P<fit>.+)\.nc$")
    baseline_re = re.compile(rf"^{re.escape(prefix)}{_GRIDDED_FILENAME_DATE_RANGE_RE}\.nc$")

    baseline_chunks: list[Path] = []
    fit_chunks: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(version_dir.glob(f"{prefix}*.nc")):
        fit_match = fit_re.match(path.name)
        if fit_match:
            fit_chunks[fit_match.group("fit")].append(path)
        elif baseline_re.match(path.name):
            baseline_chunks.append(path)

    # Sort chunks chronologically - the "start-end" range in the filename sorts
    # lexicographically the same as chronologically, so sorting by path name works.
    baseline_chunks.sort(key=lambda p: p.name)
    for chunks in fit_chunks.values():
        chunks.sort(key=lambda p: p.name)

    return baseline_chunks, dict(fit_chunks)


def load_concatenated_gridded(chunk_paths: list[Path], gas: str) -> xr.DataArray | None:
    """Load and concatenate the time-range chunks for one gridded output into a single timeseries"""
    if not chunk_paths:
        return None

    # `use_cftime=True` for every chunk (not just the ones that strictly need it,
    # e.g. years before ~1678 which don't fit in a datetime64[ns]) so that
    # concatenating chunks produces a single, uniform `CFTimeIndex` - mixing
    # cftime and datetime64 chunks otherwise silently falls back to a generic
    # object-dtype time axis that xarray/matplotlib can't plot.
    return xr.concat([xr.open_dataset(p, use_cftime=True)[gas] for p in chunk_paths], dim="time")


def gridded_diff_from_baseline(baseline_da: xr.DataArray, fit_da: xr.DataArray) -> xr.DataArray:
    """Difference between one fit's gridded output and the baseline, on their shared time axis"""
    common_time = np.intersect1d(baseline_da["time"], fit_da["time"])
    return fit_da.sel(time=common_time) - baseline_da.sel(time=common_time)


def plot_gridded_diff_for_fit(
    output_bundles_root: Path, run_id: str, gas: str, fit: str, version: str | None = None
) -> None:
    """
    Plot the final gridded (input4MIPs) diff for this notebook's own fit vs. the no-satellite baseline

    The "bottom line" answer, downstream of everything else in this
    notebook: same Hovmöller-style (time x latitude) diff as
    `check_{gas}_output_all_fits.py`'s "Per-fit diffs over time" section, but
    scoped to just this one fit. Needs the `40yy_write-input4mips` step to
    have actually been run for this fit; prints an explanation and does
    nothing if it hasn't.
    """
    esgf_ready_gas_dir = get_esgf_ready_gas_dir(output_bundles_root, run_id, gas)
    version_dir = find_version_dir(esgf_ready_gas_dir, version)
    baseline_chunks, fit_chunks = discover_gridded_files(version_dir, gas)

    if not baseline_chunks or fit not in fit_chunks:
        print(
            f"No gridded (40yy_write-input4mips) output found for {gas}/{fit} under "
            f"{esgf_ready_gas_dir} yet - nothing to diff."
        )
        return

    baseline_da = load_concatenated_gridded(baseline_chunks, gas)
    fit_da = load_concatenated_gridded(fit_chunks[fit], gas)
    diff = gridded_diff_from_baseline(baseline_da, fit_da)

    fig, ax = plt.subplots(figsize=(10, 3))
    diff.plot(ax=ax, x="time", y="lat", cmap="RdBu_r")
    ax.set_title(f"{gas.upper()} {fit} - final gridded output minus no-satellite baseline")
    plt.tight_layout()
    plt.show()

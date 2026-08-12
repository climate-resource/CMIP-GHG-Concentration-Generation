"""
Generate one satellite-vs-no-satellite diagnostics notebook per (gas, fit)

Reads the available scaled satellite fits straight from `data/raw/scaled_sat/`
(so re-running this script after a new fit is downloaded picks it up
automatically) and writes one thin notebook per (gas, fit) under
`notebooks/diagnostics/{CO2,CH4}/`. All the actual loading/plotting logic
lives in `local.diagnostics_reporting` - these notebooks just call it, so
adding a diagnostic there makes it show up in every generated notebook
without needing to regenerate anything.

This script always overwrites what it generates, and also deletes any
previously-generated notebook whose fit is no longer discovered or has since
been excluded (see `EXCLUDED_FITS`), so it's always safe to just re-run this
after changing `EXCLUDED_FITS` or downloading new fits - don't hand-delete or
hand-edit files in `notebooks/diagnostics/` if you want the change to stick.

Usage
-----
    pixi run python scripts/generate_satellite_diagnostics_notebooks.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RAW_SCALED_SAT_DIR = REPO_ROOT / "output-bundles" / "dev-test-run" / "data" / "raw" / "scaled_sat"
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks" / "diagnostics"

GASES = {"co2": "CO2", "ch4": "CH4"}

# Fits we don't want a comparison notebook for at all:
# - "ML_FIT": excluded per request, not to be plotted/compared.
# - anything ending "UNC_WEIGHT_FIT": excluded per request.
EXCLUDED_FITS = {"ML_FIT"}


def is_excluded(fit: str) -> bool:
    """Whether a fit should be skipped entirely"""
    return fit in EXCLUDED_FITS or fit.endswith("UNC_WEIGHT_FIT")


# Matches e.g. "200301_202312-C3S-L3_XCO2-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.6_LINEAR_FIT.nc"
FIT_FILENAME_RE = re.compile(r"MERGED-v4\.6_(?P<fit>.+)\.nc$")


def discover_fits(gas: str) -> list[str]:
    """Discover available (non-excluded) satellite fits for a gas from the raw data directory"""
    pattern = f"*X{gas.upper()}-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.6_*.nc"
    fits = []
    for path in sorted(RAW_SCALED_SAT_DIR.glob(pattern)):
        match = FIT_FILENAME_RE.search(path.name)
        if match and not is_excluded(match.group("fit")):
            fits.append(match.group("fit"))

    return sorted(fits)


def render_notebook(gas: str, fit: str) -> str:
    """Render the jupytext source for a single (gas, fit) comparison notebook"""
    is_co2 = gas == "co2"
    is_ch4 = gas == "ch4"
    interp_nb = (
        "1201_co2_interpolate-observational-network"
        if is_co2
        else "1101_ch4_interpolate-observational-network"
    )
    obs_network_nb = (
        "1202_co2_observational-network-global-mean-latitudinal-gradient-seasonality"
        if is_co2
        else "1102_ch4_global-mean-latitudinal-gradient-seasonality"
    )
    extend_pcs_nb = "1203_co2_extend-lat-gradient-pcs" if is_co2 else "1103_ch4_extend-pcs"
    extend_mean_nb = "1204_co2_extend-global-annual-mean" if is_co2 else "1104_ch4_extend-global-annual-mean"

    neem_tolerance_cell = (
        """
# %% [markdown]
# ### CH4: NEEM check vs. its tolerance
#
# **Derived in:** `1104_ch4_extend-global-annual-mean`
#
# The headline number for this section: how far off is the fully
# reconstructed field from the raw NEEM ice-core measurements, and does that
# fall inside the tolerance the pipeline actually checks it against?
# `neem_rtol` is deliberately widened from 1e-3 to 2e-3 when satellite data is
# switched on (satellite data shifts the EOFs this check depends on, even
# for years centuries before satellite coverage began) - `neem_max_rel_diff`
# landing between the two `passes_at_*` columns means this run only clears
# the check because of that widening.

# %%
report.neem_tolerance_check_table(diagnostics_root_path, suffixes, labels)
"""
        if is_ch4
        else ""
    )

    seasonality_description = (
        """# CO2's seasonality diagnostics track how the *shape* of the seasonal cycle
# changes over time, not just the cycle itself: each year's monthly anomaly
# (month value minus a smoothed annual mean, per latitude) is compared
# against the multi-year-average seasonal cycle, and the year-to-year
# deviations from that average are decomposed via SVD across `lat x month`
# into EOFs (spatial-monthly patterns) and PCs (per-year weights) - see
# `local.seasonality.calculate_seasonality_change_eofs_pcs`. The plot below
# shows PC0 (extended back to 1850 in `1205`) and EOF0 by latitude - EOF1
# onwards are computed and saved (see the explained variance ratio panel)
# but not plotted here."""
        if is_co2
        else """# CH4's seasonality diagnostic is a single fixed climatological cycle, not a
# year-varying decomposition: monthly anomalies (month minus a smoothed
# annual mean, per latitude) are averaged across all years, then divided by
# the global annual mean to express them as a fraction - see
# `local.seasonality.calculate_seasonality`. The plot below shows this
# relative seasonality directly, at five representative latitudes."""
    )

    co2_seasonality_extend_section = (
        """
# %% [markdown]
# ## 7. CO2: seasonality-change extension (all years)
#
# **Derived in:** `1205_co2_extend-seasonality-change-pcs`
#
# CO2-only - CH4 has no seasonality-change EOFs to extend (see section 3).

# %%
report.plot_co2_seasonality_extend(diagnostics_root_path, suffixes, labels, colors)

# %%
report.co2_seasonality_extend_regression_table(diagnostics_root_path, suffixes, labels)
"""
        if is_co2
        else ""
    )

    return f"""# ---
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
# # {gas.upper()} satellite diagnostics: {fit}
#
# Compares `{gas}` pipeline diagnostics with satellite data (`{fit}` fit)
# against the no-satellite baseline, using whatever the `12yy`/`11yy`
# notebooks saved under `data/diagnostics/{gas}/`. If you haven't run the
# pipeline for this fit yet, the affected sections below will just come back
# empty rather than erroring.
#
# **Generated file - do not hand-edit.** Regenerate with
# `scripts/generate_satellite_diagnostics_notebooks.py` after changing
# `local.diagnostics_reporting` or the template in that script.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import local.diagnostics_reporting as report

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
run_id: str = "dev-test-run"
gas: str = "{gas}"
satellite_fit: str = "{fit}"
diagnostics_root: str = "../../../output-bundles/{{run_id}}/data/diagnostics"
output_bundles_root: str = "../../../output-bundles"

# %%
diagnostics_root_path = Path(diagnostics_root.format(run_id=run_id))
output_bundles_root_path = Path(output_bundles_root)
satellite_suffix = f"SAT_{{satellite_fit}}"
suffixes = ("nosat", satellite_suffix)
labels = {{"nosat": "No satellite data", satellite_suffix: f"Satellite data ({{satellite_fit}})"}}
colors = {{"nosat": "tab:blue", satellite_suffix: "tab:orange"}}
diagnostics_root_path

# %% [markdown]
# ## 1. Input coverage
#
# **Derived in:** `{interp_nb}`
#
# How much of the grid is spatial bins with real data, vs. left for
# `griddata` to interpolate? Shown at two scopes: the *satellite period*
# (only the years the satellite product covers - satellite data cannot
# possibly affect anything outside it), and the *full record*, which dilutes
# satellite data's real effect across ~5-6 decades it can't touch - judge the
# effect from the satellite-period rows.

# %%
report.coverage_table(diagnostics_root_path, gas, suffixes, labels)

# %%
report.plot_spatial_bin_coverage(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 2. Latitudinal gradient EOFs (observational-network period)
#
# **Derived in:** `{obs_network_nb}`
#
# The latitudinal gradient is decomposed into Empirical Orthogonal Functions
# (EOFs, spatial north-south patterns) and Principal Components (PCs, how
# strongly each pattern is expressed each year). Satellite data feeds
# directly into this decomposition. A EOF pattern that visibly changes shape
# (not just scale) between configurations means satellite data is picking up
# a genuinely different spatial structure, not just adding noise.

# %%
report.plot_lat_gradient_eofs(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.plot_lat_gradient_pcs_obs_network(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# Combining the PC and EOF values: how much does the *product* change, not
# just each side individually? Shown as a Hovmoeller (year x latitude) plot
# of the reconstructed field from the leading two modes.

# %%
report.plot_lat_gradient_reconstruction(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 3. Seasonality (observational-network period)
#
# **Derived in:** `{obs_network_nb}`
#
{seasonality_description}

# %%
report.plot_seasonality(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 4. Global-, annual-mean (observational-network period)
#
# **Derived in:** `{obs_network_nb}`
#
# No EOF to compare here - it's a scalar-per-year timeseries.

# %%
report.plot_global_mean_obs_network(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 5. Latitudinal gradient extension (all years)
#
# **Derived in:** `{extend_pcs_nb}`
#
# The PCs above, extended back to year 1 using a regression against PRIMAP
# fossil emissions{" (plus, for CH4, a joint ice-core optimisation)" if is_ch4 else ""}.

# %%
report.plot_lat_gradient_extend_pcs(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.regression_table(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 6. Global-, annual-mean extension (all years) and correctness checks
#
# **Derived in:** `{extend_mean_nb}`
#
# Where the ice-core-derived history gets stitched onto the
# observational-network period, and where the pipeline's own correctness
# checks live. These normally only ever raise if they fail; here we print
# the actual numbers so a *passing* check that got noticeably worse is still
# visible.
{neem_tolerance_cell}
# %%
report.plot_global_mean_allyears(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.checks_table(diagnostics_root_path, gas, suffixes, labels)

# %%
report.seam_table(diagnostics_root_path, gas, suffixes, labels)
{co2_seasonality_extend_section}
# %% [markdown]
# ## {"8" if is_co2 else "7"}. Final gridded output diff
#
# **Derived in:** `40yy_write-input4mips` (the final gridding/ESGF-ready
# output step, downstream of everything else in this notebook).
#
# The bottom line: `{fit}` minus the no-satellite baseline, in the actual
# final input4MIPs product, not just the internal pieces above. Needs the
# `40yy` step to have been run for this fit - if it hasn't, this section
# prints a note and shows nothing rather than erroring. For a view across
# *all* fits at once, see `check_{gas}_output_all_fits.py` in this folder.

# %%
report.plot_gridded_diff_for_fit(output_bundles_root_path, run_id, gas, satellite_fit)
"""


def main() -> None:
    for gas, gas_dir_name in GASES.items():
        gas_dir = NOTEBOOKS_ROOT / gas_dir_name
        gas_dir.mkdir(parents=True, exist_ok=True)

        fits = discover_fits(gas)
        if not fits:
            print(f"No satellite fits found for {gas}, skipping")

        wanted_paths = set()
        for fit in fits:
            out_path = gas_dir / f"compare_{gas}_{fit}.py"
            out_path.write_text(render_notebook(gas, fit))
            wanted_paths.add(out_path)
            print(f"Wrote {out_path.relative_to(REPO_ROOT)}")

        # Remove any previously-generated per-fit notebook that's no longer wanted
        # (fit excluded, or its raw data file no longer present).
        for existing in gas_dir.glob(f"compare_{gas}_*.py"):
            if existing not in wanted_paths:
                existing.unlink()
                existing_ipynb = existing.with_suffix(".ipynb")
                if existing_ipynb.exists():
                    existing_ipynb.unlink()
                print(f"Removed {existing.relative_to(REPO_ROOT)} (no longer wanted)")


if __name__ == "__main__":
    main()

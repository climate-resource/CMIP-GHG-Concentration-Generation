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
# # CO2 satellite-period diagnostics: LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT
#
# Same comparison as `compare_co2_LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT.py` in `notebooks/diagnostics/CO2/`,
# but both configurations here have the ground-based observational network
# restricted to 2003 onwards (`year_drop_observational_data_before_and_including
# = 2002` - see `scripts/write-sat-period-config.py`), so this isolates the
# effect of adding satellite data *given* a shorter, more recent
# observational record, rather than the full multi-decade one. Compare
# against `../CO2/compare_co2_LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT.py` to see the difference restricting
# the record makes.
#
# **Generated file - do not hand-edit.** Regenerate with
# `scripts/generate_sat_period_diagnostics_notebooks.py` after changing
# `local.diagnostics_reporting` or the template in that script.
#
# **Disposable**: this notebook, this generator, and the
# `dev-test-run-sat-period` output bundle are all meant to be deleted
# together once this check is done.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import local.diagnostics_reporting as report

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
run_id: str = "dev-test-run-sat-period"
gas: str = "co2"
satellite_fit: str = "LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT"
diagnostics_root: str = "../../../../output-bundles/{run_id}/data/diagnostics"
output_bundles_root: str = "../../../../output-bundles"

# %%
diagnostics_root_path = Path(diagnostics_root.format(run_id=run_id))
output_bundles_root_path = Path(output_bundles_root)
satellite_suffix = f"SAT_{satellite_fit}"
suffixes = ("nosat", satellite_suffix)
labels = {
    "nosat": "No satellite data (2003+ only)",
    satellite_suffix: f"Satellite data ({satellite_fit}, 2003+ only)",
}
colors = {"nosat": "tab:blue", satellite_suffix: "tab:orange"}
diagnostics_root_path

# %% [markdown]
# ## 1. Input coverage
#
# **Derived in:** `1201_co2_interpolate-observational-network`
#
# How much of the grid is spatial bins with real data, vs. left for
# `griddata` to interpolate? Both configurations only have 2003-2023 to work
# with here, unlike the main comparison notebooks.

# %%
report.coverage_table(diagnostics_root_path, gas, suffixes, labels)

# %%
report.plot_spatial_bin_coverage(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 2. Latitudinal gradient EOFs (observational-network period)
#
# **Derived in:** `1202_co2_observational-network-global-mean-latitudinal-gradient-seasonality`
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
# **Derived in:** `1202_co2_observational-network-global-mean-latitudinal-gradient-seasonality`
#
# CO2's seasonality diagnostics track how the *shape* of the seasonal cycle
# changes over time, not just the cycle itself: each year's monthly anomaly
# (month value minus a smoothed annual mean, per latitude) is compared
# against the multi-year-average seasonal cycle, and the year-to-year
# deviations from that average are decomposed via SVD across `lat x month`
# into EOFs (spatial-monthly patterns) and PCs (per-year weights) - see
# `local.seasonality.calculate_seasonality_change_eofs_pcs`. The plot below
# shows PC0 (extended back to 1850 in `1205`) and EOF0 by latitude - EOF1
# onwards are computed and saved (see the explained variance ratio panel)
# but not plotted here.

# %%
report.plot_seasonality(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 4. Global-, annual-mean (observational-network period)
#
# **Derived in:** `1202_co2_observational-network-global-mean-latitudinal-gradient-seasonality`
#
# No EOF to compare here - it's a scalar-per-year timeseries.

# %%
report.plot_global_mean_obs_network(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 5. Latitudinal gradient extension (all years)
#
# **Derived in:** `1203_co2_extend-lat-gradient-pcs`
#
# The PCs above, extended back to year 1 using a regression against PRIMAP
# fossil emissions. This regression only anchors on the (now 2003-2023-only)
# observational-network PCs above - the ice-core/PRIMAP inputs it regresses
# against are untouched by the sat-period restriction.

# %%
report.plot_lat_gradient_extend_pcs(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.regression_table(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 6. Global-, annual-mean extension (all years) and correctness checks
#
# **Derived in:** `1204_co2_extend-global-annual-mean`
#
# Where the ice-core-derived history gets stitched onto the
# observational-network period, and where the pipeline's own correctness
# checks live. These normally only ever raise if they fail; here we print
# the actual numbers so a *passing* check that got noticeably worse is still
# visible. Unlike CH4, CO2's harmonisation here joins to the continuous
# Mauna Loa/Scripps record (not an ice core), so it isn't affected by the
# gap that made restricting CH4 to 2003+ infeasible for this experiment.

# %%
report.plot_global_mean_allyears(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.checks_table(diagnostics_root_path, gas, suffixes, labels)

# %%
report.seam_table(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 7. CO2: seasonality-change extension (all years)
#
# **Derived in:** `1205_co2_extend-seasonality-change-pcs`

# %%
report.plot_co2_seasonality_extend(diagnostics_root_path, suffixes, labels, colors)

# %%
report.co2_seasonality_extend_regression_table(diagnostics_root_path, suffixes, labels)

# %% [markdown]
# ## 8. Final gridded output diff
#
# **Derived in:** `40yy_write-input4mips` (the final gridding/ESGF-ready
# output step, downstream of everything else in this notebook).
#
# The bottom line: `LINEAR_SEASONAL_LAT_STD_WEIGHT_FIT` minus the no-satellite baseline, in the actual
# final input4MIPs product, both restricted to the 2003+ observational
# network. Needs the `40yy` step to have been run for this fit under
# `dev-test-run-sat-period` - if it hasn't, this section prints a note and shows nothing
# rather than erroring.

# %%
report.plot_gridded_diff_for_fit(output_bundles_root_path, run_id, gas, satellite_fit)

# %%

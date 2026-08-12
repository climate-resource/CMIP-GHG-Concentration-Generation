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
# # CH4 satellite diagnostics: NONLINEAR_LAT_FIT
#
# Compares `ch4` pipeline diagnostics with satellite data (`NONLINEAR_LAT_FIT` fit)
# against the no-satellite baseline, using whatever the `12yy`/`11yy`
# notebooks saved under `data/diagnostics/ch4/`. If you haven't run the
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
gas: str = "ch4"
satellite_fit: str = "NONLINEAR_LAT_FIT"
diagnostics_root: str = "../../../output-bundles/{run_id}/data/diagnostics"
output_bundles_root: str = "../../../output-bundles"

# %%
diagnostics_root_path = Path(diagnostics_root.format(run_id=run_id))
output_bundles_root_path = Path(output_bundles_root)
satellite_suffix = f"SAT_{satellite_fit}"
suffixes = ("nosat", satellite_suffix)
labels = {"nosat": "No satellite data", satellite_suffix: f"Satellite data ({satellite_fit})"}
colors = {"nosat": "tab:blue", satellite_suffix: "tab:orange"}
diagnostics_root_path

# %% [markdown]
# ## 1. Input coverage
#
# **Derived in:** `1101_ch4_interpolate-observational-network`
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
# **Derived in:** `1102_ch4_global-mean-latitudinal-gradient-seasonality`
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
# ## 3. Seasonality (observational-network period)
#
# **Derived in:** `1102_ch4_global-mean-latitudinal-gradient-seasonality`

# %%
report.plot_seasonality(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 4. Global-, annual-mean (observational-network period)
#
# **Derived in:** `1102_ch4_global-mean-latitudinal-gradient-seasonality`
#
# No EOF to compare here - it's a scalar-per-year timeseries.

# %%
report.plot_global_mean_obs_network(diagnostics_root_path, gas, suffixes, labels, colors)

# %% [markdown]
# ## 5. Latitudinal gradient extension (all years)
#
# **Derived in:** `1103_ch4_extend-pcs`
#
# The PCs above, extended back to year 1 using a regression against PRIMAP
# fossil emissions (plus, for CH4, a joint ice-core optimisation).

# %%
report.plot_lat_gradient_extend_pcs(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.regression_table(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 6. Global-, annual-mean extension (all years) and correctness checks
#
# **Derived in:** `1104_ch4_extend-global-annual-mean`
#
# Where the ice-core-derived history gets stitched onto the
# observational-network period, and where the pipeline's own correctness
# checks live. These normally only ever raise if they fail; here we print
# the actual numbers so a *passing* check that got noticeably worse is still
# visible.

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

# %%
report.plot_global_mean_allyears(diagnostics_root_path, gas, suffixes, labels, colors)

# %%
report.checks_table(diagnostics_root_path, gas, suffixes, labels)

# %%
report.seam_table(diagnostics_root_path, gas, suffixes, labels)

# %% [markdown]
# ## 7. Final gridded output diff
#
# **Derived in:** `40yy_write-input4mips` (the final gridding/ESGF-ready
# output step, downstream of everything else in this notebook).
#
# The bottom line: `NONLINEAR_LAT_FIT` minus the no-satellite baseline, in the actual
# final input4MIPs product, not just the internal pieces above. Needs the
# `40yy` step to have been run for this fit - if it hasn't, this section
# prints a note and shows nothing rather than erroring. For a view across
# *all* fits at once, see `check_ch4_output_all_fits.py` in this folder.

# %%
report.plot_gridded_diff_for_fit(output_bundles_root_path, run_id, gas, satellite_fit)

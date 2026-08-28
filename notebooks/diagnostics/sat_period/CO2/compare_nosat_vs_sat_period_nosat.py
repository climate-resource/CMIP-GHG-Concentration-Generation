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
# # CO2: original nosat vs. sat-period-restricted nosat
#
# A simple, hand-maintained notebook (not auto-generated, edit freely).
#
# Both runs compared here have satellite data switched **off** - the only
# difference is which years of the ground-based observational network were
# used: the original run uses the full record, the "sat-period" run drops
# everything before 2003 (`year_drop_observational_data_before_and_including
# = 2002`), matching the years the satellite product covers, so it isolates
# the effect of restricting to a shorter, more recent observational record -
# independent of anything about satellite data itself.
#
# This crosses two different `run_id`s (`dev-test-run` vs
# `dev-test-run-sat-period`), so it doesn't fit `local.diagnostics_reporting`'s
# usual one-root-two-suffixes functions - this reuses its individual loader
# functions directly against two different roots instead.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt

import local.diagnostics_reporting as report

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
gas: str = "co2"
original_run_id: str = "dev-test-run"
sat_period_run_id: str = "dev-test-run-sat-period"
output_bundles_root: str = "../../../../output-bundles"

# %%
output_bundles_root_path = Path(output_bundles_root)
original_root = output_bundles_root_path / original_run_id / "data" / "diagnostics"
sat_period_root = output_bundles_root_path / sat_period_run_id / "data" / "diagnostics"
labels = {"original": "Original nosat (full record)", "sat_period": "Sat-period nosat (2003+ only)"}
colors = {"original": "tab:blue", "sat_period": "tab:orange"}
original_root, sat_period_root

# %% [markdown]
# ## Load data
#
# Observational-network-period diagnostics (from `1202`), same `nosat`
# suffix in both roots.

# %%
obs_network = {
    "original": report.load_nc_diagnostics(original_root, gas, "obs-network", "nosat"),
    "sat_period": report.load_nc_diagnostics(sat_period_root, gas, "obs-network", "nosat"),
}
{k: (v.sizes if v is not None else None) for k, v in obs_network.items()}

# %% [markdown]
# ## Global-, annual-mean (observational-network period)
#
# The sat-period run only has 2003 onwards by construction - shown next to
# the original's full record for context, not because they're expected to
# match outside the overlap.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for key, ds in obs_network.items():
    if ds is None:
        continue
    ds["global_annual_mean_obs_network"].plot(ax=ax, color=colors[key], label=labels[key], marker="o")
ax.set_title(f"{gas.upper()} - global-, annual-mean (observational network)")
ax.set_ylabel(f"{gas.upper()} [ppm]")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Latitudinal gradient EOFs
#
# Does restricting the observational network to 2003+ change the shape of
# the leading spatial patterns, independent of satellite data?

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for key, ds in obs_network.items():
    if ds is None:
        continue
    for eof in range(2):
        axes[eof].plot(
            ds["lat_gradient_eofs_full"].sel(lat_gradient_eof=eof),
            ds["lat"],
            color=colors[key],
            label=labels[key] if eof == 0 else None,
        )
        axes[eof].set_title(f"EOF {eof}")
        axes[eof].set_xlabel(f"{gas.upper()} anomaly")
        axes[eof].set_ylabel("Latitude")
axes[0].legend(fontsize=8)
fig.suptitle(f"{gas.upper()} - latitudinal gradient EOFs (observational network)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Latitudinal gradient PCs, over the overlapping years (2003+)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
for key, ds in obs_network.items():
    if ds is None:
        continue
    for eof in range(2):
        ds["lat_gradient_pcs_full"].sel(lat_gradient_eof=eof, year=slice(2003, None)).plot(
            ax=axes[eof], color=colors[key], label=labels[key]
        )
        axes[eof].set_title(f"PC{eof}")
axes[0].legend(fontsize=8)
fig.suptitle(f"{gas.upper()} - latitudinal gradient PCs, 2003 onwards (observational network)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Global-, annual-mean, all years (extended back to year 1)
#
# The sat-period run's observational-network EOFs/PCs only span 2003-2023,
# but the regression-based backward extension (`1204`, against PRIMAP fossil
# emissions and ice cores) still runs over the full record - this shows how
# much anchoring that extension on a much shorter, more recent window
# changes the reconstructed history.

# %%
allyears = {
    "original": report.load_nc_diagnostics(original_root, gas, "global-mean-extend", "nosat"),
    "sat_period": report.load_nc_diagnostics(sat_period_root, gas, "global-mean-extend", "nosat"),
}

fig, ax = plt.subplots(figsize=(10, 5))
for key, ds in allyears.items():
    if ds is None:
        continue
    ds["global_annual_mean_allyears"].plot(ax=ax, color=colors[key], label=labels[key])
ax.axvline(
    2003, color="grey", linestyle="--", linewidth=1, alpha=0.8, label="sat-period run's obs-network starts"
)
ax.set_title(f"{gas.upper()} - global-, annual-mean, all years")
ax.set_ylabel(f"{gas.upper()} [ppm]")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final gridded output diff
#
# **Derived in:** `40yy_write-input4mips`. The bottom line in the actual
# final input4MIPs product: sat-period nosat minus original nosat. Needs the
# `40yy` step to have been run for both runs.

# %%
original_esgf_dir = report.get_esgf_ready_gas_dir(output_bundles_root_path, original_run_id, gas)
sat_period_esgf_dir = report.get_esgf_ready_gas_dir(output_bundles_root_path, sat_period_run_id, gas)

original_baseline_chunks, _ = report.discover_gridded_files(original_esgf_dir, gas)
sat_period_baseline_chunks, _ = report.discover_gridded_files(sat_period_esgf_dir, gas)

if not original_baseline_chunks or not sat_period_baseline_chunks:
    print("Missing gridded (40yy_write-input4mips) output for one or both runs - nothing to diff.")
else:
    original_da = report.load_concatenated_gridded(original_baseline_chunks, gas)
    sat_period_da = report.load_concatenated_gridded(sat_period_baseline_chunks, gas)
    diff = report.gridded_diff_from_baseline(original_da, sat_period_da)

    fig, ax = plt.subplots(figsize=(10, 3))
    diff.plot(ax=ax, x="time", y="lat", cmap="RdBu_r")
    ax.set_title(f"{gas.upper()} - sat-period nosat minus original nosat, final gridded output")
    plt.tight_layout()
    plt.show()

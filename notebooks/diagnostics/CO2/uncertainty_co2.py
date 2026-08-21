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
# # CO2 - measurement uncertainty by data source
#
# A simple, hand-maintained notebook (not auto-generated, edit freely) that
# compares the *reported* measurement uncertainty of each raw data source
# that feeds into the CO2 pipeline, before any of it gets processed.
#
# This is independent of everything downstream: `local.binning.calculate_bin_averages`
# currently drops every uncertainty column and does a plain, unweighted
# mean, so none of what's plotted here is actually used by the pipeline
# today - it's here to inform whether/how it should be.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# %% [markdown]
# ## Parameters

# %% tags=["parameters"]
run_id: str = "dev-test-run"
output_bundles_root: str = "../../../output-bundles"
# xco2_stderr is identical across every fit - it's a property of the raw
# satellite retrieval, not something the bias-correction fit touches - so
# any available fit's file works here.
satellite_fit_file: str = "LINEAR_FIT"

# %%
data_root = Path(output_bundles_root) / run_id / "data"
data_root

# %% [markdown]
# ## Load data
#
# Each source's raw/interim file, aggregated to one global monthly mean per
# source (unweighted mean across pixels/stations) so trends are simple to
# read and compare side by side. Rows with the dataset's missing-value
# sentinel (`-999.999`) are dropped before averaging.

# %% [markdown]
# ### Satellite (`xco2_stderr`)
#
# `xco2`/`xco2_stderr` are a fraction (mole fraction), so both are scaled by
# `1e6` to ppm to match the ground-based sources below.

# %%
sat_path = (
    data_root
    / "raw"
    / "scaled_sat"
    / f"200301_202312-C3S-L3_XCO2-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.6_{satellite_fit_file}.nc"
)
sat_ds = xr.open_dataset(sat_path)

satellite_monthly = ((sat_ds[["xco2", "xco2_stderr"]] * 1e6).mean(("lat", "lon")).to_dataframe()).rename(
    columns={"xco2": "value", "xco2_stderr": "uncertainty"}
)
satellite_monthly.index.name = "time"
satellite_monthly

# %% [markdown]
# ### NOAA in-situ (`value_std_dev`)

# %%
noaa_insitu = pd.read_csv(data_root / "interim" / "noaa" / "monthly_co2_in-situ_raw-consolidated.csv")
noaa_insitu = noaa_insitu.replace(-999.999, np.nan)
noaa_insitu["time"] = pd.to_datetime({"year": noaa_insitu["year"], "month": noaa_insitu["month"], "day": 1})

noaa_insitu_monthly = (
    noaa_insitu.groupby("time")[["value", "value_std_dev"]]
    .mean()
    .rename(columns={"value_std_dev": "uncertainty"})
)
noaa_insitu_monthly

# %% [markdown]
# ### NOAA surface-flask (`value_unc`)

# %%
noaa_flask = pd.read_csv(data_root / "interim" / "noaa" / "events_co2_surface-flask_raw-consolidated.csv")
noaa_flask = noaa_flask.replace(-999.999, np.nan)
noaa_flask["time"] = pd.to_datetime(noaa_flask["datetime"]).dt.to_period("M").dt.to_timestamp()

noaa_flask_monthly = (
    noaa_flask.groupby("time")[["value", "value_unc"]].mean().rename(columns={"value_unc": "uncertainty"})
)
noaa_flask_monthly

# %% [markdown]
# ## Metadata: what each uncertainty column actually means
#
# These are not all the same *kind* of quantity. `xco2_stderr` and
# `value_std_dev` both describe the spread/retrieval uncertainty of an
# average over some period - they're broadly comparable to each other.
# `value_unc` is different: it's the lab/instrument analytical precision on
# a single flask sample, not a spread over repeated measurements - it's
# typically much smaller, and comparing it directly against the other two
# (e.g. for inverse-variance weighting) isn't apples-to-apples.

# %%
metadata = pd.DataFrame(
    [
        {
            "source": "Satellite",
            "column": "xco2_stderr",
            "long_name": None,
            "description": (
                "Standard error of the average including single sounding noise "
                "and potential seasonal and regional biases."
            ),
            "units": "ppm (converted from a dimensionless mole fraction)",
        },
        {
            "source": "NOAA in-situ",
            "column": "value_std_dev",
            "long_name": "standard_deviation_in_reported_value",
            "description": "Standard deviation of the reported mean value when nvalue is greater than 1.",
            "units": "micromol mol-1 (ppm)",
        },
        {
            "source": "NOAA surface-flask",
            "column": "value_unc",
            "long_name": "estimated_uncertainty_in_reported_value",
            "description": (
                "Estimated uncertainty of the reported value - lab/instrument analytical "
                "precision on a single flask sample, not a spread over repeated measurements."
            ),
            "units": "micromol mol-1 (ppm)",
        },
    ]
).set_index(["source", "column"])
metadata

# %% [markdown]
# ## Value trends with their own uncertainty
#
# One panel per source: the value trend, with its uncertainty shown as a
# shaded band around it.

# %%
sources = {
    "Satellite": satellite_monthly,
    "NOAA in-situ": noaa_insitu_monthly,
    "NOAA surface-flask": noaa_flask_monthly,
}

fig, axes = plt.subplots(len(sources), 1, figsize=(10, 3 * len(sources)), sharex=True)
for ax, (name, df) in zip(axes, sources.items()):
    ax.plot(df.index, df["value"], color="tab:blue")
    ax.fill_between(
        df.index,
        df["value"] - df["uncertainty"],
        df["value"] + df["uncertainty"],
        color="tab:blue",
        alpha=0.3,
    )
    ax.set_title(name)
    ax.set_ylabel("CO2 [ppm]")
axes[-1].set_xlabel("Time")
fig.suptitle("CO2 - value trend with uncertainty band, by source")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Uncertainty trends, all sources together

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for name, df in sources.items():
    ax.plot(df.index, df["uncertainty"], label=name)
ax.set_xlabel("Time")
ax.set_ylabel("Uncertainty [ppm]")
ax.set_title("CO2 - uncertainty trends by source")
ax.legend()
plt.tight_layout()
plt.show()

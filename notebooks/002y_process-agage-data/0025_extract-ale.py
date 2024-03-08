# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Process ALE
#
# Process data from the ALE part of the AGAGE network. We extract the monthly data with lat-lon information.

# %% [markdown]
# ## Imports

# %%
from io import StringIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file
from local.regexp_helpers import re_search_and_retrieve_group

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_ale_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "monthly"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

config_retrieve = get_config_for_step_id(
    config=config, step="retrieve", step_config_id="only"
)


# %% [markdown]
# ## Action

# %% [markdown]
# ### Find relevant files


# %%
def is_relevant_file(f: Path) -> bool:
    """
    Check if a data file is relevant for this notebook
    """
    return f.name.endswith(".mon")


# %%
relevant_files = [f for f in list(config_step.raw_dir.glob("*")) if is_relevant_file(f)]
relevant_files


# %% [markdown]
# ### Load relevant files


# %%
def read_ale_file(f: Path, skiprows: int = 3, sep: str = r"\s+") -> pd.DataFrame:
    """
    Read a data file from the ALE experiment
    """
    with open(f) as fh:
        file_content = fh.read()

    site_code = f.name.split("-ale")[0]
    lat_str = re_search_and_retrieve_group(
        r"Lat.: (?P<latitude>-?\d*(\.\d*)?[SN])", file_content, "latitude"
    )
    if lat_str.endswith("S"):
        lat = -float(lat_str[:-1])
    elif lat_str.endswith("N"):
        lat = float(lat_str[:-1])
    else:
        raise NotImplementedError(lat_str)

    lon_str = re_search_and_retrieve_group(
        r"Lon.: (?P<longitude>-?\d*(\.\d*)?[EW])", file_content, "longitude"
    )
    if lon_str.endswith("W"):
        lon = -float(lon_str[:-1])
    elif lon_str.endswith("E"):
        lon = float(lon_str[:-1])
    else:
        raise NotImplementedError(lon_str)

    res = pd.read_csv(StringIO(file_content), skiprows=skiprows, sep=sep)

    gas_units = {}
    for gas, unit in zip(res.iloc[1, :], res.iloc[0, :]):
        if ":" not in unit and "---" not in unit:
            gas_units[gas] = unit

    res.columns = res.iloc[1, :]  # type: ignore
    res = res.iloc[2:, :]
    res = res.rename({"MM": "month", "YYYY": "year"}, axis="columns")
    res = res.set_index(["month", "year"]).sort_index()
    res = res[gas_units.keys()]
    res = res.melt(var_name="gas", ignore_index=False).reset_index()
    res["year"] = res["year"].astype(int)
    res["month"] = res["month"].astype(int)
    res["value"] = res["value"].astype(float)
    res["unit"] = res["gas"].map(gas_units)
    res["latitude"] = float(lat)
    res["longitude"] = float(lon)
    res["source"] = "ALE"
    res["site_code"] = site_code
    # Not sure why it is like this, but ok
    res = res[res["value"] > 0]

    return res


# %%
read_info = [read_ale_file(f) for f in tqdman.tqdm(relevant_files)]
df_monthly = pd.concat([v for v in read_info], axis=0)
df_monthly

# %% [markdown]
# ### Plot

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir
    / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
for gas, gas_df in tqdman.tqdm(df_monthly.groupby("gas"), desc="gas"):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    countries.plot(color="lightgray", ax=axes[0])
    colours = (c for c in ["tab:blue", "tab:green", "tab:red", "tab:pink", "tab:brown"])
    markers = (m for m in ["o", "x", ".", ",", "v"])
    for station, station_df in tqdman.tqdm(
        gas_df.groupby("site_code"), desc="Observing site"
    ):
        colour = next(colours)
        marker = next(markers)

        station_df[["longitude", "latitude"]].drop_duplicates().plot(
            x="longitude",
            y="latitude",
            kind="scatter",
            ax=axes[0],
            alpha=0.5,
            label=station,
            color=colour,
            zorder=3,
            s=100,
            marker=marker,
        )

        pdf = station_df.copy()
        pdf["year-month"] = pdf["year"] + pdf["month"] / 12
        pdf.plot(
            x="year-month",
            y="value",
            kind="scatter",
            ax=axes[1],
            label=station,
            color=colour,
            marker=marker,
            alpha=0.4,
        )

    axes[0].set_xlim([-180, 180])
    axes[0].set_ylim([-90, 90])

    axes[1].set_xticks(
        range(station_df["year"].min(), station_df["year"].max() + 2), minor=True
    )
    axes[1].legend()

    plt.suptitle(str(gas))
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Save

# %%
config_step.processed_monthly_data_with_loc_file.parent.mkdir(
    exist_ok=True, parents=True
)
df_monthly.to_csv(config_step.processed_monthly_data_with_loc_file, index=False)
df_monthly

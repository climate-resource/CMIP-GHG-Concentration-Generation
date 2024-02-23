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
# # Process AGAGE
#
# Process data from the AGAGE network. We extract the monthly data with lat-lon information.

# %% [markdown]
# ## Imports

# %%
import re
from io import StringIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_agage_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "ccl4_gc-md_monthly"  # config ID to select for this branch

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
if config_step.time_frequency == "monthly":
    suffix = "_mon.txt"
else:
    raise NotImplementedError(suffix)


# %%
def is_relevant_file(f):
    """
    Check if a data file is relevant for this notebook
    """
    if not (f.name.endswith(suffix) and config_step.gas in f.name):
        return False

    if config_step.instrument == "gc-ms-medusa" and "GCMS-Medusa" not in f.name:
        return False
    elif config_step.instrument == "gc-ms":
        if "GCMS-Medusa" in f.name:
            return False

        if "GCMS-" not in f.name:
            return False

    elif config_step.instrument == "gc-md" and "-GCMD_" not in f.name:
        return False

    return True


# %%
relevant_files = [f for f in list(config_step.raw_dir.glob("*")) if is_relevant_file(f)]
relevant_files


# %% [markdown]
# ### Load relevant files


# %%
def read_agage_file(f: Path, skiprows: int = 32, sep=r"\s+") -> pd.DataFrame:
    """
    Read a data file from the AGAGE experiment
    """
    with open(f) as fh:
        file_content = fh.read()

    site_code = f.name.split("_")[1]
    gas = re.search(r"species: (?P<species>\S*)", file_content).group("species")
    lat = re.search(r"inlet_latitude: (?P<latitude>-?\d*\.\d*)", file_content).group(
        "latitude"
    )
    lat = re.search(r"inlet_latitude: (?P<latitude>-?\d*\.\d*)", file_content).group(
        "latitude"
    )
    lon = re.search(r"inlet_longitude: (?P<longitude>-?\d*\.\d*)", file_content).group(
        "longitude"
    )
    unit = re.search(r"units: (?P<unit>\S*)", file_content).group("unit")
    contact_points = re.search(
        r"CONTACT POINT: (?P<contact_points>.*)", file_content
    ).group("contact_points")
    contacts = [v.strip() for v in contact_points.split(";")]

    res = pd.read_csv(StringIO(file_content), skiprows=skiprows, sep=sep)
    res["gas"] = gas
    res["site_code"] = site_code
    res["instrument"] = config_step.instrument
    res["latitude"] = float(lat)
    res["longitude"] = float(lon)
    res["unit"] = unit
    res["source"] = "AGAGE"
    res = res.rename({"mean": "value"}, axis="columns")

    return {"df": res, "contacts": contacts}


# %%
read_info = [read_agage_file(f) for f in tqdman.tqdm(relevant_files)]
contacts = set([c for v in read_info for c in v["contacts"]])
print(f"{contacts=}")
df_monthly = pd.concat([v["df"] for v in read_info], axis=0)
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
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
colours = (c for c in ["tab:blue", "tab:green", "tab:red", "tab:pink", "tab:brown"])
markers = (m for m in ["o", "x", ".", ",", "v"])

countries.plot(color="lightgray", ax=axes[0])

for station, station_df in tqdman.tqdm(
    df_monthly.groupby("site_code"), desc="Observing site"
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

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Save

# %%
assert set(df_monthly["gas"]) == {config_step.gas}
config_step.processed_monthly_data_with_loc_file.parent.mkdir(
    exist_ok=True, parents=True
)
df_monthly.to_csv(config_step.processed_monthly_data_with_loc_file, index=False)
df_monthly

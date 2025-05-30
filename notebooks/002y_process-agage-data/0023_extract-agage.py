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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Process AGAGE
#
# Process data from the AGAGE network. We extract the monthly data with lat-lon information.

# %% [markdown]
# ## Imports

# %%
from io import StringIO
from pathlib import Path

import geopandas as gpd
import matplotlib.axes
import matplotlib.pyplot as plt
import openscm_units
import pandas as pd
import pint
import pooch
import tqdm.autonotebook as tqdman
from pydoit_nb.config_handling import get_config_for_step_id

import local.agage_processing
import local.raw_data_processing
from local.config import load_config_from_file
from local.config_creation.agage_handling import (
    AGAGE_GAS_MAPPING,
    AGAGE_GAS_MAPPING_REVERSED,
)
from local.regexp_helpers import re_search_and_retrieve_group

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_agage_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "n2o_gc-md_monthly"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

config_retrieve = get_config_for_step_id(config=config, step="retrieve_misc_data", step_config_id="only")

# %% [markdown]
# ## Action

# %% [markdown]
# ### Find relevant files

# %%
if config_step.time_frequency == "monthly":
    suffix = "_mon.txt"
else:
    raise NotImplementedError(config_step.time_frequency)


# %%
def is_relevant_file(f: Path) -> bool:
    """
    Check if a data file is relevant for this notebook
    """
    try:
        gas_to_find = AGAGE_GAS_MAPPING[config_step.gas]

    except KeyError:
        gas_to_find = config_step.gas

    if not (f.name.endswith(suffix) and f"_{gas_to_find}_" in f.name):
        return False

    if config_step.instrument == "gc-ms-medusa" and "GCMS-Medusa" not in f.name:
        return False

    elif config_step.instrument == "gc-ms":
        if "GCMS-Medusa" in f.name:
            return False

        if "GCMS-" not in f.name:
            return False

    # Honestly, this SF6 exception, what is that?
    elif config_step.instrument == "gc-md" and not (
        "-GCMD_" in f.name or ("sf6" in f.name and "-GCECD_CGO_sf6" in f.name)
    ):
        return False

    return True


# %%
# [f for f in list(config_step.raw_dir.glob("*")) if "hcfc" in str(f)]

# %%
relevant_files = [f for f in list(config_step.raw_dir.glob("*")) if is_relevant_file(f)]
if not relevant_files:
    raise AssertionError()
relevant_files


# %% [markdown]
# ### Load relevant files


# %%
def read_agage_file(f: Path, skiprows: int = 34, sep: str = r"\s+") -> tuple[tuple[str, ...], pd.DataFrame]:
    """
    Read a data file from the AGAGE experiment
    """
    with open(f) as fh:
        file_content = fh.read()

    site_code = f.name.split("_")[1]

    try:
        gas = re_search_and_retrieve_group(r"species: (?P<species>\S*)", file_content, "species")
    except ValueError:
        print(f"File is missing species information: {f}")
        gas = config_step.gas

    lat = re_search_and_retrieve_group(r"inlet_latitude: (?P<latitude>-?\d*\.\d*)", file_content, "latitude")
    lon = re_search_and_retrieve_group(
        r"inlet_longitude: (?P<longitude>-?\d*\.\d*)", file_content, "longitude"
    )
    try:
        unit = re_search_and_retrieve_group(r"units: (?P<unit>\S*)", file_content, "unit")
    except ValueError:
        print(f"File is missing units information: {f}")
        if any(
            v in f.name
            for v in (
                "cf4",
                "cfc-13",
                "cfc-113",
                "cfc-114",
                "cfc-115",
                "ch2cl2",
                "ch3br",
                "ch3ccl3",
                "ch3cl",
                "chcl3",
                "h-1211",
                "h-1301",
                "h-2402",
                "hcfc-22",
                "hcfc-124",
                "hcfc-132b",
                "hcfc-133a",
                "hcfc-141b",
                "hcfc-142b",
                "hfc-23",
                "hfc-32",
                "hfc-125",
                "hfc-134a",
                "hfc-143a",
                "hfc-152a",
                "hfc-227ea",
                "hfc-236fa",
                "hfc-245fa",
                "hfc-365mfc",
                "hfc-4310mee",
                "nf3",
                "pce",
                "pfc-116",
                "pfc-218",
                "pfc-318",
                "sf6",
                "so2f2",
                "ccl4",
                "cfc-11",
                "cfc-12",
            )
        ):
            # Guess, file is missing metadata
            unit = "ppt"
        else:
            raise

    contact_points = re_search_and_retrieve_group(
        r"CONTACT POINT: (?P<contact_points>.*)", file_content, "contact_points"
    )
    contacts = tuple(v.strip() for v in contact_points.split(";"))

    header_row = re_search_and_retrieve_group(r"(?P<header_row>#    time.*)", file_content, "header_row")
    columns = [v.strip() for v in header_row.split("  ") if v][1:]
    res = pd.read_csv(StringIO(file_content), skiprows=skiprows, sep=sep, header=None)
    res.columns = columns  # type: ignore
    res["gas"] = gas
    res["site_code"] = site_code
    res["instrument"] = config_step.instrument
    res["latitude"] = float(lat)
    res["longitude"] = float(lon)
    res["unit"] = unit
    res["source"] = "AGAGE"
    res = res.rename({"mean": "value"}, axis="columns")

    return contacts, res


# %%
read_info = [read_agage_file(f) for f in tqdman.tqdm(relevant_files)]
contacts = set([c for v in read_info for c in v[0]])
print(f"{contacts=}")
df_monthly = pd.concat([v[1] for v in read_info], axis=0)
df_monthly

# %%
for gas_file_option in df_monthly["gas"].unique():
    if gas_file_option == config_step.gas:
        continue

    mapped_name = AGAGE_GAS_MAPPING_REVERSED[gas_file_option]
    if mapped_name == config_step.gas:
        print(f"Assuming {gas_file_option} is the same as {mapped_name}")
        continue

    raise NotImplementedError(gas_file_option)

# Now we can re-name with confidence
df_monthly["gas"] = config_step.gas
df_monthly

# %% [markdown]
# ### Plot

# %%
countries = gpd.read_file(
    config_retrieve.natural_earth.raw_dir / config_retrieve.natural_earth.countries_shape_file_name
)
# countries.columns.tolist()

# %%
fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
if isinstance(axes, matplotlib.axes.Axes):
    raise TypeError(type(axes))

colours = (
    c
    for c in [
        "tab:blue",
        "tab:green",
        "tab:red",
        "tab:pink",
        "tab:brown",
        "tab:cyan",
        "lime",
        "purple",
        "magenta",
        "blue",
        "darkgreen",
        "firered",
    ]
)
markers = (m for m in ["o", "x", ".", ",", "v", "+", "1", "2", "3", "4", "p", "P"])

countries.plot(color="lightgray", ax=axes[0])

for station, station_df in tqdman.tqdm(df_monthly.groupby("site_code"), desc="Observing site"):
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

axes[1].set_xticks(range(station_df["year"].min(), station_df["year"].max() + 2), minor=True)
axes[1].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Prepare and check output

# %%
out = df_monthly.copy()
out["network"] = "AGAGE"
out["station"] = out["site_code"].str.lower()
out["measurement_method"] = out["instrument"]

out

# %%
local.raw_data_processing.check_processed_data_columns_for_spatial_binning(out)

# %% [markdown]
# ### Save

# %%
assert set(out["gas"]) == {config_step.gas}
config_step.processed_monthly_data_with_loc_file.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(config_step.processed_monthly_data_with_loc_file, index=False)
out

# %%
readme_path = pooch.retrieve(
    url=config_step.readme.url,
    known_hash=config_step.readme.known_hash,
    fname="AGAGE_README.txt",
    path=config_step.raw_dir,
    progressbar=True,
)

# %%
if isinstance(readme_path, list):
    raise TypeError(readme_path)

with open(readme_path, encoding="iso-8859-1") as fh:
    readme_raw = fh.read()

source_infos_gas = local.agage_processing.extract_agage_source_info(readme_raw, gas=config_step.gas)
source_infos_gas

# %%
source_info_short_names = []
for si in source_infos_gas:
    local.dependencies.save_source_info_to_db(
        db=config.dependency_db,
        source_info=si,
    )
    source_info_short_names.append(si.short_name)

source_info_short_names

# %%
local.dependencies.save_source_info_short_names(
    short_names=source_info_short_names, out_path=config_step.source_info_short_names_file
)

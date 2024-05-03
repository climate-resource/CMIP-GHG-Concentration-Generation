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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Scripps - process
#
# Process data from the Scripps CO2 program.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import io

import numpy as np
import openscm_units
import pandas as pd
import pint
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "retrieve_and_process_scripps_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% [markdown]
# ### Read and process merged ice core data

# %%
with open(
    config_step.raw_dir / config_step.merged_ice_core_data.url.split("/")[-1]
) as fh:
    raw_ice_core = fh.read()

assert "CO2 in ppm" in raw_ice_core


merged_ice_core = pd.read_csv(
    config_step.raw_dir / config_step.merged_ice_core_data.url.split("/")[-1],
    skiprows=47,
    header=0,
)
merged_ice_core.columns = [v.strip() for v in merged_ice_core.columns]  # type: ignore
merged_ice_core["unit"] = "ppm"
merged_ice_core = merged_ice_core.rename({"sample_date": "time"}, axis="columns")
merged_ice_core

# %%
merged_ice_core.set_index("time")["co2"].plot()

# %% [markdown]
# ### Save
#
# This can be saved as is because it will just be used for later comparisons (and we can tweak the format in future).

# %%
config_step.merged_ice_core_data_processed_data_file.parent.mkdir(
    exist_ok=True, parents=True
)
merged_ice_core.to_csv(
    config_step.merged_ice_core_data_processed_data_file, index=False
)
config_step.merged_ice_core_data_processed_data_file

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Read and process station data

# %%
LAST_INTERESTING_DATA_COLUMN: int = 5
"""
Last column which contains data we're interested in.

The README of the data files species that column 5 is the CO2 data,
after that we can ignore the info.
"""

EXPECTED_FORMAT_LINES: list[str] = [
    "The data file below contains 10 columns",
    "Columns 1-4 give the dates in several redundant",
    "Column 5 below gives monthly CO2 concentrations in micro-mol CO2",
]
"""
Lines in the data file which give information about the format.
"""

monthly_dfs_with_loc = []
for scripps_source in config_step.station_data:
    filepath = config_step.raw_dir / scripps_source.url_source.url.split("/")[-1]
    with open(filepath) as fh:
        raw = fh.read()
        contents = io.StringIO(raw)

    for expected_line in EXPECTED_FORMAT_LINES:
        assert expected_line in raw.replace("Mauna Loa CO2 ", "").replace(
            "monthly concentrations", "monthly CO2 concentrations"
        ), expected_line

    found_source = False
    found_unit = False
    next_line = contents.readline()
    while next_line.startswith('"'):
        loc = contents.tell()
        next_line = contents.readline()

        if next_line.startswith(
            '" Monthly average CO2 concentrations (ppm)'
        ) or next_line.startswith('" Atmospheric CO2 concentrations (ppm)'):
            found_unit = True
            unit = "ppm"

            if "derived from flask air samples" in next_line:
                source = "flask"
                found_source = True

            elif "derived from in situ air measurements" in next_line:
                source = "in_situ"
                found_source = True

            elif "derived from daily flask and in situ (continuous) data" in next_line:
                source = "flask-in_situ-blend"
                found_source = True

            else:
                raise NotImplementedError(next_line)

        continue

    if not found_unit:
        msg = f"No unit found for: {filepath}"
        raise ValueError(msg)

    if not found_source:
        msg = f"No source found for: {filepath}"
        raise ValueError(msg)

    contents.seek(loc)
    raw_df = pd.read_csv(contents)

    raw_df.columns = [v.strip() for v in raw_df.columns]  # type: ignore

    keep = (
        raw_df.iloc[:, :5][["Yr", "Mn", "CO2"]]
        .iloc[2:, :]
        .rename({"CO2": "value", "Yr": "year", "Mn": "month"}, axis="columns")
    )
    keep["unit"] = unit
    keep["source"] = source
    if scripps_source.lon.endswith("W"):
        keep["longitude"] = -float(scripps_source.lon.split(" ")[0])
    elif scripps_source.lon.endswith("E"):
        keep["longitude"] = float(scripps_source.lon.split(" ")[0])
    else:
        raise NotImplementedError(scripps_source.lon)

    if scripps_source.lat.endswith("S"):
        keep["latitude"] = -float(scripps_source.lat.split(" ")[0])
    elif scripps_source.lat.endswith("N"):
        keep["latitude"] = float(scripps_source.lat.split(" ")[0])
    else:
        raise NotImplementedError(scripps_source.lat)

    keep["station_code"] = scripps_source.station_code
    keep["gas"] = "co2"
    keep[["year", "month"]] = keep[["year", "month"]].astype(int)
    keep["value"] = keep["value"].astype(float)
    keep["value"] = keep["value"].replace(-99.99, np.NaN)

    monthly_dfs_with_loc.append(keep)

monthly_df_with_loc = pd.concat(monthly_dfs_with_loc)
monthly_df_with_loc

# %% [markdown]
# ### Save out result

# %%
config_step.processed_data_with_loc_file.parent.mkdir(exist_ok=True, parents=True)
monthly_df_with_loc.to_csv(config_step.processed_data_with_loc_file, index=False)
monthly_df_with_loc

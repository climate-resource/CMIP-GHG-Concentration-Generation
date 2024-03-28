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
# # Law dome ice core - process
#
# Process data from the Law Dome record.

# %% [markdown]
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.lib.pretty import pretty
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define branch this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "retrieve_and_process_law_dome_data"

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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Define Law Dome's location

# %% editable=true slideshow={"slide_type": ""}
LAW_DOME_LATITUDE = -(66 + 44 / 60)
LAW_DOME_LONGITUDE = 112 + 50 / 60

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Load and clean data

# %% editable=true slideshow={"slide_type": ""}
file_name_dict = {k.name: k for k in config_step.files_md5_sum}
print(pretty(file_name_dict))  # type: ignore

# %% editable=true slideshow={"slide_type": ""}
processed_dfs = []
for sheet, gas, unit in [
    ("CO2byAge", "CO2", "ppm"),
    ("CH4byAge", "CH4", "ppb"),
    ("N2ObyAge", "N2O", "ppb"),
]:
    raw = pd.read_excel(file_name_dict["Law_Dome_GHG_2000years.xlsx"], sheet_name=sheet)
    col_map = {
        f"{gas} Age (year AD)": "time",
        # "CO2 Age (year AD)": "x",
        f"{gas} ({unit})": "value",
    }
    useable = raw[col_map.keys()].copy()
    useable.columns = useable.columns.map(col_map)
    useable["unit"] = unit
    useable["gas"] = gas.lower()
    # TODO: Check, should there be polynomial smoothing here?
    useable["year"] = useable["time"].apply(np.floor).astype(int)
    month = ((useable["time"] - useable["year"]) * 12).apply(np.ceil).astype(int)
    month[month == 0] = 1
    useable["month"] = month
    useable["latitude"] = LAW_DOME_LATITUDE
    useable["longitude"] = LAW_DOME_LONGITUDE
    # # TODO: work out convention for this
    # useable["source"] = "CSIRO-law-dome"

    processed_dfs.append(useable)
#     # TODO: be more careful with time conversions
#     processed_runs.append(BaseScmRun(useable))

processed = pd.concat(processed_dfs)

if config.ci:
    # Chop the data to speed things up
    processed = processed[processed["year"] >= 1750]  # noqa: PLR2004

processed

# %% editable=true slideshow={"slide_type": ""}
for gas_label, gdf in processed.groupby("gas"):
    ax = gdf.plot(x="time", y="value")
    unit_gas = gdf["unit"].unique()
    assert len(unit_gas) == 1
    ax.set_ylabel(f"{gas_label} ({unit_gas[0]})")

    plt.show()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Save out result

# %% editable=true slideshow={"slide_type": ""}
config_step.processed_data_with_loc_file.parent.mkdir(exist_ok=True, parents=True)
processed.to_csv(config_step.processed_data_with_loc_file, index=False)
config_step.processed_data_with_loc_file

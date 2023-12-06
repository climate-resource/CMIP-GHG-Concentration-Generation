# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Law dome

# %% [markdown]
# ## Imports

# %%
import pandas as pd
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "process"

# %% [markdown]
# ## Parameters

# %%
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %%
config_retrieve = get_config_for_step_id(
    config=config, step="retrieve", step_config_id="only"
)

# %% [markdown]
# ## Action

# %% [markdown]
# - read CO2, CH4 and N2O into scmdata
#     - will need lat and lon information in future, for now ignore that
# - save

# %%
file_name_dict = {k.name: k for k in config_retrieve.law_dome.files_md5_sum}
file_name_dict

# %%

# TODO: remove hard-coding of sheet name?
raw = pd.read_excel(
    file_name_dict["Law_Dome_GHG_2000years.xlsx"], sheet_name="CO2byAge"
)
col_map = {
    "CO2 Age (year AD)": "time",
    # "CO2 Age (year AD)": "x",
    "CO2 (ppm)": "value",
}
useable = raw[col_map.keys()].copy()
useable.columns = useable.columns.map(col_map)
useable["unit"] = "ppm"
useable["variable"] = "Atmospheric Concentrations|CO2"
# hack
# useable["time"] = useable["time"].apply(lambda x: dt.datetime(int(x), 1, 1))
useable

# %%
output_notebook()

# %%
law_dome = ColumnDataSource(useable)

figure_bokeh = figure(
    title="CO2 by age",
    x_axis_label="time",
    y_axis_label="value",
)

renderer_line = figure_bokeh.line(
    "time", "value", legend_label="Law Dome", source=law_dome
)

figure_bokeh.add_tools(
    HoverTool(
        tooltips="@time{%f} @value@unit",
        formatters={"@time": "printf"},
        renderers=[renderer_line],
        mode="mouse",
    )
)

figure_bokeh.legend.location = "top_left"
figure_bokeh.legend.click_policy = "hide"

show(figure_bokeh)

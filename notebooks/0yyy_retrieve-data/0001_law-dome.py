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
from doit.dependency import get_file_md5

from local.config import get_config_for_branch_id, load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
branch: str = "retrieve"

# %% [markdown]
# ## Parameters

# %%
config_file: str = "../../dev-config-absolute.yaml"  # config file
branch_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_branch = get_config_for_branch_id(
    config=config, branch=branch, branch_config_id=branch_config_id
)

# %% [markdown]
# ## Action

# %%
print(f"DOI for dataset: {config_branch.law_dome.doi}")

# %% [markdown]
# ### Check we're using the intended files

# %%
for fp, expected_md5 in config_branch.law_dome.files_md5_sum.items():
    actual_md5 = get_file_md5(fp)
    if not expected_md5 == actual_md5:
        error_msg = (
            f"Unexpected MD5 hash for {fp}. " f"{expected_md5=} " f"{actual_md5=} "
        )
        raise AssertionError(error_msg)

# %% [markdown]
# ### Read the file

# %%
import pandas as pd

# %%
file_name_dict = {k.name: k for k in config_branch.law_dome.files_md5_sum}
file_name_dict

# %%
config_branch.law_dome.files_md5_sum

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
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

output_notebook()

# %%
# Not sure what trickery is going to be needed to make the datetimes behave here
law_dome = ColumnDataSource(useable)

figure_bokeh = figure(
    title="CO2 by age",
    # x_axis_type='datetime',
    # x_axis_type='numerical',
    # TODO: remove hard-coding
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
        # tooltips="@time @value@unit",
        # formatters={"@time": "numeral"},
        renderers=[renderer_line],
        # mode="mouse"
        # mode="hline"
        mode="vline",
    )
)

figure_bokeh.legend.location = "top_left"
figure_bokeh.legend.click_policy = "hide"

show(figure_bokeh)

# %% [markdown]
# ## Sandbox

# %%
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.models import HoverTool
from bokeh.plotting import figure

output_notebook()

# %%
x = np.linspace(0, 2 * np.pi, 2000)
y = np.sin(x)

# %%
figure_bokeh = figure(title="Figure with HoverTool", toolbar_location=None)

renderer_line_1 = figure_bokeh.line(
    [1.1, 2.3, 3.3, 4.4, 5.1],
    [3.3, 4.5, 5.1, 6.6, 7.4],
    legend_label="A",
    line_width=2,
    line_color="red",
)
renderer_line_2 = figure_bokeh.line(
    [1.1, 2.3, 3.3, 4.4, 5.1], [1.3, 2.5, 3.1, 4.6, 5.4], legend_label="B", line_width=2
)

figure_bokeh.add_tools(
    HoverTool(
        tooltips="y: @y, x: @x",
        renderers=[renderer_line_1, renderer_line_2],
        # mode="mouse"
        #            mode="hline"
        mode="vline",
    )
)

figure_bokeh.legend.location = "top_left"
figure_bokeh.legend.click_policy = "hide"

show(figure_bokeh)

# %%

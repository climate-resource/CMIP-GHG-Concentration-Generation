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
    "CO2 Age (year AD)": "year",
    "CO2 (ppm)": "value",
}
useable = raw[col_map.keys()].copy()
useable.columns = useable.columns.map(col_map)
useable["unit"] = "ppm"
useable["variable"] = "Atmospheric Concentrations|CO2"
useable

# %%
law_dome = ColumnDataSource(useable)

figure_bokeh = figure(
    title="CO2 by age",
    # x_axis_type='datetime',
    # TODO: remove hard-coding
    x_axis_label="year",
    y_axis_label="value",
)

renderer_line = figure_bokeh.line(
    "year",
    "value",
    # color='#CE1141',
    # legend='Law Dome',
    source=law_dome
    # [1,2,3,4,5], [3,4,5,6,7], legend_label="A", line_width=2, line_color="red"
)

figure_bokeh.add_tools(
    HoverTool(
        tooltips="@year @value@unit",
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
from bokeh.io import output_notebook, push_notebook, show
from bokeh.models import HoverTool
from bokeh.plotting import figure

output_notebook()

# %%
x = np.linspace(0, 2 * np.pi, 2000)
y = np.sin(x)

# %%
figure_bokeh = figure(title="Figure with HoverTool")

renderer_line_1 = figure_bokeh.line(
    [1, 2, 3, 4, 5], [3, 4, 5, 6, 7], legend_label="A", line_width=2, line_color="red"
)
renderer_line_2 = figure_bokeh.line(
    [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], legend_label="B", line_width=2
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
print(HoverTool.__doc__)

# %%
figure_bokeh = figure(
    title="simple line example",
    height=300,
    width=600,
    y_range=(-5, 5),
    background_fill_color="#efefef",
)
renderer_bokeh = figure_bokeh.line(x, y, color="#8888cc", line_width=1.5, alpha=0.8)


# %%
def update(f, w=1, A=1, phi=0):
    if f == "sin":
        func = np.sin
    elif f == "cos":
        func = np.cos
    renderer_bokeh.data_source.data["y"] = A * func(w * x + phi)
    push_notebook()


# %%
show(figure_bokeh, notebook_handle=True)

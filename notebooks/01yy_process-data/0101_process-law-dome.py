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
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from scmdata.run import BaseScmRun, run_append

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "process"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
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
pprint(file_name_dict)

# %%
processed_runs = []
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
    useable["variable"] = f"Atmospheric Concentrations|{gas}"
    useable["region"] = "World"
    useable["scenario"] = "historical"
    useable["source"] = "CSIRO-law-dome"

    # TODO: be more careful with time conversions
    processed_runs.append(BaseScmRun(useable))

processed = run_append(processed_runs)
if config.ci:
    # Chop the data to speed things up
    processed = processed.filter(year=range(1850, 3000))

processed

# %%
config_step.law_dome.processed_file.parent.mkdir(exist_ok=True, parents=True)
processed.to_csv(config_step.law_dome.processed_file)
config_step.law_dome.processed_file

# %%
for vdf in processed.groupby("variable"):
    vdf.lineplot(style="variable")  # type: ignore
    plt.show()

# %%
output_notebook()

# %%
pdf: pd.DataFrame = processed.long_data(time_axis="seconds since 1970-01-01")  # type: ignore
# Hackity hack
seconds_per_year = 60 * 60 * 24 * 365.25
pdf["time"] = pdf["time"] / (seconds_per_year) + 1970
pdf

# %%
gas = "N2O"
law_dome = ColumnDataSource(
    pdf.loc[pdf["variable"].isin([f"Atmospheric Concentrations|{gas}"])]
)

figure_bokeh = figure(
    title=gas,
    x_axis_label="time",
    y_axis_label="value",
)

renderer_line = figure_bokeh.line(
    "time", "value", legend_label="Law Dome", source=law_dome
)

figure_bokeh.add_tools(
    HoverTool(
        tooltips="@time{%.1f} @value{%.1f}@unit",
        formatters={"@time": "printf", "@value": "printf"},
        renderers=[renderer_line],
        mode="mouse",
    )
)

figure_bokeh.legend.location = "top_left"
figure_bokeh.legend.click_policy = "hide"

show(figure_bokeh)

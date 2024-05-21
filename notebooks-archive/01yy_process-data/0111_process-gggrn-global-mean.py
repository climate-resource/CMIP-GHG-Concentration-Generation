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
# # Global greenhouse gas research network
#
# Process global-mean data from the Global Greenhouse Gas Research Network (GGGRN). This notebook just retrieves global-mean data. In future, we will pull in individual station data instead.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
from pydoit_nb.config_handling import get_config_for_step_id
from scmdata.run import BaseScmRun, run_append

from local.config import load_config_from_file

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

# %%
gas_units = {"CO2": "ppm", "CH4": "ppb", "N2O": "ppb"}

# %%
out_list = []
for url_source in config_retrieve.gggrn.urls_global_mean:
    filename = url_source.url.split("/")[-1]
    file = config_retrieve.gggrn.raw_dir / filename

    gas = filename.split("_")[0].upper()

    with open(file) as fh:
        raw_txt = fh.read()

    for i, line in enumerate(raw_txt.splitlines()):
        if line.strip().startswith("# year"):
            skiprows = i
            break
    else:
        raise ValueError("header not found")  # noqa: TRY003

    raw = pd.read_csv(
        config_retrieve.gggrn.raw_dir / filename,
        skiprows=skiprows,
        sep=r"\s+",
    )

    unit = gas_units[gas]
    assert f"abbreviated as {unit}" in raw_txt
    clean = raw.copy()
    cols = clean.columns[1:]
    clean = clean.dropna(axis="columns")
    clean.columns = cols

    col_map = {
        "decimal": "time",
        "average": "value",
    }
    translated = clean[col_map.keys()]
    translated.columns = translated.columns.map(col_map)
    translated = (
        translated.set_index("time")["value"]
        .to_frame(f"Atmospheric Concentrations|{gas}")
        .T
    )
    translated.index.name = "variable"
    translated["scenario"] = "historical"
    translated["region"] = "World"
    translated["unit"] = unit
    translated["source"] = "GGGRN_global-mean-estimates"

    translated_run = BaseScmRun(translated)

    out_list.append(translated_run)

out = run_append(out_list)

if config.ci:
    # Chop the data to speed things up
    out = out.filter(year=range(1850, 3000))

out

# %%
config_step.gggrn.processed_file_global_mean.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(config_step.gggrn.processed_file_global_mean)
config_step.gggrn.processed_file_global_mean

# %%
for vdf in out.groupby("variable"):
    vdf.lineplot(style="variable")  # type: ignore
    plt.show()

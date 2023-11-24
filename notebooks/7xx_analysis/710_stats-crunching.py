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
# # Stats crunching
#
# This notebook demonstrates the `generate_directory_checklist` functionality. The idea is that, sometimes you know that files will be produced in a directory, but you can't or it isn't practical to predict/put in the config their names, how many files etc. for whatever reason. We show a way to work around this here.

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %%
import pandas as pd

from local.config import get_config_for_step_id, load_config_from_file
from local.pydoit_nb.checklist import generate_directory_checklist

# %% [markdown]
# ## Define step this notebook belongs to

# %%
step: str = "analysis"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this step

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown]
# ## Action

# %% [markdown]
# ### Calculate mean of each variation in x direction

# %%
for covariance_variation in config.covariance:
    draws = pd.read_csv(covariance_variation.draw_file)
    mean = draws["x"].mean()
    # This is the bit we can't easily predict. There are other ways around
    # this but for illustration's sake we use this. We hope to find a better
    # example in future (if you have one in mind, PRs are welcome)
    assert len(config.analysis) == 1
    out_filename = (
        config.analysis[0].mean_dir
        / f"{covariance_variation.step_config_id}_x-mean_{mean:0.2f}.txt"
    )

    out_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(out_filename, "w") as fh:
        fh.write(f"x-mean: {mean:.10f}")

# %% [markdown]
# ## Generate the directory checklist file
#
# This captures the files that are created by this notebook in a way that allows us to see if they're up to date or not. The documentation could definitely be improved so if the concept isn't clear or you think there is a better way to explain it please make a PR.

# %%
generate_directory_checklist(config.analysis[0].mean_dir)

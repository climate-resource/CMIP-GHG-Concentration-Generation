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
# # Set seed

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
from local.config import get_config_for_step_id, load_config_from_file

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define step this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "preparation"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this step

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
# ### Write out seed

# %% editable=true slideshow={"slide_type": ""}
config_step.seed_file.parent.mkdir(exist_ok=True, parents=True)
with open(config_step.seed_file, "w") as fh:
    fh.write(str(config_step.seed))

# %%
config_step.seed_file

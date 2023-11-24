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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Draw samples

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Imports

# %% editable=true slideshow={"slide_type": ""}
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local.config import get_config_for_step_id, load_config_from_file

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Define step this notebook belongs to

# %% editable=true slideshow={"slide_type": ""}
step: str = "covariance"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "cov"  # config ID to select for this step

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% editable=true slideshow={"slide_type": ""}
config_preparation = get_config_for_step_id(
    config=config, step="preparation", step_config_id="only"
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% editable=true slideshow={"slide_type": ""}
N_DRAWS: int = 250

# %% editable=true slideshow={"slide_type": ""}
with open(config_preparation.seed_file) as fh:
    seed = int(fh.read())

seed

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Make draws

# %%
generator = np.random.Generator(np.random.PCG64(seed))

# %%
draws = pd.DataFrame(
    generator.multivariate_normal(
        mean=np.zeros_like(np.diag(config_step.covariance)),
        cov=config_step.covariance,
        size=N_DRAWS,
    ),
    columns=["x", "y"],
)

# %%
plt.scatter(draws["x"], draws["y"])

# %%
draws.to_csv(config_step.draw_file, index=False)
config_step.draw_file

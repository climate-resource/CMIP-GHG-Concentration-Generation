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
# # Draw samples

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local.config import load_config_from_file
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define step this notebook belongs to

# %%
step: str = "constraint"

# %% [markdown]
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

# %% editable=true slideshow={"slide_type": ""}
generator = np.random.Generator(np.random.PCG64(seed))

# %%
draws_list: list[list[float]] = []
while len(draws_list) < N_DRAWS:
    x = generator.uniform(high=2.0)
    y = generator.uniform()

    if y > x * config_step.constraint_gradient:
        draws_list.append([x, y])

draws = pd.DataFrame(draws_list, columns=["x", "y"])
draws

# %%
ax = plt.subplot(111)
ax.scatter(draws["x"], draws["y"], s=10, alpha=0.7)
ax.axline((0, 0), slope=config_step.constraint_gradient, color="k")

# %%
draws.to_csv(config_step.draw_file, index=False)
config_step.draw_file

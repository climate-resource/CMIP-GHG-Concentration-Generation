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

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from local.config import get_config_for_branch_id, load_config_from_file

# %%
branch: str = "covariance"

# %%
config_file: str = "../../dev-config-absolute.yaml"  # config file
branch_config_id: str = "cov"  # config ID to select for this branch

# %%
config = load_config_from_file(config_file)
config_branch = get_config_for_branch_id(
    config=config, branch=branch, branch_config_id=branch_config_id
)

# %%
config_preparation = get_config_for_branch_id(
    config=config, branch="preparation", branch_config_id="only"
)

# %%
N_DRAWS: int = 250

# %%
with open(config_preparation.seed_file) as fh:
    seed = int(fh.read())

seed

# %%
generator = np.random.Generator(np.random.PCG64(seed))

# %%
draws = pd.DataFrame(
    generator.multivariate_normal(
        mean=np.zeros_like(np.diag(config_branch.covariance)),
        cov=config_branch.covariance,
        size=N_DRAWS,
    ),
    columns=["x", "y"],
)

# %%
plt.scatter(draws["x"], draws["y"])

# %%
draws.to_csv(config_branch.draw_file, index=False)
config_branch.draw_file

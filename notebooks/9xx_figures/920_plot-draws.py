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
# # Plot

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from local.config import get_config_for_branch_id, load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
branch: str = "figures"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
branch_config_id: str = "only"  # config ID to select for this branch

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_branch = get_config_for_branch_id(
    config=config, branch=branch, branch_config_id=branch_config_id
)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Action

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Load data

# %% editable=true slideshow={"slide_type": ""}
pdf = pd.read_csv(config_branch.draw_comparison_table)
pdf

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ### Plot

# %% editable=true slideshow={"slide_type": ""}
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["text.usetex"] = False
plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["font.size"] = 12
# plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.xmargin"] = 0.05

# %% editable=true slideshow={"slide_type": ""}
ax = sns.jointplot(
    data=pdf,
    x="x",
    y="y",
    hue="label",
    alpha=0.5,
)

config_branch.draw_comparison_figure.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(config_branch.draw_comparison_figure, transparent=True, bbox_inches="tight")

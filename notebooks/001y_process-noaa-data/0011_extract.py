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
# # NOAA - extract
#
# Extract data from NOAA from the downloaded zip file.

# %% [markdown]
# ## Imports

# %%
from local.config import load_config_from_file
from local.noaa_processing import read_noaa_flask_zip, read_noaa_in_situ_zip
from local.pydoit_nb.config_handling import get_config_for_step_id

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_noaa_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "co2_in-situ"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown]
# ## Action

# %%
zip_files = [
    config_step.raw_dir / url_source.url.split("/")[-1]
    for url_source in config_step.download_urls
]
zip_files

# %%
import zipfile

with zipfile.ZipFile(zip_files[0]) as zip:
    print(zip.filelist)
    # event_files = [item for item in zip.filelist if event_file_identifier(item.filename)]
    # df_events = pd.concat(
    #     [read_data_incl_datetime(zip, event_file_item) for event_file_item in tqdman.tqdm(event_files)]
    # )

    # month_files = [item for item in zip.filelist if month_file_identifier(item.filename)]
    # df_months = pd.concat(
    #     [read_flask_monthly_data(zip, month_files_item) for month_files_item in tqdman.tqdm(month_files)]
    # )
    # df_months["unit"] = ASSUMED_MONTHLY_UNITS[gas]


# %%
assert len(zip_files) == 1, "Re-think how you're doing this"
zf = zip_files[0]

if config_step.source == "surface-flask":
    df_events, df_months = read_noaa_flask_zip(zf, gas=config_step.gas)

    print("df_events")
    display(df_events)

    print("df_months")
    display(df_months)

elif config_step.source == "in-situ":
    df_months = read_noaa_in_situ_zip(zf)

    print("df_months")
    display(df_months)
else:
    raise NotImplementedError(config_step.source)

# %%
if config_step.source == "surface-flask":
    config_step.interim_files["events_data"].parent.mkdir(exist_ok=True, parents=True)
    df_events.to_csv(config_step.interim_files["events_data"], index=False)

    config_step.interim_files["monthly_data"].parent.mkdir(exist_ok=True, parents=True)
    df_months.to_csv(config_step.interim_files["monthly_data"], index=False)

elif config_step.source == "in-situ":
    config_step.interim_files["monthly_data"].parent.mkdir(exist_ok=True, parents=True)
    df_months.to_csv(config_step.interim_files["monthly_data"], index=False)

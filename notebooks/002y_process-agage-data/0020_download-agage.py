# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AGAGE - download
#
# Download data from the [Advanced Global Atmospheric Gases Experiment (AGAGE)](https://agage.mit.edu/), specifically their [data page](https://agage.mit.edu/data/use-agage-data).
#
# For simplicity, we refer to all data from AGAGE, including its predecessors GAGE and ALE, as AGAGE data.

# %% [markdown]
# ## Imports

# %%
import tempfile
import urllib.request
from pathlib import Path

import openscm_units
import pint
import pooch
from attrs import evolve
from bs4 import BeautifulSoup
from pydoit_nb.complete import write_complete_file
from pydoit_nb.config_handling import get_config_for_step_id
from pydoit_nb.config_tools import URLSource

from local.config import load_config_from_file
from local.config_creation.agage_handling import AGAGE_GAS_MAPPING

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_agage_data"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "ch4_gc-md_monthly"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

# %% [markdown]
# ## Action

# %% [markdown]
# ### Find out which sources are available

# %%
time_frequency_map_url = {"monthly": "monthly_mean"}

# %%
start_url = f"https://agage2.eas.gatech.edu/data_archive/agage/{config_step.instrument}/{time_frequency_map_url[config_step.time_frequency]}"
print(f"{start_url=}")
soup_base = BeautifulSoup(
    urllib.request.urlopen(start_url).read(),  # noqa: S310
    "html.parser",
)
soup_base

# %%
gas_search = AGAGE_GAS_MAPPING[config_step.gas] if config_step.gas in AGAGE_GAS_MAPPING else config_step.gas
print(f"Searching for {gas_search} in URLs")

url_sources = []
for link in soup_base.find_all("a"):
    loc = link.get("href")
    if loc.endswith("/") and not loc.startswith("/"):
        print("-----")
        print(f"Checking observing location: {loc}")
        url_loc = f"{start_url}/{loc}"
        print(f"{url_loc=}")

        soup_loc = BeautifulSoup(
            urllib.request.urlopen(url_loc).read(),  # noqa: S310
            "html.parser",
        )
        soup_loc_file_formats = [
            link.get("href")
            for link in soup_loc.find_all("a")
            if link.get("href").endswith("/") and not link.get("href").startswith("/")
        ]
        if soup_loc_file_formats != ["ascii/"]:
            raise AssertionError("Unexpected data formats")  # noqa: TRY003

        for file_format in soup_loc_file_formats:
            url_loc_file_format = f"{start_url}/{loc}{file_format}"
            print(f"{url_loc_file_format=}")

            soup_loc_file_format = BeautifulSoup(
                urllib.request.urlopen(url_loc_file_format).read(),  # noqa: S310
                "html.parser",
            )
            soup_loc_gas_format_data_files = [
                link.get("href")
                for link in soup_loc_file_format.find_all("a")
                if link.get("href").endswith(".txt") and f"_{gas_search}_" in link.get("href")
            ]
            if config_step.gas == "h2":
                # Weird, not sure what this file is meant for
                soup_loc_gas_format_data_files = [
                    file for file in soup_loc_gas_format_data_files if "h2_pdd" not in file
                ]

            if len(soup_loc_gas_format_data_files) == 0:
                print(
                    f"No data available for {config_step.gas} from observing site {loc}. "
                    f"Looked for gas {gas_search!r} in URLs"
                )
                continue

            if len(soup_loc_gas_format_data_files) > 1:
                raise AssertionError(  # noqa: TRY003
                    f"Unexpected number of files, found: {soup_loc_gas_format_data_files}"
                )

            soup_loc_gas_format_data_file = soup_loc_gas_format_data_files[0]

            data_file_url = f"{start_url}/{loc}{file_format}{soup_loc_gas_format_data_file}"
            print(f"{data_file_url=}")
            url_sources.append(URLSource(url=data_file_url, known_hash="placeholder"))

url_sources

# %% [markdown]
# #### If requested, generate hashes

# %%
if config_step.generate_hashes:
    with tempfile.TemporaryDirectory() as tmp_path:
        # Download to tmp and get the hashes
        for i, source in enumerate(url_sources):
            tmp_file = pooch.retrieve(
                url=source.url,
                known_hash=None,
                path=Path(tmp_path),
                progressbar=True,
            )
            if isinstance(tmp_file, list):
                raise NotImplementedError("More than one file: tmp_file")

            hash = pooch.file_hash(tmp_file)
            url_sources[i] = evolve(source, known_hash=hash)

    print("Here are your URLSource's")
    print(url_sources)
    # print("Here are your serialised URLSource's")
    # print("")
    # print(converter_yaml.dumps(url_sources))

# %% [markdown]
# ### Make sure we're not missing any sources

# %%
missing_urls = set(v.url for v in url_sources) - set(v.url for v in config_step.download_urls)
if missing_urls:
    raise AssertionError(  # noqa: TRY003
        f"You are missing download urls for: {missing_urls}"
    )

# %% [markdown]
# ### Download

# %%
for url_source in config_step.download_urls:
    pooch.retrieve(
        url=url_source.url,
        known_hash=url_source.known_hash,
        fname=url_source.url.split("/")[-1],
        path=config_step.raw_dir,
        progressbar=True,
    )

# %%
write_complete_file(config_step.download_complete_file)

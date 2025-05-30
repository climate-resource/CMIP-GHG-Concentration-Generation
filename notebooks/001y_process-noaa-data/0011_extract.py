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

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # NOAA - extract
#
# Extract data from NOAA from the downloaded zip file.

# %% [markdown]
# ## Imports

# %%
import json
import zipfile
from pathlib import Path

import openscm_units
import pint
from attrs import asdict
from pydoit_nb.config_handling import get_config_for_step_id

import local.dependencies
from local.config import load_config_from_file
from local.noaa_processing import (
    read_noaa_flask_zip,
    read_noaa_hats,
    read_noaa_hats_combined,
    read_noaa_hats_m2_and_pr1,
    read_noaa_in_situ_zip,
)

# %%
pint.set_application_registry(openscm_units.unit_registry)  # type: ignore

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve_and_extract_noaa_data"

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "nf3_hats"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %% editable=true slideshow={"slide_type": ""}
config = load_config_from_file(Path(config_file))
config_step = get_config_for_step_id(config=config, step=step, step_config_id=step_config_id)

# %% [markdown]
# ## Action

# %%
files = [config_step.raw_dir / url_source.url.split("/")[-1] for url_source in config_step.download_urls]
files

# %%
assert len(files) == 1, "Re-think how you're doing this"
zf = files[0]

if config_step.source == "surface-flask":
    df_events, df_months = read_noaa_flask_zip(zf, gas=config_step.gas)

    print("df_events")
    print(df_events)

    print("df_months")
    print(df_months)

elif config_step.source == "in-situ":
    df_months = read_noaa_in_situ_zip(zf)

    print("df_months")
    print(df_months)

elif config_step.source == "hats":
    if config_step.gas in ("n2o", "ccl4", "cfc11", "cfc113", "cfc12", "sf6"):
        df_months = read_noaa_hats_combined(zf, gas=config_step.gas, source=config_step.source)

    elif config_step.gas in (
        "c2f6",
        "cf4",
        "halon1301",
        "hfc125",
        "hfc143a",
        "hfc236fa",
        "hfc32",
        "nf3",
        "so2f2",
    ):
        df_months = read_noaa_hats_m2_and_pr1(zf, gas=config_step.gas, source=config_step.source)

    else:
        df_months = read_noaa_hats(zf, gas=config_step.gas, source=config_step.source)

    print("df_months")
    print(df_months)

else:
    raise NotImplementedError(config_step.source)

# %% editable=true slideshow={"slide_type": ""}
if config_step.source == "surface-flask":
    config_step.interim_files["events_data"].parent.mkdir(exist_ok=True, parents=True)
    df_events.to_csv(config_step.interim_files["events_data"], index=False)

    config_step.interim_files["monthly_data"].parent.mkdir(exist_ok=True, parents=True)
    df_months.to_csv(config_step.interim_files["monthly_data"], index=False)

elif config_step.source == "in-situ":
    config_step.interim_files["monthly_data"].parent.mkdir(exist_ok=True, parents=True)
    df_months.to_csv(config_step.interim_files["monthly_data"], index=False)

elif config_step.source == "hats":
    config_step.interim_files["monthly_data"].parent.mkdir(exist_ok=True, parents=True)
    df_months.to_csv(config_step.interim_files["monthly_data"], index=False)

# %% editable=true slideshow={"slide_type": ""}
config_step.interim_files

# %%
if config_step.source == "hats":
    with open(zf) as fh:
        data_raw = fh.read()

    licence = "Reciprocity, see https://gml.noaa.gov/hats/hats_datause.html"

    readme_raw = "\n".join([v.replace("#  ", "") for v in data_raw.splitlines() if v.startswith("#")])

    citation_start_line = "Citation:"
    citation_increment = 1
    line_after_citation_startswith = "For contact"

else:
    with zipfile.ZipFile(zf) as zipf:
        readme = [
            item for item in zipf.filelist if item.filename.endswith("html") and "README" in item.filename
        ]
        if len(readme) > 1:
            raise AssertionError

        readme_raw = zipf.read(readme[0]).decode()

    if "CC0 1.0 Universal" not in readme_raw:
        raise AssertionError

    licence = "CC0 1.0 Universal"

    citation_start_line = "Please reference these data as"
    citation_increment = 1
    line_after_citation_startswith = "----"

# %%
in_citation = False
position = 0
readme_raw_split = readme_raw.splitlines()
citation_lines_l = []

while position < len(readme_raw_split):
    line = readme_raw_split[position]

    if line == citation_start_line:
        in_citation = True
        position += 1 + citation_increment
        continue

    if in_citation:
        if not line:
            if not readme_raw_split[position + 1].startswith(line_after_citation_startswith):
                raise AssertionError

            in_citation = False
        else:
            citation_lines_l.append(line)

    position += 1

# %%
url_l = [v.url for v in config_step.download_urls]
if len(url_l) > 1:
    raise AssertionError

url = url_l[0]

# %%
full_ref = " ".join(v.strip() for v in citation_lines_l)

if full_ref:
    doi = full_ref.split(" ")[-1]
    if "doi" not in doi:
        raise AssertionError


else:
    if not (
        config_step.gas.startswith("hfc")
        or config_step.gas.startswith("halon")
        or config_step.gas.startswith("hcfc")
        or (config_step.gas in ["ccl4", "cfc113", "ch2cl2", "ch3br", "ch3ccl3", "ch3cl"])
        # Note: when splitting out,
        # there seems to be no special info for these gases:
        # just use general refs
        or (config_step.gas in ["cf4", "c2f6", "so2f2", "nf3"])
    ):
        msg = f"{config_step.gas} {url}"
        raise AssertionError(msg)

    # TODO: split this out to cover all the cases given in
    # https://gml.noaa.gov/aftp/data/hats/hfcs/ReadMe.txt
    full_ref = (
        "S. A. Montzka, J. H. Butler, J. W. Elkins, T. M. Thompson, A. D. Clarke, and L. T. Lock, "
        "Present and Future Trends in the Atmospheric Burden of Ozone-Depleting Halogens, "
        "Nature, 398, 690-694, 1999."
    )
    doi = "https://doi.org/10.1038/19499"

source_info = local.dependencies.SourceInfo(
    short_name=f"NOAA {config_step.step_config_id.replace('_', ' ')}",
    licence=licence,
    reference=full_ref,
    url=url,
    doi=doi,
    resource_type="dataset",
)

source_info

# %%
with open(config_step.interim_files["source_info"], "w") as fh:
    json.dump(asdict(source_info), fh)

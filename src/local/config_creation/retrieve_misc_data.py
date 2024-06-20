"""
Retrieve misc data config creation
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_misc_data import (
    HadCRUT5Config,
    NaturalEarthConfig,
    PRIMAPConfig,
    RetrieveMiscDataConfig,
)

RETRIEVE_MISC_DATA_STEPS = [
    RetrieveMiscDataConfig(
        step_config_id="only",
        natural_earth=NaturalEarthConfig(
            raw_dir=Path("data/raw/natural_earth"),
            download_urls=[
                URLSource(
                    url="https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
                    known_hash="0f243aeac8ac6cf26f0417285b0bd33ac47f1b5bdb719fd3e0df37d03ea37110",
                )
            ],
            countries_shape_file_name="ne_110m_admin_0_countries.shx",
        ),
        primap=PRIMAPConfig(
            raw_dir=Path("data/raw/primap"),
            download_url=URLSource(
                url="https://zenodo.org/records/10705513/files/Guetschow_et_al_2024-PRIMAP-hist_v2.5.1_final_no_rounding_27-Feb-2024.nc?download=1",
                known_hash="be25ecff6639638015e3a7fc7b9488de9c048bddaed1fa1a7f1d08fde12e9c04",
            ),
        ),
        hadcrut5=HadCRUT5Config(
            raw_dir=Path("data/raw/hadcrut5"),
            download_url=URLSource(
                # Use the analysis time series, rather than non-infilled
                url="https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/analysis/diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.nc",
                known_hash="c1e6b0b6b372a428adea4fac109eca0278acf857ace4da0f43221fd0379ea353",
            ),
        ),
    )
]

"""
WMO 2022 ozone assessment ch. 7 data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.retrieve_and_process_wmo_2022_ozone_assessment_ch7_data import (
    RetrieveProcessWMO2022OzoneAssessmentCh7Config,
)
from local.dependencies import SourceInfo

RETRIEVE_AND_PROCESS_WMO_2022_OZONE_ASSESSMENT_CH7_DATA_STEPS = [
    RetrieveProcessWMO2022OzoneAssessmentCh7Config(
        step_config_id="only",
        raw_data=Path("data/raw/wmo-2022-ozone-assessment-ch7/wmo2022_Ch7_mixingratios.xlsx"),
        processed_data_file=Path(
            "data/interim/wmo-2022-ozone-assessment-ch7/wmo_2022_ozone_assessment_ch7.csv"
        ),
        source_info=SourceInfo(
            short_name="WMO 2022 Ozone Assessment Ch. 7",
            licence="Underlying data all openly licensed, so assuming the same, but not 100% clear",
            reference=(
                "Daniel, J. S., Reimann, S., ..., Schofield, R., Walter-Terrinoni, H. "
                "(2022). "
                "Chapter 7: Scenarios and Information for Policymakers. "
                "In World Meteorological Organization (WMO), "
                "Scientific Assessment of Ozone Depletion: 2022, GAW Report No. 278"
                "(pp. 509); WMO: Geneva, 2022."
            ),
            # Are there proper DOIs?
            doi=None,
            url="https://ozone.unep.org/sites/default/files/2023-02/Scientific-Assessment-of-Ozone-Depletion-2022.pdf",
            resource_type="publication-book",
        ),
    )
]

"""
WMO 2022 ozone assessment ch. 7 data handling config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.retrieve_and_process_wmo_2022_ozone_assessment_ch7_data import (
    RetrieveProcessWMO2022OzoneAssessmentCh7Config,
)

RETRIEVE_AND_PROCESS_WMO_2022_OZONE_ASSESSMENT_CH7_DATA_STEPS = [
    RetrieveProcessWMO2022OzoneAssessmentCh7Config(
        step_config_id="only",
        raw_data=Path("data/raw/wmo-2022-ozone-assessment-ch7/wmo2022_Ch7_mixingratios.xlsx"),
        processed_data_file=Path(
            "data/interim/wmo-2022-ozone-assessment-ch7/wmo_2022_ozone_assessment_ch7.csv"
        ),
    )
]

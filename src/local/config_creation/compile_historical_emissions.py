"""
Historical emissions config creation
"""

from __future__ import annotations

from pathlib import Path

from local.config.compile_historical_emissions import CompileHistoricalEmissionsConfig

COMPILE_HISTORICAL_EMISSIONS_STEPS = [
    CompileHistoricalEmissionsConfig(
        step_config_id="only",
        complete_historical_emissions_file=Path("data/processed/historical_emissions"),
    )
]

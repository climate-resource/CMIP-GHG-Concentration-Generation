"""
Create config for crunching equivalent species
"""

from __future__ import annotations

from pathlib import Path

from local.config.crunch_equivalent_species import EquivalentSpeciesCrunchingConfig

EQUIVALENT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "cfc11eq": (
        "c2f6",
        "c3f8",
        "c4f10",
        "c5f12",
        "c6f14",
        "c7f16",
        "c8f18",
        "cc4f8",
        "ccl4",
        "cf4",
        "cfc11",
        "cfc113",
        "cfc114",
        "cfc115",
        "ch2cl2",
        "ch3br",
        "ch3ccl3",
        "ch3cl",
        "chcl3",
        "halon1211",
        "halon1301",
        "halon2402",
        "hcfc141b",
        "hcfc142b",
        "hcfc22",
        "hfc125",
        "hfc134a",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc23",
        "hfc236fa",
        "hfc245fa",
        "hfc32",
        "hfc365mfc",
        "hfc4310mee",
        "nf3",
        "sf6",
        "so2f2",
    ),
    "cfc12eq": (
        "cfc11",
        "cfc113",
        "cfc114",
        "cfc115",
        "cfc12",
        "ccl4",
        "ch2cl2",
        "ch3br",
        "ch3ccl3",
        "ch3cl",
        "chcl3",
        "halon1211",
        "halon1301",
        "halon2402",
        "hcfc141b",
        "hcfc142b",
        "hcfc22",
    ),
    "hfc134aeq": (
        "c2f6",
        "c3f8",
        "c4f10",
        "c5f12",
        "c6f14",
        "c7f16",
        "c8f18",
        "cc4f8",
        "cf4",
        "hfc125",
        "hfc134a",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc23",
        "hfc236fa",
        "hfc245fa",
        "hfc32",
        "hfc365mfc",
        "hfc4310mee",
        "nf3",
        "sf6",
        "so2f2",
    ),
}


def create_crunch_equivalent_species_config(
    gases: tuple[str, ...]
) -> list[EquivalentSpeciesCrunchingConfig]:
    """
    Create configuration for crunching the equivalent species

    Parameters
    ----------
    gases
        Gases for which to create equivalent species crunching configurations

    Returns
    -------
        Equivalent species crunching configurations for the requested gases
    """
    out = []

    for gas in gases:
        interim_dir = Path("data/interim") / gas

        out.append(
            EquivalentSpeciesCrunchingConfig(
                step_config_id=gas,
                gas=gas,
                equivalent_component_gases=EQUIVALENT_COMPONENTS[gas],
                fifteen_degree_monthly_file=interim_dir
                / f"{gas}_fifteen-degree_monthly.nc",
                # half_degree_monthly_file=interim_dir / f"{gas}_half-degree_monthly.nc",
                global_mean_monthly_file=interim_dir / f"{gas}_global-mean_monthly.nc",
                hemispheric_mean_monthly_file=interim_dir
                / f"{gas}_hemispheric-mean_monthly.nc",
                global_mean_annual_mean_file=interim_dir
                / f"{gas}_global-mean_annual-mean.nc",
                hemispheric_mean_annual_mean_file=interim_dir
                / f"{gas}_hemispheric-mean_annual-mean.nc",
            )
        )

    return out

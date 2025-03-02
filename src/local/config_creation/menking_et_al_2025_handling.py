"""
Menking et al. (2025) data handling config creation
"""

from __future__ import annotations

from pathlib import Path

import local.config_creation.law_dome_handling
from local.config.retrieve_and_process_menking_et_al_2025_data import (
    RetrieveExtractMenkingEtal2025Data,
)
from local.dependencies import SourceInfo

RAW_DIR = Path("data/raw/menking-et-al-2025")

RETRIEVE_AND_PROCESS_MENKING_ET_AL_2025_DATA_STEPS = [
    RetrieveExtractMenkingEtal2025Data(
        step_config_id="only",
        raw_data_file=RAW_DIR / "spline-fits-for-ZN_CMIP7.xlsx",
        expected_hash="20df9337cddb739dc805e53847a19424",
        processed_data_file=Path("data/interim/menking-et-al-2025/menking_et_al_2025.csv"),
        source_info=SourceInfo(
            short_name="Menking et al., 2025 (in-prep.)",
            licence="Author supplied",
            reference=(
                "Menking, J. A., Etheridge, D., ..., Spencer, D., and Caldow, C. (in prep.). "
                "Filling gaps and reducing uncertainty in existing Law Dome ice core records."
            ),
            doi=None,
            url="author-supplied.invalid",
            resource_type="publication-article",
        ),
        second_order_deps={
            "co2": (
                local.config_creation.law_dome_handling.SOURCE_INFO,
                SourceInfo(
                    short_name="Bauska et al., 2015",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Bauska, T., Joos, F., Mix, A. et al. "
                        "Links between atmospheric carbon dioxide, "
                        "the land carbon reservoir and climate over the past millennium. "
                        "Nature Geosci 8, 383-387 (2015). https://doi.org/10.1038/ngeo2422"
                    ),
                    doi="https://doi.org/10.1038/ngeo2422",
                    url="https://doi.org/10.1038/ngeo2422",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="King et al., 2024",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "King, A.C.F., Bauska, T.K., Brook, E.J. et al. "
                        "Reconciling ice core CO2 and land-use change following New World-Old World contact. "
                        "Nat Commun 15, 1735 (2024). https://doi.org/10.1038/s41467-024-45894-9"
                    ),
                    doi="https://doi.org/10.1038/s41467-024-45894-9",
                    url="https://doi.org/10.1038/s41467-024-45894-9",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Ahn et al., 2012",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Ahn, J., E. J. Brook, A. Schmittner, and K. Kreutz (2012), "
                        "Abrupt change in atmospheric CO2 during the last ice age, "
                        "Geophys. Res. Lett., 39, L18711, doi:10.1029/2012GL053018."
                    ),
                    doi="https://doi.org/10.1029/2012GL053018",
                    url="https://doi.org/10.1029/2012GL053018",
                    resource_type="publication-article",
                ),
            ),
            "n2o": (
                local.config_creation.law_dome_handling.SOURCE_INFO,
                SourceInfo(
                    short_name="Ghosh et al., 2023",
                    licence="Creative Commons Attribution Only v4.0 Generic [CC BY 4.0]",
                    reference=(
                        "Ghosh, S., Toyoda, S., Buizert, C., ..., Yoshida, N., et al. (2023). "
                        "Concentration and isotopic composition of atmospheric N2O "
                        "over the last century. "
                        "Journal of Geophysical Research: Atmospheres, 128, e2022JD038281. "
                        "https://doi.org/10.1029/2022JD038281"
                    ),
                    doi="https://doi.org/10.15784/601693",
                    url="https://doi.org/10.15784/601693",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Schilt et al., 2010",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Schilt A, Baumgartner M, Blunier T, et al. "
                        "Glacial-interglacial and millennial-scale variations in the atmospheric nitrous "
                        "oxide concentration during the last 800,000 years. "
                        "Quaternary Science Reviews. 2010;29(1):182-192. "
                        "doi:10.1016/j.quascirev.2009.03.011"
                    ),
                    doi="https://doi.org/10.1016/j.quascirev.2009.03.011",
                    url="https://doi.org/10.1016/j.quascirev.2009.03.011",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Azharuddin et al., 2024",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Azharuddin, S., Ahn, J., Ryu, Y., Brook, E., & Salehnia, N. (2024). "
                        "Millennial-scale changes in atmospheric nitrous oxide during the Holocene. "
                        "Earth and Space Science, 11, e2023EA002840. https://doi.org/10.1029/2023EA002840"
                    ),
                    doi="https://doi.org/10.1029/2023EA002840",
                    url="https://doi.org/10.1029/2023EA002840",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Prokopiou et al., 2018",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Prokopiou, M., Sapart, C. J., Rosen, J., ..., & Röckmann, T. (2018). "
                        "Changes in the isotopic signature of atmospheric nitrous oxide "
                        "and its global average source during the last three millennia. "
                        "Journal of Geophysical Research: Atmospheres, 123, 10,757-10,773. "
                        "https://doi.org/10.1029/2018JD029008"
                    ),
                    doi="https://doi.org/10.1029/2018JD029008",
                    url="https://doi.org/10.1029/2018JD029008",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Ryu et al., 2020",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Ryu, Y., Ahn, J., Yang, J.-W., Brook, E. J., Timmermann, A., Blunier, T., et al. "
                        "(2020). Atmospheric nitrous oxide variations on centennial time scales "
                        "during the past two millennia. Global Biogeochemical Cycles, 34, e2020GB006568. "
                        "https://doi.org/10.1029/2020GB006568"
                    ),
                    doi="https://doi.org/10.1029/2020GB006568",
                    url="https://doi.org/10.1029/2020GB006568",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Ishijima et al., 2007",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Ishijima, K., S. Sugawara, K. Kawamura, G. Hashida, ..., and T. Nakazawa (2007), "
                        "Temporal variations of the atmospheric nitrous oxide concentration and its δ15N and "
                        "δ18O for the latter half of the 20th century reconstructed from firn air analyses, "
                        "J. Geophys. Res., 112, D03305, doi:10.1029/2006JD007208."
                    ),
                    doi="https://doi.org/10.1029/2006JD007208",
                    url="https://doi.org/10.1029/2006JD007208",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Park et al., 2012",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Park, S., Croteau, P., Boering, K. et al. "
                        "Trends and seasonal cycles in the isotopic composition of nitrous oxide since 1940. "
                        "Nature Geosci 5, 261-265 (2012). https://doi.org/10.1038/ngeo1421"
                    ),
                    doi="https://doi.org/10.1038/ngeo1421",
                    url="https://doi.org/10.1038/ngeo1421",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Bernard et al., 2006",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Bernard, S., Röckmann, T., Kaiser, J., Barnola, J.-M., ..., and Chappellaz, J.: "
                        "Constraints on N2O budget changes since pre-industrial time from new firn air "
                        "and ice core isotope measurements, "
                        "Atmos. Chem. Phys., 6, 493-503, https://doi.org/10.5194/acp-6-493-2006, 2006."
                    ),
                    doi="https://doi.org/10.5194/acp-6-493-2006",
                    url="https://doi.org/10.5194/acp-6-493-2006",
                    resource_type="publication-article",
                ),
                SourceInfo(
                    short_name="Roeckmann et al., 2006",
                    licence="Supplied by Menking et al. author",
                    reference=(
                        "Roeckmann, T., Kaiser, J., and Brenninkmeijer, C. A. M.: "
                        "The isotopic fingerprint of the pre-industrial "
                        "and the anthropogenic N2O source, "
                        "Atmos. Chem. Phys., 3, 315-323, "
                        "https://doi.org/10.5194/acp-3-315-2003, 2003."
                    ),
                    doi="https://doi.org/10.5194/acp-3-315-2003",
                    url="https://doi.org/10.5194/acp-3-315-2003",
                    resource_type="publication-article",
                ),
            ),
        },
    )
]

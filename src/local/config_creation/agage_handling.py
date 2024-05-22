"""
Creation of configuration for handling AGAGE's data
"""

from __future__ import annotations

from pathlib import Path

from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_extract_agage import RetrieveExtractAGAGEDataConfig

DOWNLOAD_URLS = {
    ("ch4", "gc-md", "monthly"): [
        URLSource(
            known_hash="e6c3955c0e9178333c5f2177088a9fe84ec27b557901364750a82241f3477300",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_ch4_mon.txt",
        ),
        URLSource(
            known_hash="91cbef846e4158a880515b3b86b5b28d7510dcc6cf9494e3fec823e0c3f0678c",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_ch4_mon.txt",
        ),
        URLSource(
            known_hash="3d295bad0b883b6099ed5171044ed7a46e5ae93e8646a2020058a72c648ed0a6",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_ch4_mon.txt",
        ),
        URLSource(
            known_hash="e775e79fcf6cb833aa7d139c79725f25aefb81d4e90557616c4939d497f80719",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_ch4_mon.txt",
        ),
        URLSource(
            known_hash="fceb3a14534ce94d550f24831c7fc1258700f24b1a917005b6c06a85843ce0e1",
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_ch4_mon.txt",
        ),
    ],
    ("n2o", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_n2o_mon.txt",
            known_hash="7ac04ee39e56544dc6d98e68d52be58ab30cde620f511d70192a005c95b85fc0",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_n2o_mon.txt",
            known_hash="7a0d7481d6d5492bf4501c107c57c6f6b0bb9991eb9ff23abb8c9c966fefa79c",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_n2o_mon.txt",
            known_hash="849efb23da2bfad5e76e5abd7303bb3b74b3bd9f924daa4303bbc10d04cf67da",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_n2o_mon.txt",
            known_hash="3739081cf7778e205dc1c77c74ffdb72d748baaf346858850fd39e41f16f42c3",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_n2o_mon.txt",
            known_hash="e4cdc474f71d6f80ac63fe13b7fa61a86ce0361b7995874bf074a01c69facac3",
        ),
    ],
    ("sf6", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim-sf6-ecd/ascii/AGAGE-GC-ECD-SF6_CGO_sf6_mon.txt",
            known_hash="9ec255bbd55447d8ac521a4683951e0b0aa682d8252a1072e5fa555b90af5aa1",
        )
    ],
    ("sf6", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/barbados/ascii/AGAGE-GCMS-Medusa_RPB_sf6_mon.txt",
            known_hash="1611fb6ed6087b506af19ca8a08cdf750e2c185b136b2460ba013ace39b47714",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/capegrim/ascii/AGAGE-GCMS-Medusa_CGO_sf6_mon.txt",
            known_hash="d387ab096cc53fae4efa5026eaaba4f3df0ceec7a1afcfc1128687372e6505d3",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_sf6_mon.txt",
            known_hash="b556daf22abd2edf0874b875a60bd02f246315c4c5a9748273362cb210c7e077",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_sf6_mon.txt",
            known_hash="030a37af25e2d806d8ac65be66f264f449923a1b8b077c92c647577d4efe3720",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/macehead/ascii/AGAGE-GCMS-Medusa_MHD_sf6_mon.txt",
            known_hash="42d1f57226972df7d16786310e95288db0b604033d8fac82b8b03630d36d908a",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_sf6_mon.txt",
            known_hash="cb05383d875d6020a0017942551f909954354ed2701a12754da6bdb80e45612f",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/samoa/ascii/AGAGE-GCMS-Medusa_SMO_sf6_mon.txt",
            known_hash="95fa2ccc93fe2dfefcb64d1405e81e0bc556a1892e5b4818312003de92eaf23b",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_sf6_mon.txt",
            known_hash="12270c0ab91decaaffc0f47710c27de9147a125938c3e98b1c10a98a2922f441",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/trinidad/ascii/AGAGE-GCMS-Medusa_THD_sf6_mon.txt",
            known_hash="da146fc89d0e2b2e87846c6e5fc5712820a5f7bff69f054d90ac0ce80a1cf2a7",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_sf6_mon.txt",
            known_hash="2de77c7f417510878c18d4ea452815ac4555ae17719e6786548b063ee471e5bf",
        ),
    ],
    ("cfc11", "gc-md", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/barbados/ascii/AGAGE-GCMD_RPB_cfc-11_mon.txt",
            known_hash="7f52297786487e9ede8a785c89b1d793250c4198223bafaa265cdcc268bbc978",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/capegrim/ascii/AGAGE-GCMD_CGO_cfc-11_mon.txt",
            known_hash="ebc838159a6ccc0bb98ac16203e927abea3371aa8aea801db572c816761074cd",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/macehead/ascii/AGAGE-GCMD_MHD_cfc-11_mon.txt",
            known_hash="aa6ef6875861e0d127fadc4697467b08ba8272d0fd3be91dd63713490de86645",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/samoa/ascii/AGAGE-GCMD_SMO_cfc-11_mon.txt",
            known_hash="d1722aa15bf3a77415b97f0f9a1c1e58912e0ace054b00c8767e75666f877318",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-md/monthly/trinidad/ascii/AGAGE-GCMD_THD_cfc-11_mon.txt",
            known_hash="9929b38d196ef1397615b51988631f1e790797d86c260f57b387074cb667ef56",
        ),
    ],
    ("cfc11", "gc-ms-medusa", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/gosan/ascii/AGAGE-GCMS-Medusa_GSN_cfc-11_mon.txt",
            known_hash="6d8b653daf80d3fd8c295c91e8842e8c81242c36a543da88b19964c1de7ef7ad",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/jungfraujoch/ascii/AGAGE-GCMS-Medusa_JFJ_cfc-11_mon.txt",
            known_hash="1f2dd4650b49c7ee5d9e5a763e1be3daeb72f0243ea249ab3b9c9d586e71f8c4",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/mtecimone/ascii/AGAGE-GCMS-Medusa_CMN_cfc-11_mon.txt",
            known_hash="35231a8b2a6776e815c843ad7ba0f99378733dbbaa6c112f33d30d0558e66ad8",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/tacolneston/ascii/AGAGE-GCMS-Medusa_TAC_cfc-11_mon.txt",
            known_hash="33fbb30c985d2ae36a48f5a7e6e92e66ea84c83004db006ca2fb1de72f922112",
        ),
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms-medusa/monthly/zeppelin/ascii/AGAGE-GCMS-Medusa_ZEP_cfc-11_mon.txt",
            known_hash="f1eb53ecfa4294ef3581b536804fccac36519dbc4ddafa856c4c4aeb9e7aa048",
        ),
    ],
    ("cfc11", "gc-ms", "monthly"): [
        URLSource(
            url="https://agage2.eas.gatech.edu/data_archive/agage/gc-ms/monthly/mtecimone/ascii/AGAGE-GCMS-MteCimone_CMN_cfc-11_mon.txt",
            known_hash="3394d57cc17222ccc5de8f98edbe26131afc63e718ff9d4563727a098704aa93",
        )
    ],
}


def create_agage_handling_config(
    data_sources: tuple[tuple[str, str, str]],
) -> list[RetrieveExtractAGAGEDataConfig]:
    """
    Create config for handling AGAGE data

    Parameters
    ----------
    data_sources
        Data sources to retrieve.
        Each input tuple should contain
        the gas of interest (zeroth element),
        the instrument of interest (first element)
        and the time frequency of interest (second element).

    Returns
    -------
        Configuration for handling AGAGE data for the requested data sources.
    """
    res = []
    for data_source in data_sources:
        gas, instrument, frequency = data_source

        raw_dir = Path("data/raw/agage/agage")
        interim_dir = Path("data/interim/agage/agage")

        res.append(
            RetrieveExtractAGAGEDataConfig(
                step_config_id=f"{gas}_{instrument}_{frequency}",
                gas=gas,
                instrument=instrument,
                time_frequency=frequency,
                raw_dir=raw_dir,
                download_complete_file=raw_dir
                / f"{gas}_{instrument}_{frequency}.complete",
                processed_monthly_data_with_loc_file=interim_dir
                / f"{gas}_{instrument}_{frequency}.csv",
                generate_hashes=False,
                download_urls=DOWNLOAD_URLS[(gas, instrument, frequency)],
            )
        )

    return res

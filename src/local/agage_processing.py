"""
Helpers for handling AGAGE data and metadata
"""

from __future__ import annotations

import re
import string

from local.dependencies import SourceInfo


def get_source_info(ref_l: list[str], max_author_len: int = 75) -> SourceInfo:
    """
    Get source info for a given reference block

    Parameters
    ----------
    ref_l
        Reference block

    max_author_len
        Maximum length of authors to show

    Returns
    -------
    :
        Source info for the reference block
    """
    full_ref = " ".join([v.strip() for v in ref_l]).strip()

    if (
        "A History of Chemically and Radiatively Important Gases in Air " "deduced from ALE/GAGE/AGAGE"
    ) in full_ref:
        return SourceInfo(
            short_name="AGAGE",
            licence="Free for scientific use, offer co-authorship. See https://www-air.larc.nasa.gov/missions/agage/data/policy",
            reference=(
                "Prinn et al., A history of chemically and radiatively important "
                "gases in air deduced from ALE/GAGE/AGAGE, J. Geophys. Res., 105, "
                "No. D14, p17,751-17,792, 2000."
            ),
            url="https://agage2.eas.gatech.edu/data_archive/agage/readme",
            resource_type="dataset",
        )

    toks_comma = full_ref.split(",")
    first_author = toks_comma[0].strip()
    year = toks_comma[-1].strip().strip(".")
    doi = toks_comma[-2].strip()
    if "doi.org" not in doi:
        doi = f"https://doi.org/{doi}"

    if ".:" in full_ref:
        author_rest_delim = ".:"
        author_list, rest = full_ref.split(author_rest_delim, maxsplit=1)
        author_delim = ".,"
        ellipsis = ".."

        authors = author_list.split(author_delim)

    else:
        author_rest_delim = ","
        authors_most, tmp = full_ref.split(", and ", maxsplit=1)
        last_author, rest = tmp.split(",", maxsplit=1)
        author_delim = ","
        ellipsis = "..."

        authors_most_split = authors_most.split(",")
        authors = [",".join(authors_most_split[:2]), *authors_most_split[2:], f" and {last_author}"]

    authors_keep_l = [
        authors[0],
        authors[1],
        f" {ellipsis}",
        authors[-2],
        authors[-1],
    ]

    for i, a_add in enumerate(authors[1:-2]):
        author_list_cut = ".,".join(authors_keep_l)
        if len(author_list_cut) + len(a_add) < max_author_len:
            authors_keep_l.insert(i + 1, a_add)
        else:
            break

    author_list_cut = author_delim.join(authors_keep_l)

    res = SourceInfo(
        short_name=f"AGAGE {first_author} et al. {year}",
        licence="Paper, NA",
        reference=author_rest_delim.join([author_list_cut, rest]),
        url=doi,
        doi=doi,
        resource_type="publication-article",
    )

    return res


def extract_agage_source_info(raw_readme: str, gas: str) -> tuple[SourceInfo, ...]:  # noqa: PLR0912, PLR0915
    """
    Extract AGAGE source information

    Parameters
    ----------
    raw_readme
        Raw README text

    gas
        Gas for which to extract source information

    Returns
    -------
    :
        Source information for the given gas
    """
    gas_block_indicators_d = {"ccl4": ["CCl4"], "chcl3": ["CHCl3"]}
    if gas in gas_block_indicators_d:
        gas_block_indicators = gas_block_indicators_d[gas]
    else:
        gas_block_indicators = [gas.upper()]

    if gas in [
        "c2f6",
        "c3f8",
        "cc4f8",
        "ccl4",
        "cf4",
        "cfc11",
        "cfc113",
        "cfc114",
        "cfc115",
        "cfc12",
        "ch3ccl3",
        "hcfc141b",
        "hcfc142b",
        "hcfc22",
        "hfc125",
        "hfc134a",
        "hfc143a",
        "hfc152a",
        "hfc227ea",
        "hfc23",
        "hfc245fa",
        "hfc32",
        "hfc365mfc",
        "nf3",
        "sf6",
        "so2f2",
    ]:
        gas_block_indicators.append("Synthetic GHG")

    if gas in ["cfc11", "cfc113", "cfc12", "ch3ccl3"]:
        gas_block_indicators.append("Major CFCs/CH3CCl3")

    if gas in [
        "cfc13",
        "cfc114",
        "cfc115",
    ]:
        gas_block_indicators.append("Minor CFCs")

    if "hfc" in gas:
        gas_block_indicators.append("HFCs")

    if gas in ["hcfc22", "hfc23"]:
        gas_block_indicators.append("HCFC-22/HFC-23")

    if "hcfc" in gas or "hfc" in gas:
        gas_block_indicators.append("HCFCs/HFCs")

    if "halon" in gas:
        gas_block_indicators.append("Halons")

    if gas in ["ch2cl2", "chcl3", "ch3cl", "ch3br"]:
        gas_block_indicators.append("VSLS")

    if gas in ["cf4", "c2f6", "c3f8", "nf3"]:
        gas_block_indicators.append("PFCs (CF4, C2F6, C3F8), and NF3")

    source_info_l = []

    indent = "   "
    general_ref_start = "(i) General reference"
    in_general_refs = False
    in_gas_block = False
    found_gas_block = False
    position = 0
    readme_raw_split = tuple(raw_readme.splitlines())
    while position < len(readme_raw_split):
        line = readme_raw_split[position]

        if line.startswith(general_ref_start):
            in_general_refs = True
            position += 1
            continue

        elif in_general_refs and line.strip() and not line.startswith(indent):
            in_general_refs = False
            position += 1
            continue

        if any(line.startswith(gbi) for gbi in gas_block_indicators):
            in_gas_block = True
            found_gas_block = True
            position += 1
            continue

        elif in_gas_block and line.strip() and not line.startswith(indent):
            in_gas_block = False
            position += 1
            continue

        if in_general_refs or in_gas_block:
            if not line.strip():
                position += 1
                continue

            ref_l = []
            while line.strip():
                ref_l.append(line)

                position += 1
                line = readme_raw_split[position]

            try:
                si = get_source_info(ref_l)

                if si.short_name != "AGAGE":
                    matches = [v for v in source_info_l if v.short_name.startswith(si.short_name)]
                    if matches:
                        si.short_name = f"{si.short_name} {string.ascii_lowercase[len(matches)]}"

                source_info_l.append(si)

            except:
                print("\n".join(ref_l))
                raise

        position += 1

    if not found_gas_block:
        msg = f"{raw_readme}\n{gas}"
        raise AssertionError(msg)

    res = tuple(source_info_l)

    if "hfc" in gas:
        res_ll = []
        for si in res:
            hfcs_mentioned = [
                v.lower().replace("-", "") for v in re.findall("HFC-[0-9a-zA-Z-]*", si.reference)
            ]
            if (not hfcs_mentioned) or (gas in hfcs_mentioned):
                # HFCs not named, or our HFC named explicitly
                res_ll.append(si)

            res = tuple(res_ll)

    return res

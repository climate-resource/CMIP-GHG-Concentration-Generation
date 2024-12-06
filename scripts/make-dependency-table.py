"""
Make a table of dependencies

This is a temporary solution.
In future work (https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/issues/62),
we will create dependency information on the fly.
"""

# ruff: noqa: D101, D102, D103

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import pandas as pd
import pygraphviz
import tqdm
import typer
from attrs import asdict, define


def get_doit_list_all(pixi: str, config_file: str) -> tuple[str, ...]:
    doit_list_all = subprocess.check_output(
        [pixi, "run", "-e", "all-dev", "doit", "list", "--all", "--quiet"],  # noqa: S603
        env={"DOIT_CONFIGURATION_FILE": config_file},
    )

    return tuple(doit_list_all.decode().splitlines())


def extract_input4mips_writing_tasks(doit_list_all: tuple[str, ...]) -> tuple[str, ...]:
    input4mips_writing_tasks = []
    for task in doit_list_all:
        task_clean = task.strip()
        if "write input4MIPs - write all files:" in task_clean and "eq" not in task_clean:
            input4mips_writing_tasks.append(task_clean)

    return tuple(input4mips_writing_tasks)


def get_dependency_dot_files(
    pixi: str, input4mips_writing_tasks: tuple[str, ...], force_generation: bool
) -> dict[str, Path]:
    dot_files = {}
    for writing_task in tqdm.tqdm(input4mips_writing_tasks, desc="Creating dot files"):
        gas = writing_task.split(":")[-1]

        out_dot = Path(f"{gas}-dependencies.dot")
        if force_generation or not out_dot.exists():
            subprocess.check_output(
                [  # noqa: S603
                    pixi,
                    "run",
                    "-e",
                    "all-dev",
                    "doit",
                    "graph",
                    "-o",
                    str(out_dot),
                    "--reverse",
                    "--show-subtasks",
                    writing_task,
                ]
            )

        dot_files[gas] = out_dot

    return dot_files


@define
class SourceInfo:
    licence: str
    reference: str
    doi: str


source_info: dict[str, SourceInfo] = {
    # Need to get in contact with AGAGE station PIs
    # Need to cite the DOI of the dataset
    # Need to cite most recent paper here: https://www-air.larc.nasa.gov/missions/agage/publications/
    # TODO: add acknowledgements
    # Publications must state:
    # “AGAGE is supported principally by the National Aeronautics and Space Administration (USA) "
    # "grants to the Massachusetts Institute of Technology and the Scripps Institution of Oceanography.
    # Additional statements must be included to acknowledge funding for the individual stations.
    # These are located on the individual station pages.
    # Obviously too simple, but better than zero
    **{
        key: SourceInfo(
            licence="Free for scientific use, offer co-authorship. See https://www-air.larc.nasa.gov/missions/agage/data/policy",
            # Actually incorrect, should be using gas-specific references
            reference="Prinn et al., Earth Syst. Sci. Data 2018",
            # Actually incorrect, should be using the individual station DOIs.
            # Do that next time.
            doi="https://doi.org/10.5194/essd-10-985-2018",
        )
        for key in ("AGAGE gc-ms-medusa", "AGAGE gc-ms", "AGAGE gc-md", "AGAGE GAGE", "AGAGE ALE")
    },
    # Need to get in contact with GML
    # Consider whether obs pack is a better source for future
    "NOAA HATS": SourceInfo(
        licence="Free for scientific use, offer co-authorship. See https://gml.noaa.gov/hats/hats_datause.html",
        # Incorrect, will need to work out how to navigate the NOAA website properly
        reference="https://gml.noaa.gov/hats/",
        # Obviously incorrect, but better than zero
        doi="https://gml.noaa.gov/hats/",
    ),
    **{
        key: SourceInfo(
            licence="Free for scientific use, offer co-authorship. See https://gml.noaa.gov/ccgg/data/datause.html",
            # Incorrect, will need to work out how to navigate the NOAA website properly
            reference="https://gml.noaa.gov/ccgg",
            # Obviously incorrect, but better than zero
            doi="https://gml.noaa.gov/ccgg",
        )
        for key in ("NOAA surface flask", "NOAA in-situ")
    },
    "WMO 2022 ozone assessment Ch. 7": SourceInfo(
        licence="Underlying data all openly licensed, so assuming the same, but not 100% clear",
        # TODO: see if we can get WMO to provide bibtex
        reference=(
            "Daniel, J. S., Reimann, S., Ashford, P., Fleming, E. L., Hossaini, R., Lickley, M. J., Schofield, R., Walter-Terrinoni, H. "  # noqa: E501
            "(2022). "
            "Chapter 7: Scenarios and Information for Policymakers. "
            "In World Meteorological Organization (WMO), "
            "Scientific Assessment of Ozone Depletion: 2022, GAW Report No. 278"
            "(pp. 509); WMO: Geneva, 2022."
        ),
        # Are there proper DOIs?
        doi="https://ozone.unep.org/sites/default/files/2023-02/Scientific-Assessment-of-Ozone-Depletion-2022.pdf",
    ),
    "EPICA": SourceInfo(
        licence="CC BY 3.0",
        reference=(
            "EPICA Community Members (2006): Methane of ice core EDML [dataset]. "
            "PANGAEA, https://doi.org/10.1594/PANGAEA.552232, In supplement to: "
            "Barbante, Carlo; Barnola, Jean-Marc; ... Wolff, Eric William (2006): "
            "One-to-one coupling of glacial climate variability in Greenland and Antarctica. "
            "Nature, 444, 195-198, https://doi.org/10.1038/nature05301"
        ),
        doi="https://doi.pangaea.de/10.1594/PANGAEA.552232",
    ),
    "NEEM": SourceInfo(
        licence="CC BY 4.0",
        reference=(
            "Rhodes, Rachael H; Brook, Edward J (2019): "
            "Methane in NEEM-2011-S1 ice core from North Greenland, "
            "1800 years continuous record: 5 year median, v2 [dataset]. "
            "PANGAEA, https://doi.org/10.1594/PANGAEA.899039, In supplement to: "
            "Rhodes, Rachael H; Faïn, Xavier; ...; Brook, Edward J (2013): "
            "Continuous methane measurements from a late Holocene Greenland ice core: "
            "Atmospheric and in-situ signals. Earth and Planetary Science Letters, 368, 9-19, "
            "https://doi.org/10.1016/j.epsl.2013.02.034"
        ),
        doi="https://doi.pangaea.de/10.1594/PANGAEA.899039",
    ),
    "Law Dome": SourceInfo(
        licence="CC BY 4.0",
        reference=(
            "Rubino, Mauro; Etheridge, David; ... Van Ommen, Tas; & Smith, Andrew (2019): "
            "Law Dome Ice Core 2000-Year CO2, CH4, N2O and d13C-CO2. v3. CSIRO. "
            "Data Collection. https://doi.org/10.25919/5bfe29ff807fb"
        ),
        doi="https://doi.org/10.25919/5bfe29ff807fb",
    ),
    "Western et al., 2024": SourceInfo(
        licence="CC BY 4.0",  # https://zenodo.org/records/10782689
        reference=(
            "Western, L.M., Daniel, J.S., Vollmer, M.K. et al. "
            "A decrease in radiative forcing "
            "and equivalent effective chlorine from hydrochlorofluorocarbons. "
            "Nat. Clim. Chang. 14, 805-807 (2024)."
        ),
        doi="https://doi.org/10.1038/s41558-024-02038-7",
    ),
    "Velders et al., 2022": SourceInfo(
        licence="Other (Open)",  # https://zenodo.org/records/6520707
        reference=(
            "Velders, G. J. M., Daniel, J. S., ... Weiss, R. F., and Young, D.: "
            "Projections of hydrofluorocarbon (HFC) emissions "
            "and the resulting global warming based on recent trends in observed abundances "
            "and current policies, "
            "Atmos. Chem. Phys., 22, 6087-6101, "
            "https://doi.org/10.5194/acp-22-6087-2022, 2022."
        ),
        doi="https://doi.org/10.5194/acp-22-6087-2022",
    ),
    "Droste et al., 2020": SourceInfo(
        licence="CC BY 4.0",  # https://zenodo.org/records/3519317
        reference=(
            "Droste, E. S., Adcock, K. E., ..., Sturges, W. T., and Laube, J. C.: "
            "Trends and emissions of six perfluorocarbons "
            "in the Northern Hemisphere and Southern Hemisphere, "
            "Atmos. Chem. Phys., 20, 4787-4807, https://doi.org/10.5194/acp-20-4787-2020, 2020."
        ),
        doi="https://doi.org/10.5194/acp-20-4787-2020",
    ),
}


@define
class DependencyInfoSource:
    gas: str
    source: str
    licence: str
    reference: str
    doi: str


@define
class DependencyInfo:
    sources: tuple[DependencyInfoSource, ...]

    def by_gas_serialised(self) -> dict[str, list[dict[str, str], ...]]:
        res = defaultdict(list)
        for source in self.sources:
            res[source.gas].append(asdict(source))

        return res


def extract_dependencies(dot_files: dict[str, Path]) -> DependencyInfo:  # noqa: PLR0912, PLR0915
    dependency_info_l = []
    for gas, dot_file in tqdm.tqdm(dot_files.items(), desc="Extracting dependencies from dot files"):
        gas_graph = pygraphviz.AGraph(dot_file, strict=False, directed=True)
        input_data_nodes = [n for n in gas_graph.nodes() if n.startswith("(00") or n.startswith("(011")]

        for input_data_node in input_data_nodes:
            if "Natural Earth shape files" in input_data_node:
                continue

            if "PRIMAP" in input_data_node:
                continue

            if "HadCRUT" in input_data_node:
                continue

            if "AGAGE" in input_data_node and "download" in input_data_node:
                if "gc-ms_monthly" in input_data_node:
                    dependency = "AGAGE gc-ms"

                elif "gc-ms-medusa_monthly" in input_data_node:
                    dependency = "AGAGE gc-ms-medusa"

                elif "gc-md_monthly" in input_data_node:
                    dependency = "AGAGE gc-md"

                else:
                    raise NotImplementedError(input_data_node)

            elif "GAGE" in input_data_node and "download" in input_data_node:
                dependency = "AGAGE GAGE"

            elif "ALE" in input_data_node and "download" in input_data_node:
                dependency = "AGAGE ALE"

            elif "NOAA" in input_data_node and "download" in input_data_node:
                if "hats" in input_data_node:
                    dependency = "NOAA HATS"

                elif "surface-flask" in input_data_node:
                    dependency = "NOAA surface flask"

                elif "in-situ" in input_data_node:
                    dependency = "NOAA in-situ"

                else:
                    raise NotImplementedError(input_data_node)

            elif "Velders" in input_data_node:
                dependency = "Velders et al., 2022"

            elif "Western" in input_data_node and "download" in input_data_node:
                dependency = "Western et al., 2024"

            elif "Droste" in input_data_node and "download" in input_data_node:
                dependency = "Droste et al., 2020"

            elif "WMO 2022 ozone assessment" in input_data_node:
                dependency = "WMO 2022 ozone assessment Ch. 7"

            elif "Law Dome" in input_data_node and "download" in input_data_node:
                dependency = "Law Dome"

            elif "EPICA" in input_data_node and "download" in input_data_node:
                dependency = "EPICA"

            elif "NEEM" in input_data_node and "download" in input_data_node:
                dependency = "NEEM"

            elif "process" in input_data_node:
                if any(
                    v in input_data_node
                    for v in (
                        "NOAA",
                        "AGAGE",
                        "GAGE",
                        "ALE",
                        "EPICA",
                        "NEEM",
                        "Law Dome",
                        "Western",
                        "Droste",
                    )
                ):
                    # Sources which have a download step then a process step
                    continue
                else:
                    raise NotImplementedError(input_data_node)

            else:
                raise NotImplementedError(input_data_node)

            di = DependencyInfoSource(
                gas=gas,
                source=dependency,
                licence=source_info[dependency].licence,
                reference=source_info[dependency].reference,
                doi=source_info[dependency].doi,
            )
            dependency_info_l.append(di)

    return DependencyInfo(tuple(dependency_info_l))


def main(  # noqa: PLR0913
    force_dot_generation: Annotated[
        bool, typer.Option(help="Should we force the dot files to be re-generated?")
    ] = False,
    out_file: Annotated[Path, typer.Option(help="Path in which to write the dependency table)")] = Path(
        "dependencies-table.md"
    ),
    out_file_by_gas_json: Annotated[
        Path, typer.Option(help="Path in which to write the dependencies, grouped by gas")
    ] = Path("dependencies-by-gas.json"),
    out_file_csv: Annotated[
        Path, typer.Option(help="Path in which to write the dependencies as a csv")
    ] = Path("dependencies-table.csv"),
    config_file: Annotated[
        str, typer.Option(help="Config file to use when determining the dependencies")
    ] = "v0.4.0-config.yaml",
    expected_number_of_writing_tasks: Annotated[
        int, typer.Option(help="Expected number of file writing tasks")
    ] = 43,
) -> None:
    """
    Create the dependency table
    """
    pixi = shutil.which("pixi")

    doit_list_all = get_doit_list_all(
        pixi=pixi,
        config_file=config_file,
    )

    input4mips_writing_tasks = extract_input4mips_writing_tasks(doit_list_all)
    if len(input4mips_writing_tasks) != expected_number_of_writing_tasks:
        raise AssertionError(len(input4mips_writing_tasks))

    dot_files = get_dependency_dot_files(
        pixi=pixi, input4mips_writing_tasks=input4mips_writing_tasks, force_generation=force_dot_generation
    )

    dependency_info = extract_dependencies(dot_files)
    with open(out_file_by_gas_json, "w") as fh:
        json.dump(dependency_info.by_gas_serialised(), fh, indent=2)

    pd.DataFrame([asdict(source) for source in dependency_info.sources]).sort_values(
        ["gas", "source"]
    ).to_csv(out_file_csv, index=False)

    md_summary_by_source_d = defaultdict(list)
    for source in dependency_info.sources:
        md_summary_by_source_d[source.source].append(source.gas)

    md_summary_by_gas_d = defaultdict(list)
    for source in dependency_info.sources:
        md_summary_by_gas_d[source.gas].append(source.source)

    md_summary_by_source_l = []
    sorted_by_n_gases = sorted(
        md_summary_by_source_d.items(), key=lambda x: f"{len(x[1]):02d}-{x[0]}", reverse=True
    )
    for source, gases in sorted_by_n_gases:
        line = f"**{source}** - {len(gases)}: {', '.join(sorted(set(gases)))}"
        md_summary_by_source_l.append(line)

    gas_order = ["co2", "ch4", "n2o", "cfc12", "cfc11", "hfc134a", "sf6"]
    for gas in sorted(md_summary_by_gas_d):
        if gas not in gas_order:
            gas_order.append(gas)

    md_summary_by_gas_l = []
    for gas in gas_order:
        sources = md_summary_by_gas_d[gas]
        line = f"**{gas}**: {', '.join(sorted(set(sources)))}"
        md_summary_by_gas_l.append(line)

    with open(out_file, "w") as fh:
        fh.write("# Summary by source\n")
        fh.write("\n")
        source_txt = "\n".join(md_summary_by_source_l)
        fh.write(f"{source_txt}\n")

        fh.write("\n")

        fh.write("# Summary by gas\n")
        fh.write("\n")
        gas_txt = "\n".join(md_summary_by_gas_l)
        fh.write(f"{gas_txt}\n")


if __name__ == "__main__":
    typer.run(main)

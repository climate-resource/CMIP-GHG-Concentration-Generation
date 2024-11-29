"""
Make a table of dependencies
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pygraphviz
import tqdm


def main(force_dot_generation: bool = False, out_file: Path = Path("dependencies-table.md")) -> None:  # noqa: PLR0912, PLR0915
    """
    Create the dependency table
    """
    pixi = shutil.which("pixi")
    doit_list_all = subprocess.check_output(
        [pixi, "run", "-e", "all-dev", "doit", "list", "--all", "--quiet"],  # noqa: S603
        env={"DOIT_CONFIGURATION_FILE": "v0.4.0-config.yaml"},
    )

    input4mips_writing_tasks = []
    for task in doit_list_all.decode().splitlines():
        task_clean = task.strip()
        if "write input4MIPs - write all files:" in task_clean and "eq" not in task_clean:
            input4mips_writing_tasks.append(task_clean)

    expected_number_of_writing_tasks = 43
    if len(input4mips_writing_tasks) != expected_number_of_writing_tasks:
        raise AssertionError

    dot_files = {}
    for writing_task in tqdm.tqdm(input4mips_writing_tasks, desc="Creating dot files"):
        gas = writing_task.split(":")[-1]

        out_dot = Path(f"{gas}-dependencies.dot")
        if force_dot_generation or not out_dot.exists():
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

    all_gas_deps = {}
    for gas, dot_file in tqdm.tqdm(dot_files.items(), desc="Extracting dependencies from dot files"):
        gas_graph = pygraphviz.AGraph(dot_file, strict=False, directed=True)
        input_data_nodes = [n for n in gas_graph.nodes() if n.startswith("(00")]

        gas_deps = []
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

            elif "WMO 2022 ozone assessment" in input_data_node:
                dependency = "WMO 2022 ozone assessment"

            elif "Law Dome" in input_data_node and "download" in input_data_node:
                dependency = "Law Dome"

            elif "EPICA" in input_data_node and "download" in input_data_node:
                dependency = "EPICA"

            elif "NEEM" in input_data_node and "download" in input_data_node:
                dependency = "NEEM"

            elif "process" in input_data_node:
                if any(
                    v in input_data_node
                    for v in ("NOAA", "AGAGE", "GAGE", "ALE", "EPICA", "NEEM", "Law Dome", "Western")
                ):
                    # Sources which have a download step then a process step
                    continue
                else:
                    raise NotImplementedError(input_data_node)

            else:
                raise NotImplementedError(input_data_node)

            gas_deps.append(dependency)

        all_gas_deps[gas] = gas_deps

    pretty_deps = {}
    for gas in sorted(all_gas_deps):
        deps = all_gas_deps[gas]
        non_agage_non_noaa = [d for d in deps if "AGAGE" not in d and "NOAA" not in d]

        pd = []
        if non_agage_non_noaa:
            pd.append(", ".join(non_agage_non_noaa))

        noaa = [d for d in deps if "NOAA" in d]
        if noaa:
            noaa_d = [d.replace("NOAA ", "") for d in noaa]
            pd.append(f"NOAA ({', '.join(noaa_d)})")

        agage = [d for d in deps if "AGAGE" in d]
        if agage:
            agage_d = [d.replace("AGAGE ", "") for d in agage]
            pd.append(f"AGAGE ({', '.join(agage_d)})")

        pretty_deps[gas] = pd

    with open(out_file, "w") as fh:
        dep_sums = {
            "AGAGE": [],
            "NOAA": [],
            "WMO": [],
            "Velders": [],
            "Western": [],
            "Law Dome": [],
            "EPICA": [],
            "NEEM": [],
        }
        for gas, deps in pretty_deps.items():
            for id in dep_sums:
                if any(id in v for v in deps):
                    dep_sums[id].append(gas)

        for id, gases in dep_sums.items():
            fh.write(f"**{id}**: {len(gases)} ({', '.join(gases)})")
            fh.write("\n")

        fh.write("\n\n")

        for gas in sorted(pretty_deps):
            fh.write(f"**{gas}**: ")
            fh.write(". ".join(pretty_deps[gas]))
            fh.write("\n")


if __name__ == "__main__":
    main()

"""
Find the data available from AGAGE without so much clicking
"""
import urllib.request
from pathlib import Path

import pooch
from bs4 import BeautifulSoup
from pydoit_nb.config_tools import URLSource

from local.config.retrieve_and_extract_agage import RetrieveExtractAGAGEDataConfig

start_url = "https://agage2.eas.gatech.edu/data_archive"
soup_base = BeautifulSoup(
    urllib.request.urlopen(start_url).read(),  # noqa: S310
    "html.parser",
)


def find_agage_gases(base_url, time_frequency="monthly"):  # noqa: PLR0915
    """
    Find gases measured in the AGAGE network
    """
    print("============")
    print(base_url)
    soup_base = BeautifulSoup(
        urllib.request.urlopen(base_url).read(),  # noqa: S310
        "html.parser",
    )
    with open("agage-gases.txt", "w") as fh:
        for link in soup_base.find_all("a"):
            all_gases = []
            instrument = link.get("href")
            if instrument.endswith("/") and not instrument.startswith("/"):
                print(instrument)
                instrument_time_url = f"{base_url}{instrument}{time_frequency}/"
                soup_instrument_time = BeautifulSoup(
                    urllib.request.urlopen(instrument_time_url).read(),  # noqa: S310
                    "html.parser",
                )

                download_urls = {}
                for link_instrument in soup_instrument_time.find_all("a"):
                    loc = link_instrument.get("href")
                    if loc.endswith("/") and not loc.startswith("/"):
                        url_loc = f"{instrument_time_url}{loc}"
                        print(f"{url_loc=}")
                        soup_loc = BeautifulSoup(
                            urllib.request.urlopen(url_loc).read(),  # noqa: S310
                            "html.parser",
                        )
                        soup_loc_file_formats = [
                            link.get("href")
                            for link in soup_loc.find_all("a")
                            if link.get("href").endswith("/")
                            and not link.get("href").startswith("/")
                        ]
                        if soup_loc_file_formats != ["ascii/"]:
                            raise AssertionError(  # noqa: TRY003
                                "Unexpected data formats"
                            )

                        for file_format in soup_loc_file_formats:
                            url_loc_file_format = f"{url_loc}{file_format}"
                            print(f"{url_loc_file_format=}")

                            uh = url_loc_file_format
                            tmp = urllib.request.urlopen(uh)  # noqa: S310
                            soup_loc_file_format = BeautifulSoup(
                                tmp.read(),
                                "html.parser",
                            )
                            soup_loc_gas_format_data_files = [
                                link.get("href")
                                for link in soup_loc_file_format.find_all("a")
                                if link.get("href").endswith(".txt")
                            ]
                            gases = [
                                f.split("_")[2] for f in soup_loc_gas_format_data_files
                            ]

                            for file in soup_loc_gas_format_data_files:
                                gas = file.split("_")[2]
                                if "h2" in gas and "pdd" in gas:
                                    # Not sure what this file is
                                    continue

                                url = f"{url_loc_file_format}{file}"
                                tmp_file = pooch.retrieve(
                                    url=url,
                                    known_hash=None,
                                    progressbar=True,
                                )
                                if isinstance(tmp_file, list):
                                    raise NotImplementedError(
                                        "More than one file: tmp_file"
                                    )

                                hash = pooch.file_hash(tmp_file)
                                if gas not in download_urls:
                                    download_urls[gas] = []
                                download_urls[gas].append(
                                    URLSource(url=url, known_hash=hash)
                                )

                            print("--------------")
                            print(f"{instrument=}")
                            print(f"{loc=}")
                            print(f"{file_format=}")
                            print(f"{gases=}")
                            all_gases.extend(gases)

                for gas, urls in download_urls.items():
                    # TODO: introduce gas_clean here
                    step_config_id = (
                        f"{gas}_{instrument.replace('/', '')}_{time_frequency}"
                    )
                    raw_dir = Path("data") / "raw" / "agage" / "agage"
                    step_conf = RetrieveExtractAGAGEDataConfig(
                        step_config_id=step_config_id,
                        gas=gas,
                        instrument=instrument.replace("/", ""),
                        time_frequency=time_frequency,
                        download_urls=urls,
                        raw_dir=raw_dir,
                        download_complete_file=raw_dir / f"{step_config_id}.complete",
                        generate_hashes=False,
                    )
                    print(f"{step_conf},")
                    fh.write(f"{step_conf},\n")

                print(f"AGAGE {instrument} available gases")
                print(f"{set(all_gases)}")
                print("=----=-=")


for link in soup_base.find_all("a"):
    experiment = link.get("href")

    if (
        not any(v in experiment for v in ("data_figures", "global_mean", "readme"))
        and experiment.endswith("/")
        and not experiment.startswith("/")
    ):
        print(experiment)
        experiment_url = f"{start_url}/{experiment}"
        if "agage" in experiment:
            find_agage_gases(experiment_url)

"""
Find the data available from AGAGE without so much clicking
"""
import urllib.request

from bs4 import BeautifulSoup

start_url = "https://agage2.eas.gatech.edu/data_archive"
soup_base = BeautifulSoup(
    urllib.request.urlopen(start_url).read(),  # noqa: S310
    "html.parser",
)


def find_agage_gases(base_url, time_frequency="monthly"):
    """
    Find gases measured in the AGAGE network
    """
    print("============")
    print(base_url)
    soup_base = BeautifulSoup(
        urllib.request.urlopen(base_url).read(),  # noqa: S310
        "html.parser",
    )

    all_gases = []
    for link in soup_base.find_all("a"):
        instrument = link.get("href")
        if instrument.endswith("/") and not instrument.startswith("/"):
            print(instrument)
            instrument_time_url = f"{base_url}{instrument}{time_frequency}/"
            soup_instrument_time = BeautifulSoup(
                urllib.request.urlopen(instrument_time_url).read(),  # noqa: S310
                "html.parser",
            )

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
                        raise AssertionError("Unexpected data formats")  # noqa: TRY003

                    for file_format in soup_loc_file_formats:
                        url_loc_file_format = f"{url_loc}{file_format}"
                        print(f"{url_loc_file_format=}")

                        tmp = urllib.request.urlopen(url_loc_file_format)  # noqa: S310
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
                        print("--------------")
                        print(f"{instrument=}")
                        print(f"{loc=}")
                        print(f"{file_format=}")
                        print(f"{gases=}")
                        print(soup_loc_gas_format_data_files)
                        print()
                        print()

                        all_gases.extend(gases)

    print("AGAGE available gases")
    print(f"{set(all_gases)}")


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

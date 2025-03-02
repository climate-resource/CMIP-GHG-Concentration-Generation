# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## CMIP GHG Concentration Generation 1.0.0 (2025-03-02)

### ⚠️ Breaking Changes

- Switched to more conventional names for the output files.

  The affected gases are:

  - c2f6: previously pfc116
  - c3f8: previously pfc218
  - c4f10: previously pfc3110
  - c5f12: previously pfc4112
  - c6f14: previously pfc5114
  - c7f16: previously pfc6116
  - c8f18: previously pfc7118
  - cc4f8: previously pfc318
  - ch3ccl3: previously hcc140a

  ([#90](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/90))

### 🆕 Features

- Added hfc23 data from [Adam et al., 2024](https://doi.org/10.1038/s43247-024-01946-y) ([#84](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/84))
- Added much more detailed reference tracking to the output files. ([#93](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/93))

### 🎉 Improvements

- Switch to using data from Menking et al., 2025 (in prep.) for N2O ice cores. ([#85](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/85))
- Added upload to Zenodo and fixed up the metadata included in the output files to use the Zenodo DOI. ([#86](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/86))
- Update to the latest version of the AGAGE data ([#87](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/87))
- Harmonise ice core and flask data when crunching CH4 data. ([#88](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/88))
- Use Scripps data and updated ice core when crunching CO2 data. Harmonise both before use. ([#89](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/89))
- - Added funding information to the output files.
  - Upgraded to the latest AGAGE data.
  - Upgraded to the latest NOAA HATS data.
  - Added more detail in the `zenodo.json` description.

  ([#90](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/90))
- Use Trudinger et al. (2016) for historical values of CF4, C2F6 and C3F8 ([#92](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/92))

### 🐛 Bug Fixes

- Updated the expected hash for HadCRUT5 ([#83](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/83))
- Fixed up the instructions in the output bundle's README. ([#90](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/90))
- - Include Menking et al. (2025, in-prep) data that isn't missing an 1850 value
  - Cleaned up pre-industrial values of multiple gases

  ([#91](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/91))
- Fixed incorrect reference to publication in preparation. ([#92](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/92))

### 🔧 Trivial/Internal Changes

- [#86](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/86), [#90](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/90)


## CMIP GHG Concentration Generation 0.3.0 (2024-12-05)

This version of the software was used to produce v0.4.0 of the GHG concentrations.
As far as we are aware, this version fixes all major bugs in v0.3.0 of the GHG concentrations.
The difference in ERF between the CMIP6 data and v0.4.0 is less than 0.02 W / m^2.
In most cases, it is much less than this.
For the full list of updates since v0.3.0 (and before), see below.

For further analysis of the changes,
see https://github.com/climate-resource/CMIP6-vs-CMIP7-GHG-Concentrations.

### ⚠️ Breaking Changes

- Updated to produce the v0.3.0 concentrations.
  Re-wrote a bunch of stuff in the process,
  including no longer producing the 0.5 degree resolution data.
  This can be added in future again if there is demand. ([#57](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/57))
- Reset the CI. As a result, all sorts of results could have changed. ([#64](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/64))
- Re-wrote the mean-preserving interpolation module.
  The behaviour is now completely different,
  but also much easier to control and more performant. ([#70](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/70))

### 🆕 Features

- Added reading of data from the NOAA network

  Supports reading both flask and in-situ data for CO2, CH4, N2O and SF6 ([#7](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/7))
- Added auto-generation of a Zenodo DOI into the production config via `scripts/write-run-config.py`. ([#65](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/65))
- The new `local.mean_preserving_interpolation` module.
  This is tested in isolation by the tests in `tests`
  and could be re-used in other applications. ([#70](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/70))
- Added a first list of references to the output file.
  Future work (https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/issues/62)
  will make these even more refined as needed,
  but this already provides better visibility for observations people than we had previously. ([#78](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/78))
- Use data from [Droste et al., 2020](https://doi.org/10.5194/acp-20-4787-2020)
  for the remaining PFCs (and c-C4F8, which previously used AGAGE),
  except c8f18, for which we still use CMIP6
  (due to a lack of papers since [Ivy et al., 2012](https://doi.org/10.5194/acp-12-7635-2012)). ([#81](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/81))

### 🎉 Improvements

- Added a DOI into the generated files. ([#65](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/65))
- Added a check to ensure that no negative values appear in the outputs. ([#71](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/71))
- - Switched to writing the output data as float, rather than double, to save space and avoid the illusion of precision.
  - Switched to writing files in time chunks that better suit the access pattern of CMIP (files are now written separately for 1-999, 1000-1749 and 1750- rather than all in one).

  ([#74](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/74))
- - Used data from [Chapter 7 of the WMO 2022 ozone assessment](https://csl.noaa.gov/assessments/ozone/2022/downloads/),
    [Velders et al., 2022](https://doi.org/10.5194/acp-22-6087-2022)
    and [Western et al., 2024](https://www.nature.com/articles/s41558-024-02038-7)
    for global-means of ODSs and HFCs where available.
  - Improved the interpolation for SF6-like gases to be smoother and more reliable
  - Used AR6 values for radiative efficiencies and equivalent calculations

  ([#76](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/76))
- Updated to a newer version of Velders et al. (2022) ([#77](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/77))
- Added CMIP IPO/ESA funding acknowledgement. ([#78](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/78))

### 🐛 Bug Fixes

- Removed cfc12 from the list of species which contribute to cfc11eq. ([#68](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/68))
- Fixed a bug introduced in #74 (the wrong variable name was being used for setting the data precision). ([#75](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/75))
- Fixed handling of Western data to account for the fact that the raw data is January-centred. ([#79](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/79))

### 🔧 Trivial/Internal Changes

- [#7](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/7), [#65](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/65), [#66](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/66), [#67](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/67), [#70](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/pulls/70)

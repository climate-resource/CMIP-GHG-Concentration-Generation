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

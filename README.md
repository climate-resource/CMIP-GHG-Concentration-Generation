# CMIP-GHG-Concentration-Generation

Generation of historical concentrations for CMIP (and other) experiments.
We use observations from multiple data sets to create a composite data set
which can be used for running models that participate in the Coupled Model Intercomparison Project (CMIP)
and similar experiments.
The key characteristics of the outputs are:

- a consistent data format which follows CMIP (specifically input4MIPs) conventions
- a consistent time span and spatial coverage for all 43 greenhouse gases of interest
  (plus composite products for baskets of gases like HFCs, as these are required by some models)
- an approach for combining information from multiple observational networks into one composite data product
- an approach for deriving the seasonality (variation in concentrations over a year)
  and latitudinal gradients (variation in concentrations from North Pole to South Pole)
  of the concentrations which can be used to extend these effects (seasonality and latitudinal gradients)
  both back in time and forward in time.
  The approach needs to be relatively simple
  because no high-detail data is available to extend these effects back in time
  and the approach must be consistent with the outputs of running a simple climate model to extend these effects forward in time
  (it is typically too expensive to do this forward extension using a more complex model)
  - this is the trickiest and most scientifically interesting part of the work

## Status

Status: Stable.

We have now produced greenhouse gas concentrations for the CMIP7 AR7 fast track.
Even though forcings research is going,
this repository is now considered stable.
See the issues for envisaged updates
and `changelog` for the updates that have been made.

## Installation

For all of our dependency management we use [pixi](https://pixi.sh/latest/).
Assuming you have pixi, you can then install the project using

```sh
pixi install
```

## Creating the outputs

To create all the outputs, after having installed the project, simply run

```sh
pixi run doit run --verbosity=2
```

## Development

### Installation

As for installation above, except now you will want to install the development
dependencies too, set up pre-commit etc. so the `Makefile` is your friend.
Simply run

```sh
make virtual-environment
```

You can then create the outputs as above, although the development outputs
are probably more helpful. For these, the `Makefile` is once again your friend
so just run the following, which sets the required environment variables too

```sh
make all-dev
```

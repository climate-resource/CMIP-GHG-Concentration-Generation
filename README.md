# CMIP-GHG-Concentration-Generation

Generation of historical concentrations for CMIP (and other) experiments

Status: Prototype, probably best to come back later

## Installation

For all of our dependency management we use [poetry](https://python-poetry.org/).
Assuming you have poetry, you can then install the project using

```sh
poetry install --only main
```

## Creating the outputs

To create all the outputs, after having installed the project, simply run

```sh
poetry run doit run --verbosity=2
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

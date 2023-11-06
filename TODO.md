# TODO

- add example using `generate_directory_checklist`

- think about whether using taipy would be a better way to do all this

- all paths get root_dir_output / run_id pre-pended
- things that aren't run by notebooks don't go in config (e.g. copy source stuff)

- fix up use of define vs. frozen
- fix up use of Path vs. PathLike
- unify naming :output_root_dir vs root_dir_output vs notebook_output_dir etc.
- CI
    - run all notebooks
    - check output against some regression output
    - add create bundle to CI (that should be end of pipeline already so should be automatic)
        - obviously can't/don't want to check upload
- Use pydoit-nb etc.
- Makefile
    - all
    - licence check (but this should be optional in copier)
    - venv
- move upload to zenodo functionality into openscm-zenodo (job for D2)
- think about pre-commit properly
- check Python copier repo for any other CI stuff
- add some advice about always maintaining a test pipeline, even if it means you end up maintaining two config files which can be a bit fiddly
- add advice in development docs about pinning dependencies as this is an application repo (noting also that lock file should save you mostly anyway)
- basic docs (can do this all in README as this is an application not a library)


set seed –– draw data no covariance ––– plot draws against each other –– combine data into clean table –– plot all against each other –– create bundle
          \                           |                               |
            draw data with covariance |
                                                                      |
draw data with constraint and plot ––––––––––––––––––––––––––––––––––––

Notes:

– put notebook number at start of saved data file, makes it easier to find where it came from later
– make clean data and plots in same notebook first (fast iteration), but make sure you go back and split in follow up step (fast iteration for plots)
- config gets hydrated so put sensitive information in environment variables
- config class plus cattrs is way of getting around awkwardness of passing things in memory (pass everything via serialising to disk which makes things a bit slower and more IO intense, but also way simpler, put note that if your setup does heaps of IO for the config, this solution may not work for you/you may need to work out how to pull things out of the config into somewhere else)
- `get_config_bundle` is where you spend most of your config time headaches. It defines how things interact, what is coupled to what, what can override what, how to fill in placeholders etc. This is the part where we have the least rules/guidance, it is really the wild west and we don't know patterns will work best (note for me/Jared: I think the CSIRO pattern was more complicated than it needed to be in the end, particularly the use of multiple yaml files for the pre-hydration stage)

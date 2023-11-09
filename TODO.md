# TODO

- actual docs
- add zenodo upload step (no point having it as separate script really)
- add example using `generate_directory_checklist`

- think about whether using taipy would be a better way to do all this

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
- basic docs (can do this all in README as this is an application not a library) but will need something more sophisticated for a copier repo


Notes about development like this in general to put in e.g. tips and tricks:

– put notebook number at start of saved data file, makes it easier to find where it came from later
– make clean data and plots in same notebook first (fast iteration), but make sure you go back and split in follow up step (fast iteration for plots)
- config gets hydrated and written to disk so put sensitive information in environment variables
- config class plus cattrs is way of getting around awkwardness of passing things in memory (pass everything via serialising to disk which makes things a bit slower and more IO intense, but also way simpler, put note that if your setup does heaps of IO for the config, this solution may not work for you/you may need to work out how to pull things out of the config into somewhere else)

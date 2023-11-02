# TODO

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

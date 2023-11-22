# TODO

- UnconfiguredNotebookStep becomes a notebook-based step in the workflow
    - Add somewhere in pydoit-nb docs a distinction between a task (single notebook) and a step (one or more notebooks that makes up some conceptual group/high-level step in the overall workflow)
- Rename
    - NotebookStep -> NotebookTask i.e. a doit task which is based on a notebook
    - folder notebook_steps can stay as is
    - `get_notebook_branch_tasks` can become a method of `UnconfiguredNotebookStep`
        - then we can delete (probably) `tasks_notebooks`
- Fix up docs as I go along


- actual docs
- add zenodo upload step (no point having it as separate script really)
- add example using `generate_directory_checklist`

- think about whether using taipy would be a better way to do all this
- think about whether hydra would be a good tool to use/point to

- split out pydoit-nb into separate package then use here
- Makefile
    - licence check (but this should be optional in copier)
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

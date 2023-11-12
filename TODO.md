# TODO

## Concs

- download data using e.g. pooch (make sure it goes into the bundle so we don't have to rely on data without DOIs)
- plot raw data over the top of each other (eventually Malte style/interactive)
- do join of data into single yearly timeseries
    - actual algorithm is much smarter than this, but for now just take annual-mean then mean across data sets and interpolate linearly to fill any gaps
- mean-preserving interpolation down to monthly values
- add seasonal and latitudinal gradient (do just something for now, can make it more complicated in future)
- write out in correct format
- then iterate

### URLs

- basic hover setup: https://stackoverflow.com/questions/67904185/how-to-show-values-on-hover-for-multiple-line-graphs-bokeh
- docs on hovering: https://docs.bokeh.org/en/2.4.1/docs/user_guide/tools.html#basic-tooltips
- docs on toggling lines: https://docs.bokeh.org/en/2.4.1/docs/user_guide/interaction/legends.html#hiding-glyphs
- tutorials on bokeh in notebooks: https://github.com/bokeh/bokeh-notebooks
- live updating (if we want it): https://github.com/bokeh/bokeh/blob/3.3.1/examples/output/jupyter/push_notebook/Continuous%20Updating.ipynb
- events (need to understand this for drilldown I think): https://docs.bokeh.org/en/latest/docs/reference/events.html
    - also this: https://docs.bokeh.org/en/2.4.1/docs/user_guide/server.html#callbacks-and-events
    - may also need this if we want it to be easy-ish: https://docs.bokeh.org/en/2.4.1/docs/user_guide/server.html#userguide-server-applications
    - this suggests a route: https://stackoverflow.com/questions/72756899/how-to-add-onclick-event-in-bokeh#comment128641340_72756899
    - may also be relevant: https://stackoverflow.com/questions/32418045/running-python-code-by-clicking-a-button-in-bokeh?rq=4
    - probably also google 'bokeh update plot on click'

### Data sources

- N2O budget: https://auburn.app.box.com/s/7fxesyrfzpj2k8h9cnu3vxbb014xj3sq/
    - BUT, paper basically says use CSIRO, AGAGE and NOAA: https://www.nature.com/articles/s41586-020-2780-0#Sec2
- Global methane budget: https://www.icos-cp.eu/GCP-CH4/2019
    - data file then points out to relevant observational networks
    - paper here: https://essd.copernicus.org/articles/12/1561/2020/
- Law dome ice core
    - https://data.csiro.au/collection/csiro:37077
- Scripps stations
    - https://scrippsco2.ucsd.edu/data/atmospheric_co2/alt.html
- Scripps merged (not sure whether to use raw or not)
    - https://scrippsco2.ucsd.edu/data/atmospheric_co2/icecore_merged_products.html
- Scipps other links
    - https://keelingcurve.ucsd.edu/permissions-and-data-sources/

## Repo

- actual docs
- add zenodo upload step (no point having it as separate script really)
- if you want to fix type hints, need a minimum example as doing it in live project is too hard

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

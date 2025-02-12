# TODO

File for keeping track of to-do's.
As the to-do's become concrete, take them out and turn them into [issues](https://github.com/climate-resource/CMIP-GHG-Concentration-Generation/issues).

- rename the "**02_***_global-mean-latitudinal-gradient-seasonality.py" notebooks to
  "**02_***_obs-network-global-mean-latitudinal-gradient-seasonality.py"
  given that is what they actually do.

- pre-industrial handling: Section 3.4 helps, but will also have to look at the code to work out when the spline to pre-industrial concs begins.

    - Recipe appears to be:

        - use observations
        - prescribe in global-mean from other sources (other sources specified in `switch globalpara.parameter` cases in `main_script_histGHG*`)
        - interpolate to fill gaps
        - extrapolate constant (backwards) or linear (forwards) to fill other gaps

    - (Effectively) non-zero PI concs:

        - CH2Cl2
        - CH3Br
        - CH3Cl
        - CHCl3
        - CF4

- Might need to use this extend poleward stuff, hopefully not

- use better data source to extend CO2 and N2O from the start of Law Dome back to year 1

- check polluted data with AGAGE authors.
    - I think I'm using monthly means i.e. only baseline data, but should check
- check with NOAA people why some stations only have event data, when they pass quality control and seem to have enough data to do a fit
    - ask Steve Montzka
- I don't think I'm using any pollution data as I use monthly means everywhere

- reach out to NOAA and AGAGE authors to ask them what the right choice is in terms of using monthly vs. event data

- process NOAA and AGAGE data grids

  - i.e. implement algorithm
  - tricky because we'll have data going everywhere at the same time as we try and reimplement M17

- new release of input4mips-validation so we no longer need the local install

- dealing with scales of gases is going to require sitting down and thinking for a minute

- dealing with all the data sources is going to be a grind

## Concs

To actually tackle this:

- start with CO2
- then CH4
- then N2O
- then HFCs
- then PFCs
- then CFCs
- then everything else

Use this to prioritise list of to-do's to reproduce M17 below.

While doing this:

- have to be super careful with notebook steps. Need to keep them very split/at the level of steps in Figure 1 so they can be combined with flexibility
  - going to be lots of notebook steps and config...

### CO2 steps

Inspired by the next section

- work out which data sources to use
  - we got quite far in handling AGAGE in concentrations/2.1.1 Step 1 - aggregating raw station data (AGAGE).ipynb
  - we got quite far in handling NOAA here: concentrations/2.1.1 Step 1 - aggregating raw station data (NOAA).ipynb
  - (for when we get to CH4) also get CH4 ice core data from EPICA Dronning Maud Land ice core
  - (for when we get to CH4) also get data from NEEM Greenland
  - (for when we get to other gases) copy the approach from Forster et al. (2023)
    - "Some of the more minor halogenated gases are not part of the NOAA or AGAGE operational network
      and are currently only reported in literature sources until 2019 or possibly 2015
      (Droste et al., 2020; Laube et al., 2014; Schoenenberger et al., 2015; Simmonds et al., 2017; Vollmer et al., 2018)."
  - see notes in top of notebook 0010
    - https://essd.copernicus.org/articles/15/2295/2023/essd-15-2295-2023.pdf uses both GGGRN (Forster et al. (2023) calls it NOAA Global Monitoring Laboratory (GML)) and AGAGE. It's not clear to me that these two sources are independent, will have to do some digging
      - have to email authors and also ask Malte to clarify differences between sources
  - consider whether we should use CSIRO's spline (in the Splines fits sheet of the Law Dome data) rather than inventing our own (data providers seem to often have their own spline, not sure if better to have consistency or use data providers' interpolated view)
- bin raw station data (15 degree latitudinal bins, 60 degree longitudinal bins)
- plot raw data over the top of each other (eventually Malte style/interactive)
  - somewhere in the notebooks there are already prototypes of this
- filter raw station data
  - some thoughts in #29
- average with equal station weight
  - i.e. take average over the month for the station first, then average over stations
  - this notebook got some way to showing tricky things for moving sampling: concentrations/2.1.2 Step 2–4 - binning and spatial interpolation.ipynb
- average all observations over the month to get a monthly mean (need to check this with Malte, seems odd to average measurements at different stages in the month but maybe error is negligible)
  - treak flask and in situ measurements as separate stations
- spatially interpolate any missing values using linear 2D interpolation
  - need to work out what algorithm is used for this (looking at Matlab code, it is https://de.mathworks.com/help/matlab/ref/griddata.html)
- at this point, you have a complete, interpolated, lat-long, monthly field over the instrumental period
- calculate average over longitude (i.e. end up with lat, monthly field)
- latitudinal gradient
- global-mean
- seasonality

## Meinshausen et al., 2017 reproduction

- mean-preserving interpolation down to monthly values

- seasonal gradient

- latitudinal gradient

- mean-preserving interpolation down to finer latitudinal grid

- check format etc. with Paul

- Original source code

  - https://gitlab.com/magicc/CopyForDeletion_CMIP6GHGconcentrations
  - key driver here: https://gitlab.com/magicc/CopyForDeletion_CMIP6GHGconcentrations/-/blob/master/main_script_histGHG_generator.m?ref_type=heads
  - full set of stuff here: https://drive.google.com/drive/u/1/folders/19I8VMDAwG1JJ_K_Wzw05qrv9ENpFhRD8

- Some documentation stuff: see #34

- Looking at Figure 1

  - if you only have a global-mean estimate, super unclear to me what happens
    - extrapolate somehow and then assume zero latitudinal gradient or seasonality?
  - otherwise
    - bin raw station data (15 degree latitudinal bins, 60 degree longitudinal bins)
    - filter raw station data
      - some thoughts in #29
    - average with equal station weight
      - i.e. take average over the month for the station first, then average over stations
    - average all observations over the month to get a monthly mean (need to check this with Malte, seems odd to average measurements at different stages in the month but maybe error is negligible)
      - treak flask and in situ measurements as separate stations
    - spatially interpolate any missing values using linear 2D interpolation
      - need to work out what algorithm is used for this (looking at Matlab code, it is https://de.mathworks.com/help/matlab/ref/griddata.html)
    - at this point, you have a complete, interpolated, lat-long, monthly field over the instrumental period
      - if you have any missing time points at this stage (within the series, not back in time, that comes later), have to think about how to handle them...
    - calculate average over longitude (i.e. end up with lat, monthly field)
    - branch
      - latitudinal gradient (Section 3 of M17 has suggestions for improvements, Section 5.6 also has important point about handling time-varying data coverage)
        - calculate annual-average deviations from smoothed annual-mean at each latitude
          - unclear to me why it would be smoothed annual-mean
        - now you have a (y, l) field of annual latitudinal deviations
        - do principle component analysis on this (where the PCA is done over the year dimension i.e. your principle components should have variation with latitude only)
        - get principle component scores by projecting principle components back onto original deviations
          - slide 10 onwards here best resource I could find for this https://www.ess.uci.edu/~yu/class/ess210b/lecture.5.EOF.all.pdf
          - previous implementation: https://gitlab.com/magicc/CopyForDeletion_CMIP6GHGconcentrations/-/blob/master/eof_ghg_residuals.m?ref_type=heads
        - if you have ice/firn records in both hemispheres
          - some optimisation of principal component scores to match global-mean concentrations, but I don't really understand how/what is being optimised
        - elif ice/firn record in one hemisphere
          - regress principal component score against global emissions
          - use this to extend principal component score back in time
          - then combine principal component score, principal component and observation to infer latitudinal concentrations hence global-mean concentrations
        - else
          - super unclear what goes on here, will have to speak to Malte
      - global-mean
        - calculate smoothed trendline
          - unclear to me where this is used
          - this is probably where high-frequency variations are removed (see Section 3 of M17)
        - extrapolate back/forwards in time as needed (unclear to me what this actually entails though)
      - seasonality (Section 2.1.5 describes a potentially better approach)
        - calculate monthly deviations from annual-mean at each latitude
        - now you have a (y, m, l) field of monthly deviations
        - calculate average deviation over all years
        - now you have a (m, l) field of average seasonality over the instrumental period
        - if not CO2
          - if sufficient data and clear seasonality
            - scale seasonality with global-mean concentrations for the year
            - now you have a (y, m, l) field of seasonality over the entire time period
          - else
            - assume zero seasonality
        - else
          - calculate deviations of monthly deviations from annual-mean at each latitude from average seasonality
          - now you have a (y, m, l) field of deviations from average seasonality
          - do principle component analysis on this (where the PCA is done over the year dimension i.e. your principle components should have variation with both month and latitude)
          - get principle component scores by projecting principle components back onto original deviations
            - slide 10 onwards here best resource I could find for this https://www.ess.uci.edu/~yu/class/ess210b/lecture.5.EOF.all.pdf
          - extend principal component score forward/backwards based on regression with concentrations and warming
          - principal component scores should now have dimensions (y, m)
          - apply extended principal component scores back onto principal components (M17 only used first EOF) to get change in seasonality over full time period
          - your change in seasonality should now be a (y, m, l) field
          - add on average seasonality to get full seasonality over full time period
          - now you have a (y, m, l) field of seasonality over the full time period

- There are other composite sources, use these as comparison values, e.g.

  - NOAA's global-mean product (derived based on a bunch of curve-fitting before taking the mean)
  - other ideas in #36

If/when we look at projections, need:

- gradient-preserving harmonisation (also see discussion in only office)

Off the table improvements for now:

- longitudinal gradients

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

### Data sources to follow up on

- Scripps
  - get permissions etc. from here: https://keelingcurve.ucsd.edu/permissions-and-data-sources/
  - very confusing what this is
    - why don't measurements pre-1968 appear in other networks?
    - are all their stations in e.g. NOAA, or only some?
    - should we be using Scripps' ice age core merged product and their spline?
    - need to ask Paul/Peter

- UCI network? (I emailed them on 2024-03-27 asking about data access)

- All the data sources for gases other than CO2, CH4 and N2O (that aren't AGAGAE)

- Full list of data sources is in Table 12 of M17
  - https://gmd.copernicus.org/articles/10/2057/2017/gmd-10-2057-2017.pdf

## Repo

- actual docs

- add zenodo upload step (no point having it as separate script really)

- if you want to fix type hints, need a minimum example as doing it in live project is too hard

- think about whether using taipy would be a better way to do all this

- think about whether hydra would be a good tool to use/point to

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

## Doit

- check out doit's shell integration https://pydoit.org/cmd-other.html#zsh

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Global Greenhouse Gas Reference Network
#
# Download data from the global greenhouse gas reference network (GGGRN).
#
# To-do:
#
# - move to-do's below and any learning from this notebook into 0011 and its derivatives and delete this notebook
# - just use global-mean timeseries to start with
#     - CO2
#         - data: https://gml.noaa.gov/ccgg/trends/gl_data.html, specifically https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt
#         - methods: https://gml.noaa.gov/ccgg/about/global_means.html
#     - CH4
#         - data: https://gml.noaa.gov/ccgg/trends_ch4/, specifically https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt
#         - methods: https://gml.noaa.gov/ccgg/about/global_means.html
#     - N2O
#         - data: https://gml.noaa.gov/ccgg/trends_n2o/, specifically https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.txt
#         - methods: https://gml.noaa.gov/ccgg/about/global_means.html
# - think about whether we should be using NOAA's data directly, their method directly or whether their data isn't actually what we need so just use it, but still have to do our own processing to combine etc.
# - work out how this data differs, or doesn't, from AGAGE
#     -   For example, why are there 3 timeseries in Figure 4 [here](https://www.nature.com/articles/s41586-020-2780-0#Sec2), yet CSIRO's observations are part of boath GGGRN (https://gml.noaa.gov/dv/site/?program=ccgg) and AGAGE (https://agage.mit.edu/global-network). Is there overlap yet they are somehow different products or are these products truly independent?
# - use full station data from GGGRN (and whatever other independent estimates we have) properly a la Meinshausen et al. 2017

# %% [markdown]
# ## Imports

# %%
import pooch
from pydoit_nb.checklist import generate_directory_checklist
from pydoit_nb.config_handling import get_config_for_step_id

from local.config import load_config_from_file

# %% [markdown]
# ## Define branch this notebook belongs to

# %%
step: str = "retrieve"

# %% [markdown]
# ## Parameters

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
config_file: str = "../../dev-config-absolute.yaml"  # config file
step_config_id: str = "only"  # config ID to select for this branch

# %% [markdown]
# ## Load config

# %%
config = load_config_from_file(config_file)
config_step = get_config_for_step_id(
    config=config, step=step, step_config_id=step_config_id
)

# %% [markdown]
# ## Action

# %%
for url_source in config_step.gggrn.urls_global_mean:
    pooch.retrieve(
        url=url_source.url,
        known_hash=url_source.known_hash,
        fname=url_source.url.split("/")[-1],
        path=config_step.gggrn.raw_dir,
    )

# %%
generate_directory_checklist(config_step.gggrn.raw_dir)

# %% [markdown]
# NOAA network URL, the obspack (maybe personal to me, need to check):
#
# http://gml.noaa.gov/ccgg/obspack/tmp/obspack_af7m24/obspack_co2_1_GLOBALVIEWplus_v8.0_2022-08-27.nc.tar.gz
#
# If you're not me, and you're using it, you should go and get your own link rather than using the one above. To do that, in line with fair use below, go to https://gml.noaa.gov/ccgg/obspack/
#
# ObsPack Fair Use Statement:
#
# This cooperative data product is made freely available to the scientific community and is intended to stimulate and support carbon cycle modeling studies. We rely on the ethics and integrity of the user to assure that each contributing national and university laboratory receives fair credit for their work. Fair credit will depend on the nature of the work and the requirements of the institutions involved. Your use of this data product implies an agreement to contact each contributing laboratory for data sets used to discuss the nature of the work and the appropriate level of acknowledgement. If this product is essential to the work, or if an important result or conclusion depends on this product, co-authorship may be appropriate. This should be discussed with the appropriate data providers at an early stage in the work. Contacting the data providers is not optional; if you use this data product, you must contact the applicable data providers. To help you meet your obligation, the data product includes an e-mail distribution list of all data providers. This data product must be obtained directly from the ObsPack Data Portal at https://gml.noaa.gov/ccgg/obspack/ and may not be re-distributed. In addition to the conditions of fair use as stated above, users must also include the ObsPack product citation in any publication or presentation using the product. The required citation is included in every data product and in the automated e-mail sent to the user during product download.
#
#
# Required Citation for obspack_co2_1_GLOBALVIEWplus_v8.0_2022-08-27:
#
# Kenneth N. Schuldt, John Mund, Ingrid T. Luijkx, Tuula Aalto, James B. Abshire, Ken Aikin, Arlyn Andrews, Shuji Aoki, Francesco Apadula, Bianca Baier, Peter Bakwin, Jakub Bartyzel, Gilles Bentz, Peter Bergamaschi, Andreas Beyersdorf, Tobias Biermann, Sebastien C. Biraud, Harald Boenisch, David Bowling, Gordon Brailsford, Willi A. Brand, Huilin Chen, Gao Chen, Lukasz Chmura, Shane Clark, Sites Climadat, Aurelie Colomb, Roisin Commane, Sébastien Conil, Cedric Couret, Adam Cox, Paolo Cristofanelli, Emilio Cuevas, Roger Curcoll, Bruce Daube, Kenneth Davis, Martine De Mazière, Stephan De Wekker, Julian Della Coletta, Marc Delmotte, Joshua P. DiGangi, Ed Dlugokencky, James W. Elkins, Lukas Emmenegger, Shuangxi Fang, Marc L. Fischer, Grant Forster, Arnoud Frumau, Michal Galkowski, Luciana V. Gatti, Torsten Gehrlein, Christoph Gerbig, Francois Gheusi, Emanuel Gloor, Vanessa Gomez-Trueba, Daisuke Goto, Tim Griffis, Samuel Hammer, Chad Hanson, László Haszpra, Juha Hatakka, Martin Heimann, Michal Heliasz, Daniela Heltai, Arjan Hensen, Ove Hermanssen, Eric Hintsa, Antje Hoheisel, Jutta Holst, Viktor Ivakhov, Dan Jaffe, Armin Jordan, Warren Joubert, Anna Karion, Stephan R. Kawa, Victor Kazan, Ralph Keeling, Petri Keronen, Jooil Kim, Tobias Kneuer, Pasi Kolari, Katerina Kominkova, Eric Kort, Elena Kozlova, Paul Krummel, Dagmar Kubistin, Casper Labuschagne, David H. Lam, Xin Lan, Ray Langenfelds, Olivier Laurent, Tuomas Laurila, Thomas Lauvaux, Jost Lavric, Bev Law, Olivia S. Lee, John Lee, Irene Lehner, Kari Lehtinen, Reimo Leppert, Ari Leskinen, Markus Leuenberger, Ingeborg Levin, Janne Levula, John Lin, Matthias Lindauer, Zoe Loh, Morgan Lopez, Chris R. Lunder, Toshinobu Machida, Ivan Mammarella, Giovanni Manca, Alistair Manning, Andrew Manning, Michal V. Marek, Melissa Y. Martin, Giordane A. Martins, Hidekazu Matsueda, Kathryn McKain, Harro Meijer, Frank Meinhardt, Lynne Merchant, N. Mihalopoulos, Natasha Miles, John B. Miller, Charles E. Miller, Logan Mitchell, Stephen Montzka, Fred Moore, Heiko Moossen, Eric Morgan, Josep-Anton Morgui, Shinji Morimoto, Bill Munger, David Munro, Cathrine L. Myhre, Meelis Mölder, Jennifer Müller-Williams, Jaroslaw Necki, Sally Newman, Sylvia Nichol, Yosuke Niwa, Simon O'Doherty, Florian Obersteiner, Bill Paplawsky, Jeff Peischl, Olli Peltola, Salvatore Piacentino, Jean M. Pichon, Steve Piper, Joseph Pitt, Christian Plass-Duelmer, Stephen M. Platt, Steve Prinzivalli, Michel Ramonet, Ramon Ramos, Enrique Reyes-Sanchez, Scott Richardson, Haris Riris, Pedro P. Rivas, Michael Rothe, Thomas Ryerson, Kazuyuki Saito, Maryann Sargent, Motoki Sasakawa, Bert Scheeren, Martina Schmidt, Tanja Schuck, Marcus Schumacher, Thomas Seifert, Mahesh K. Sha, Paul Shepson, Michael Shook, Christopher D. Sloop, Paul Smith, Martin Steinbacher, Britton Stephens, Colm Sweeney, Lise L. Sørensen, Pieter Tans, Kirk Thoning, Helder Timas, Margaret Torn, Pamela Trisolino, Jocelyn Turnbull, Kjetil Tørseth, Alex Vermeulen, Brian Viner, Gabriela Vitkova, Stephen Walker, Andrew Watson, Ray Weiss, Steve Wofsy, Justin Worsey, Doug Worthy, Dickon Young, Sönke Zaehle, Andreas Zahn, Miroslaw Zimnoch, Rodrigo A. de Souza, Alcide G. di Sarra, Danielle van Dinther, Pim van den Bulk; Multi-laboratory compilation of atmospheric carbon dioxide data for the period 1957-2021; obspack_co2_1_GLOBALVIEWplus_v8.0_2022-08-27; NOAA Earth System Research Laboratory, Global Monitoring Laboratory. http://doi.org/10.25925/20220808.
#
# E-mail List of Data Providers:
#
# A.Colomb@opgc.univ-bpclermont.fr, Ari.Leskinen@fmi.fi, Arjan.Hensen@tno.nl, Arnoud.frumau@tno.nl, Brian.Viner@srnl.doe.gov, Casper.Labuschagne@weathersa.co.za, Cedric.Couret@uba.de, Chad.Hanson@oregonstate.edu, Christian.Plass-Duelmer@dwd.de, Danielle.vanDinther@tno.nl, David.Bowling@utah.edu, Dickon.Young@bristol.ac.uk, Eric.J.Hintsa@noaa.gov, Giovanni.MANCA@ec.europa.eu, Gordon.Brailsford@niwa.co.nz, J.M.Pichon@opgc.fr, Jennifer.Mueller-Williams@dwd.de, Logan.Mitchell@utah.edu, MLFischer@lbl.gov, MSTorn@lbl.gov, Martina.schmidt@iup.uni-Heidelberg.de, Michal.Heliasz@cec.lu.se, Michel.Ramonet@lsce.ipsl.fr, P.Cristofanelli@isac.cnr.it, Paul.Krummel@csiro.au, Pim.vandenbulk@tno.nl, Ray.Langenfelds@csiro.au, SCBiraud@lbl.gov, Sylvia.Nichol@niwa.co.nz, Tobias.Kneuer@dwd.de, Tuomas.Laurila@fmi.fi, Xin.Lan@noaa.gov, Zoe.Loh@csiro.au, accox@ucsd.edu, ajordan@bgc-jena.mpg.de, alcide.disarra@enea.it, alex.vermeulen@nateko.lu.se, am12721@bristol.ac.uk, andreas.beyersdorf@csusb.edu, andreas.zahn@kit.edu, a.manning@uea.ac.uk, anna.karion@nist.gov, antje.hoheisel@iup.uni-heidelberg.de, aoki@m.tohoku.ac.jp, arlyn.andrews@noaa.gov, bev.law@oregonstate.edu, bianca.baier@noaa.gov, cgerbig@bgc-jena.mpg.de, christian.plass-duelmer@dwd.de, clm@nilu.no, colm.sweeney@noaa.gov, crl@nilu.no, dagmar.kubistin@dwd.de, daniela.heltai@rse-web.it, david.munro@noaa.gov, dewekker@virginia.edu, djaffe@uw.edu, dlam@hko.gov.hk, doug.worthy@canada.ca, e.gloor@leeds.ac.uk, e.kozlova@exeter.ac.uk, eakort@umich.edu, ecuevasa@aemet.es, ereyess@aemet.es, florian.obersteiner@kit.edu, francesco.apadula@rse-web.it, frank.meinhardt@uba.de, fred.moore@noaa.gov, g.forster@uea.ac.uk, gao.chen@nasa.gov, ghg_obs@met.kishou.go.jp, giordanemartins@gmail.com, h.a.j.meijer@rug.nl, h.a.scheeren@rug.nl, harald.boenisch@kit.edu, haris.riris-1@nasa.gov, haszpra.l@met.hu, heiko.moossen@bgc-jena.mpg.de, hmatsued@mri-jma.go.jp, ingeborg.levin@iup.uni-heidelberg.de, ingrid.luijkx@wur.nl, irene.lehner@cec.lu.se, ivakhooo@mail.ru, ivan.mammarella@helsinki.fi, jocelyn.turnbull@noaa.gov, janne.levula@helsinki.fi, jdella@iup.uni-heidelberg.de, jeff.peischl@noaa.gov, jjkim@ucsd.edu, john.b.miller@noaa.gov, john.mund@noaa.gov, josepanton.morgui@uab.cat, joseph.pitt@bristol.ac.uk, joshua.p.digangi@nasa.gov, jtlee@maine.edu, juha.hatakka@fmi.fi, jutta.holst@nateko.lu.se, jwmunger@seas.harvard.edu, kari.lehtinen@fmi.fi, kathryn.mckain@noaa.gov, kenneth.c.aikin@noaa.gov, kenneth.schuldt@noaa.gov, kirk.w.thoning@noaa.gov, kjd10@psu.edu, kominkova.k@czechglobe.cz, kt@nilu.no, leuenberger@climate.unibe.ch, lls@bios.au.dk, lmerchant@ucsd.edu, lukas.emmenegger@empa.ch, lukasz.chmura@fis.agh.edu.pl, lvgatti@gmail.com, mahesh.sha@aeronomie.be, marc.delmotte@lsce.ipsl.fr, marek.mv@czechglobe.cz, martin.heimann@bgc-jena.mpg.de, martin.steinbacher@empa.ch, martine.demaziere@aeronomie.be, matthias.lindauer@dwd.de, meelis.molder@nateko.lu.se, michael.shook@nasa.gov, michal.galkowski@agh.edu.pl, mon@m.tohoku.ac.jp, morgan.lopez@lsce.ipsl.fr, mracine@seas.harvard.edu, necki@agh.edu.pl, niwa.yosuke@nies.go.jp, nmiles@psu.edu, oh@nilu.no, olee@hko.gov.hk, olivier.laurent@lsce.ipsl.fr, p.trisolino@isac.cnr.it, pasi.kolari@helsinki.fi, petri.keronen@helsinki.fi, pieter.tans@noaa.gov, privass@aemet.es, pshepson@purdue.edu, rafsouza@uea.edu.br, rfweiss@ucsd.edu, rkeeling@ucsd.edu, roger.curcoll@uab.cat, s.odoherty@bristol.ac.uk, s5clark@ucsd.edu, salvatore.piacentino@enea.it, samuel.hammer@iup.uni-heidelberg.de, sasakawa.motoki@nies.go.jp, scpiper@ucsd.edu, sebastien.conil@andra.fr, sitesinnetwork@climadat.es, sjwalker@ucsd.edu, snewman@baaqmd.gov, sp@nilu.no, sprinzivalli@earthnetworks.com, srichardson@psu.edu, stephan.r.kawa@nasa.gov, stephen.a.montzka@noaa.gov, stephens@ucar.edu, swofsy@seas.harvard.edu, szaehle@bgc-jena.mpg.de, tgriffis@umn.edu, tmachida@nies.go.jp, tobias.biermann@cec.lu.se, torsten.gehrlein@kit.edu, tul5@psu.edu, tuula.aalto@fmi.fi, victor.kazan@lsce.ipsl.fr, vitkova.g@czechglobe.cz, wbrand@bgc-jena.mpg.de, wpaplawsky@ucsd.edu, zimnoch@agh.edu.pl

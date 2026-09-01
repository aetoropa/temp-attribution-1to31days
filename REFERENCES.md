The methodology of the script is documented in this manuscript which has been submitted to Geoscientific Model Development:

Toropainen, A., Rantanen, M., and Räisänen, J.: ATDT v1.0 – Attribution Tool for Daily-to-monthly Temperatures 
and its application to record-breaking Northern European heatwave of July 2025, EGUsphere [preprint],
https://doi.org/10.5194/egusphere-2026-3721, 2026.

In addition, the script and the methodology builds on and expands the methodologies described in these three papers:

Räisänen, J. and L. Ruokolainen, 2008: Estimating present climate in a
warming world: a model-based approach. Climate Dynamics, 31, 573-585

and

Räisänen, J. and L. Ruokolainen, 2008: Ongoing global warming and local
warm extremes: a case study of winter 2006-2007 in Helsinki, Finland.
Geophysica, 44, 45-65.

and

Rantanen, M., Räisänen, J., & Merikanto, J. (2024): 
A method for estimating the effect of climate change on monthly mean temperatures: 
September 2023 and other recent record-warm months in Helsinki, Finland.
Atmospheric Science Letters, 25(6), e1216.
https://doi.org/10.1002/asl.1216


The algorithm by which calendar days for non-leap years are converted to days of year in the ```doy``` function follows that of:
Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7

FMI observations are downloaded by calling the function ```read_daily_obs_from_FMI``` which requires a Python Interface fmiopendata: https://github.com/pnuu/fmiopendata

SMHI observations are downloaded by calling the function ```read_daily_obs_from_SMHI``` which utilizes the SMHI API: https://opendata.smhi.se/metobs/api

METNO observations are downloaded by calling the function ```read_daily_obs_from_FROST``` which utilizes the FROST API: see https://frost.met.no/howto.html

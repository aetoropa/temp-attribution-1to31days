# temp-attribution-1to31day

This repository contains a Python program for estimating probability distributions of local daily mean, maximum and minimum temperatures or their 2 to 31-day moving averages in a changing climate. The program modifies local observational time-series by using (a) time-series of global mean temperature and (b) regression coefficients which estimate how the mean and variance of local temperature change in response to the global mean temperature change, resulting in a detrended time-series of "pseudo-observations". Probability distributions are estimated by applying quantile-regression to the time-series of pseudo-observations and fitting a 5-parameter logistic function to the daily values of quantiles 0.01,...,0.99 and Gumbel-distributions for the left and right tails. Finally, the program calculates how intense and probable the observed temperature would be in the pre-industrial and future climates. The methodoloy is documented in: ... 

## Python requirements

You need the following libraries:
* numpy 1.26.4
* pandas 2.2.2
* xarray 2023.6.0
* scipy 1.13.1
* statsmodels.api 0.14.2
* requests 2.32.3
* tqdm 4.66.5
* fmiopendata 0.5.0

## Run the main program

In the Python script ```estimate_distributions.py``` you need to define 3 sets of parameters, as described in the following 3 subsections. The first set defines the parameters that affect the observations which are used in the attribution. The second set defines the attribution cases (e.g. pre-industrial, present-day and future years) and the time-period. The third set defines how probability for warmer/colder temperatures is calculated, emission scenario and the number of bootstrapping samples. In addition to these parameters, you need to define paths for input data and saving the output figures.

### Observation parameters

The first observation parameter is:

```obs_source```

which defines the source of the observations. Currently, the script only works for temperature observations from Finnish Meteorological Institute (FMI), Swedish Meteorological and Hydrological Institute (SMHI) and MetNorway (FROST). If you use observations from FROST, you need to specify ```frost_client_id```, which is required for downloading observations from the FROST API (See: https://frost.met.no/authentication.html). If you do not have the FROST client id, just comment out this line.

The second observation parameter is:  

```station_id```

which is the weather station's identification number (e.g. ```station_id``` = 101932 for Sodankylä, Finland; ```station_id``` = 161970 for Piteå, Sweden and ```station_id``` = SN18700 for Oslo, Norway). Currently, all FMI weather stations are available. However, the availability of SMHI and FROST-observations depends on the station and climate variable. The script works for the stations used in the paper cited above, but it may not work for every other SMHI or FROST station.\

If you use SMHI-observations, you can also define the ```station2_id``` parameter if you wish to concatenate the observational time-series of stations ```station_id``` and ```station2_id```. This can be done if a newer automatic weather station has replaced an older weather station and is located nearby to it. For example, if you wish to use observations from the "Abisko" and the "Abisko Aut" stations, you would set ```station_id``` = 188800 and ```station2_id``` = 188790. If you don't intend on using observations from a secondary station, comment the line out.  

The third observation parameter is:  

```clim_var```  

This variable defines the climate variable and follows the naming convention used in the CMIP6-data files (e.g. "tas" for mean temperature, "tasmax" for maximum temperature and "tasmin" for minimum temperature).  

### Attribution cases and time-period

This set of parameters defines the years and the time-period which are used in the attribution.
The first parameter is:

```preind_year```

which defines the pre-industrial year. Typically, ```preind_year```= 1900 is used.  

The second parameter is:

```target_year```

which defines the target year of the observation and is typically used to define the present-day climate.

The third parameter is:

```future_year```

which defines the future year. Typically, ```future_year``` = 2050 is used. If you don't want to calculate the attribution results for the future year, set ```future_year``` = None.

The fourth parameter is:

```use_obs```

which defines, whether the probability of warmer/colder temperatures and return times are calculated from observations over the baseline period and if these results are shown in the distribution plot.

The fifth parameter is:

```start_month```

which specifies the start month in the period (1-12).

The sixth parameter is:

```start_day```

which defines the day of the month (1-31).

The seventh parameter is:

```end_month```

which defines the end month in the period (1-12). Bear in mind, that only 1 to 31-day long periods are allowed. If you intend to calculate attribution results for the date speficied by ```start_month``` and ```start_day``` variables, comment this line out.

The eight parameter is:

```end_day```

which defines the end day in the period (1-31). Bear in mind, that only 1 to 31-day long periods are allowed. If you intend to calculate attribution results for the date speficied by ```start_month``` and ```start_day``` variables, comment this line out.


### Probability, scenario and uncertainty estimate

This set of parameters defines how probability is calculated, the emission scenario and number of boostrapping iterations.

The first parameter you need to define is:  

```pwarm```  

which is the probability of warmer (True) or colder (False) temperatures. If ```pwarm``` = True, the probability of higher temperatures than the one that was observed is given in the distribution plot. In contrast, if ```pwarm``` = False, the probability of lower temperatures than the one that was observed is given in the distribution plot.  

The second parameter you need to define is:  

```ssp```

which is the emission scenario for future climate (```ssp```= "ssp119", "ssp126", "ssp245" and "ssp585" are available).

The thids parameter you need to dfine is:

```n_boots```

which defines the number of bootstrapping samples. If you set ```n_boots``` = 0, bootstrapping will not be applied.

In addition to these parameters, you need to specify the first and last years of observations (```base1_year``` and ```base2_year```) used in estimating the probability distribution and the path to input_data and saving figures. Typically, ```base1_year```= 1901 and ```base2_year``` = ```target_year``` - 1.  

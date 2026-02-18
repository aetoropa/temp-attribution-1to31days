# temp-attribution-1to31day

This repository contains a Python program for estimating probability distributions of local daily mean, maximum and minimum temperatures or their 2 to 31-day moving averages in a changing climate. The program modifies local observational time-series by using (a) time-series of global mean temperature and (b) regression coefficients which estimate how the mean and variance of local temperature change in response to the global mean temperature change, resulting in a detrended time-series of "pseudo-observations". Probability distributions are estimated by applying quantile-regression to the time-series of pseudo-observations and fitting continuous probability distributions to the daily values of quantiles 0.01,...,0.99 and the left and right tails. Finally, the program calculates how intense and probable the observed temperature would be in the pre-industrial and future climates. The methodoloy is documented in: ... 

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

### Years and time-period

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

The third parameter you need to define is:

```n_boots```

which defines the number of bootstrapping samples. If you set ```n_boots``` = 0, bootstrapping will not be applied.

In addition to these parameters, you need to specify the first and last years of observations (```base1_year``` and ```base2_year```) used in estimating the probability distribution and the path to input_data and saving figures. Typically, ```base1_year```= 1901 and ```base2_year``` = ```target_year``` - 1.  

## Output

The program prints the annual probabilities and temperatures for ```preind_year```, ```target_year``` and ```future_year``` together with probability ratio and intensity change. The output looks as follows for the time-period of 12-25 July, 2025 in Sodankylä, Finland:

Attribution results for FMI station 101932 (Sodankylä Tähtelä):  

Location: (67.37 °N, 26.63 °E)  
Climate variable: 14-day mean of daily maximum temperature  
Time-period: 12 – 25 July  

Annual probabilities:  
1900: 0.38% (0.04% - 1.11%)  
2025: 1.80% (0.36% - 5.13%)  
2050: 3.90% (0.59% - 11.92%)  

Annual temperatures:  
1900: 25.8°C (24.7°C - 27.7°C)  
2025: 28.0°C  
2050: 29.0°C (28.1°C - 30.2°C)  

Probability ratio: 4.8 (1.2 - 28.1)  
Change in intensity: (2.2°C) (0.3-3.3)  

In addition, four plots are produced:  

<img width="2621" height="1444" alt="time_series_plot_tasmax_Sodankylä Tähtelä_0712-0725" src="https://github.com/user-attachments/assets/65538269-e6d3-4383-8c1a-de9084123922" />
Figure 1. Time-series plot of the 14-day moving averge of daily maximum temperature for the period of 12-25 July. Black line shows the actual observation, blue dots show the corresponding multi-model mean pseudo-observations, representing present-day climate. The red error bars show the 5th and 95th percentiles of the ensemble of 31 model-specific pseudo-observations. The blue dashed line marks the observation (28.0 °C) of year 2025.

<img width="3295" height="1747" alt="observation_plot_tasmax_Sodankylä Tähtelä_0712-0725" src="https://github.com/user-attachments/assets/701a7a50-1b6f-410d-963a-2b99ecc22bd3" />
Figure 2. 14-day moving avearge daily maximum temperature observations in Sodankylä, Finland from 1908 to 2025 (blue dots) and their quantile functions for the median (yellow) and selected low (green and blue) and high (orange and red). The observation corresponding to the heatwave of summer 2025 is highlighted with a red dot. The red shaded area higlights the time-period of the observation (12-25 July).

<img width="5434" height="1965" alt="distribution_plot_FMI_101932_0712-0725" src="https://github.com/user-attachments/assets/d75f804f-f98d-4005-83fe-e0360faff60d" />
Figure 3. a) The frequency distribution of pseudo-observations representing the 14-day moving average of daily maximum temperature observations in the present-day climate (2025) in Sodankylä, Finland (blue bars) and the corresponding continuous probability distribution function (blue line). In the upper left corner, the values of the four moments are annotated: mean (μ), variance (σ²), skewness (γ) and kurtosis (κ). b) The continuous probability distributions of pseudo-observations for climates in years 1900 (green line), 2025 (blue line) and 2050 (red line). Black vertical line marks the observation of the year 2025. The text panel shows the probabilities of observing higher temperatures than the observed temperature of 28.0 °C, return periods and intensities.

<img width="2446" height="2985" alt="model_statistics_FMI_101932_0712-0725" src="https://github.com/user-attachments/assets/947ea556-77f8-487c-86e0-e4f7445dfd0e" />
Figure 4. Model-simulated probability ratios and intensity changes for the time-period of 12-25 July, 2025 in Sodankylä, Finland. The uncertainty in the probability ratio values is entirely due to internal variability as only one realization per model is used. The boxes show the first and third quartiles and whiskers show the 5-95th percentiles of the realization. MMM at the bottom row refers to multi-model mean estimate.

## More information

More information can be asked from\
Antti Toropainen\
Doctoral Researcher, University of Helsinki\
antti.toropainen@helsinki.fi






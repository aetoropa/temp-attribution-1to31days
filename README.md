# temp-attribution-1to31day

This repository contains a Python program for estimating probability distributions of local daily mean, maximum and minimum temperatures or their 2 to 31-day moving averages in a changing climate. The program modifies local observational time-series by using (a) time-series of global mean temperature and (b) regression coefficients which estimate how the mean and variance of local temperature change in response to the global mean temperature change, resulting in a detrended time-series of "pseudo-observations". Probability distributions are estimated by applying quantile-regression to the time-series of pseudo-observations and fitting continuous probability distributions to the daily values of quantiles 0.01,...,0.99. Finally, the program calculates how intense and probable the observed temperature would be in the pre-industrial and future climates. The methodoloy is documented in: ... 

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

Before running the Python script ```estimate_distributions.py```, you need to download the input data set from Zenodo: https://doi.org/10.5281/zenodo.20742664. Follow the instructions in Zenodo to unpack the dataset and to understand the directory structure. After downloading the input data set and unpacking it, you need to define 3 sets of parameters, as described in the following 3 subsections. The first set defines the parameters that specify the observations which are used in the attribution. The second set defines the attribution cases (e.g. pre-industrial, present-day and future years) and the time-period. The third set defines how the probability for warmer/colder temperatures is calculated, emission scenario and the number of bootstrapping samples. In addition to these parameters, you need to define paths for input data, the output figures and saving the attribution results.

### Observation parameters

The first observation parameter is:

```obs_source```

which defines the source of the observations. Currently, the script only works for temperature observations from Finnish Meteorological Institute (FMI), Swedish Meteorological and Hydrological Institute (SMHI) and Meteorological Institute of Norway (METNO). If you use observations from METNO, you need to specify ```frost_client_id```, which is required for downloading observations from the FROST API (See: https://frost.met.no/authentication.html). If you do not have the FROST client id, set ```frost_client_id```= None.

The second observation parameter is:  

```station_id```

which is the weather station's identification number (e.g. ```station_id``` = 101932 for Sodankylä, Finland; ```station_id``` = 161970 for Piteå, Sweden and ```station_id``` = SN18700 for Oslo - Blindern, Norway). Currently, all FMI weather stations are available. However, the availability of SMHI and METNO-observations depends on the station and climate variable. The script works for the stations used in the paper cited above, but it may not work for every other SMHI or METNO station.

If you use SMHI-observations, you can also define the ```station2_id``` parameter if you wish to concatenate the observational time-series of stations ```station_id``` and ```station2_id```. This can be done if a newer automatic weather station has replaced an older weather station and is located nearby to it. For example, if you wish to use observations from the "Abisko" and the "Abisko Aut" stations, you would set ```station_id``` = 188800 and ```station2_id``` = 188790. If you don't intend on using observations from a secondary station, set ```station2_id```= None. Please note, that the current form of the code only allows the download of historical observations (at least 4 months old) from SMHI. For FMI and METNO, the download scripts should work for the most recently made observation (e.g. the previous day). 

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

```show_obs_distr```

which defines, whether the probability of warmer/colder temperatures and return times are also calculated from observations over the baseline period and if these results are shown in the 'distribution plot' and 'observation plot'.

The fifth parameter is:

```start_month```

which specifies the start month (1-12) for the 1-to-31-day period for which the probability distributions are calculated.

The sixth parameter is:

```start_day```

which defines the day of the month (1-31).

The seventh parameter is:

```end_month```

which defines the end month in the period (1-12). If you intend to calculate attribution results for the date speficied by ```start_month``` and ```start_day``` variables, set ```end_month``` = None.

The eight parameter is:

```end_day```

which defines the end day in the period (1-31). If you intend to calculate attribution results for the date speficied by ```start_month``` and ```start_day``` variables, set ```end_day``` = None.


### Probability, scenario and uncertainty estimate

This set of parameters defines how probability is calculated, the emission scenario and number of boostrapping iterations.

The first parameter you need to define is:  

```pwarm```  

If ```pwarm``` = True, the probability of higher temperatures than the one that was observed is given in the distribution plot. In contrast, if ```pwarm``` = False, the probability of lower temperatures than the one that was observed is given in the distribution plot.  

The second parameter you need to define is:  

```ssp```

which is the emission scenario for future climate (```ssp```= "ssp119", "ssp126", "ssp245" and "ssp585" are available).

The third parameter you need to define is:

```n_boots```

which defines the number of bootstrapping samples. If you set ```n_boots``` = 0, bootstrapping will not be performed. If you're running the code on laptop and you don't have the files containing pre-computed values of quantile functions (See the files in: "/input_data/qr_files/single_models"), it is recommended to set ```n_boots``` = 0, as performing quantile regression separately for up 31 samples of model-specific pseudo-observations in both ```preind_year```, ```target_year``` and possibly ```future_year``` years can take up to 2 hours of time, requiring a significant amount of your laptop's computational capabilities.

In addition to these parameters, you need to specify the first and last years of observations (```base1_year``` and ```base2_year```) used in estimating the probability distribution and the path to input_data and saving figures. Typically, ```base1_year```= 1901 and ```base2_year``` = ```target_year``` - 1.  

## Output

The program prints the annual probabilities and temperatures for ```preind_year```, ```target_year``` and ```future_year``` together with probability ratio and intensity change. The results are also saved to a csv-file in the "attribution_results" directory. The output looks as follows for the time-period of 12-25 July, 2025 in Sodankylä, Finland:

Attribution results for FMI station 101932 (Sodankylä Tähtelä):  

Location: (67.37 °N, 26.63 °E)  
Climate variable: 14-day mean of daily maximum temperature  
Time-period: 12 – 25 July  

Annual probabilities:  
1900: 0.38% (0.03% - 1.13%)  
2025: 1.79% (0.31% - 4.97%)  
2050: 3.86% (0.59% - 11.92%)  

Annual temperatures:  
1900: 25.9°C (24.7°C - 27.7°C)  
2025: 28.0°C  
2050: 29.0°C (28.1°C - 30.2°C)  

Probability ratio: 4.6 (1.2 - 32.5)
Change in intensity: 2.1°C (0.3 °C - 3.3°C)  

In addition, four plots are produced:  

<img width="2621" height="1517" alt="time_series_plot_tasmax_FMI_Sodankylä Tähtelä_0712-0725" src="https://github.com/user-attachments/assets/b195d451-4003-45f7-b896-6571ae145df8" />
Figure 1. Time-series plot of the 14-day moving averge of daily maximum temperature for the period of 12-25 July. Black line shows the actual observation, blue dots show the corresponding multi-model mean pseudo-observations, representing present-day climate. The red error bars show the 5th and 95th percentiles of the ensemble of 29 model-specific pseudo-observations. The blue dashed line marks the observation (28.0 °C) of year 2025.

<img width="3295" height="1747" alt="observation_plot_tasmax_Sodankylä Tähtelä_0712-0725" src="https://github.com/user-attachments/assets/701a7a50-1b6f-410d-963a-2b99ecc22bd3" />
Figure 2. 14-day moving avearge daily maximum temperature observations in Sodankylä, Finland from 1908 to 2025 (blue dots) and their quantile functions for the median (yellow) and selected low (green and blue) and high (orange and red) quantiles. The observation corresponding to the heatwave of summer 2025 is highlighted with a red dot. The red shaded area higlights the time-period of the observation (12-25 July).

<img width="5464" height="1965" alt="distribution_plot_FMI_Sodankylä Tähtelä_0712-0725" src="https://github.com/user-attachments/assets/d25b614b-f792-479c-90a3-88d4cbb61ed9" />
Figure 3. (a) The frequency distribution of pseudo-observations representing the 14-day moving average of daily maximum temperatures for the time period of 12-25 July in the climate of the year 2025 in Sodankylä, Finland. The corresponding probability distribution function is shown with a blue line. The values of the first four moments are annotated in the upper left corner of the figure. (b) Probability distribution functions for the climates of the years 1900 (green line), 2025 (blue line) and 2050 (red line) under the SSP2-4.5 scenario. The black vertical line marks the observed 14-day moving average of daily maximum temperatures for the time-period of 14-25 July 2025. The probability for the temperature being at least the observed 28.0 °C is shown by the shaded area right of the vertical black line below each probability distribution function.

<img width="2446" height="2985" alt="model_statistics_FMI_Sodankylä Tähtelä_0712-0725" src="https://github.com/user-attachments/assets/b1b468ac-9ced-423e-a69c-2baa862a6598" />
Figure 4. Model-specific probability ratios and intensity changes of 14-day average maximum temperature for the time-period of 12-25 July, 2025 in Sodankylä, Finland. The uncertainty in the probability ratio values results from bootstrap sampling of quantiles. No uncertainty estimates are provided for the intensity changes in the individual models because only one realization per model was used to calculate the regression coefficients. The boxes show the first and third quartiles and whiskers show the 5-95th percentiles of the realization. MMM at the bottom row refers to the multi-model mean estimate.

## More information

More information can be asked from\
Antti Toropainen\
Doctoral Researcher, University of Helsinki\
antti.toropainen@helsinki.fi






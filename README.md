# temp-attribution-1to31days
This repository contains shell and python scripts for:  

1. Pre-processing data from the Coupled Climate Model Inter-comparison project (CMIP6). This includes:  
    a) Extracting the simulated global mean temperature  
    b) Calculating regression coefficients  
  
2. Estimating probability distributions of local daily mean, maximum and minimum temperatures or their 2 to 31-day moving averages in a changing climate. The program modifies local observational time-series by using (a) time-series of global mean temperature and (b) regression coefficients which estimate how the mean and variance of local temperature change in response to the global mean temperature change, resulting in a detrended time-series of "pseudo-observations". Probability distributions are estimated by applying quantile-regression to the time-series of pseudo-observations and fitting a 5-parameter logistic function to the daily values of quantiles 0.01,...,0.99 and Gumbel-distributions for the left and right tails. Finally, the program calculates how intense and probable the observed temperature would be in the pre-industrial and future climates.  

## Preparing the input data

Preparing the input data (1) is accomplished by running three shell scrits. The first one these is called: ```calculate_Tglob_and_regression_coefficients.sh```, which calculates the model-specific global mean temperature change with respect to the year 2000 for the time period of 1901-2099 and the model-specific regression coefficients. You need to define the parameters ```hist_file``` (historical CMIP6-simulation file), ```proj_file``` (future CMIP6-simulation file) and ```n_days``` (the number of days in the time-period). Please note, that ```n_days``` needs to be an odd number between 1 and 31. This script needs to be applied separately to all combinations of CMIP6-models and number of days you wish to used. 

### Example use

If you wanted to calculate the regression coefficients for the ACCESS-CM2 model for 15-day periods, and the names of your```hist_file``` and ```proj_file``` were called "tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_1850-2014.nc" and "tas_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_2015-2100.nc", you would run the following:  
```bash
./calculate_Tglob_and_regression_coefficients.sh tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_1850-2014.nc tas_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_2015-2100.nc 15

This will return...

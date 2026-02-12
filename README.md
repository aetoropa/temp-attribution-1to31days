# temp-attribution-1to31days
This repository contains shell and python scripts for:

1. Pre-processing data from the Coupled Climate Model Inter-comparison project (CMIP6). This includes:
    a) Extracting the simulated global mean temperature
    b) Calculating regression coefficients
  
2. Estimating probability distributions of local daily mean, maximum and minimum temperatures or their 2 to 31-day moving averages in a changing climate. The program modifies local observational time-series by using (a) time-series of global mean temperature and (b) regression coefficients which estimate how the mean and variance of local temperature change in response to the global mean temperature change, resulting in a detrended time-series of "pseudo-observations". Probability distributions are estimated by applying quantile-regression to the time-series of pseudo-observations and fitting a 5-parameter logistic function to the daily values of quantiles 0.01,...,0.99 and Gumbel-distributions for the left and right tails. Finally, the program calculates how intense and probable the observed temperature would be in the pre-industrial and future climates. 

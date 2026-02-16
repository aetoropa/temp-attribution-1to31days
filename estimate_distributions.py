#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

This script calculates probability distributions of observed daily mean, maximum and minimum temperatures
or their 2 to 31-day moving averages in the past, observed and future climates for weather station observations 
provided by the Finnish Meteorological Institute (FMI), Swedish Meteorological and Hydrological Institute (SMHI) and MetNorway (FROST).

The script is based on and expands the methodology documented in these three papers:

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

@authors: Antti Toropainen and Mika Rantanen
"""

# Import needed libraries
import numpy as np
import functions

"""
################################################################################
First, give the input parameters: obs_source, station_id, climate variable,
emission scenario and the number of bootstrap samples
################################################################################
"""

# Frost client ID (You need this to download MetNorway observations from Frost API)
# You can obtain a personal ID from: https://frost.met.no/authentication.html (Comment this line out, if you don't have 'frost_client_id')
frost_client_id = ""

# Observations source (FMI, SMHI, FROST)
obs_source = "FMI"

# Station ID (101932 = Sodankylä, Tähtelä; 100971 = Helsinki, Kaisaniemi)
station_id = "101932"

# Station2 ID (only some SMHI stations)
#station2_id = "134110"

# Climate variable (tas, tasmax, tasmin)
clim_var = "tasmax"

"""
################################################################################
Next, sepcify the years and if results are calculated for the baseline observations.
In addition, define a time-period of 1 to 31 days.
################################################################################
"""

# Preindustrial climate year
preind_year = 1900

# Target climate year
target_year = 2025

# Future climate year (Set 'future_year' = None, if you don't want attribution results for future climate)
future_year = 2050

# Compute attribution results for observations for the baseline period (False, if observations are neglected)
use_obs = True

# Start Month (1-12)
start_month = 7

# Start day (1-31)
start_day = 12

# End month (1-12) (Comment this out, if attribution is done for single day)
end_month = 7

# End day (1-31) (Comment this out, if attribution is done for single day)
end_day = 25

# Get the 'day_of_year' index (the center or the last day of the time-period) and the number of days 'n_days' in the time-period
doy_index, n_days = functions.get_doy_index_and_ndays(start_month, start_day, end_month, end_day)

"""
################################################################################
Lastly, define probability for warmer/colder temperatures, emission scenario
and number of bootstrapping samples.
In addition, specify some auxiliary variables, such as, baseline period,
temperature range of the distribution, quantiles and directory paths
################################################################################
"""

# Probability of warmer temperatures (colder if false)
pwarm = True

# Scenario for future climate: Currently, only ssp245 is available.
ssp = "ssp245"

# Number of bootstrap samples (if 'n_boots' = 0, bootstrapping will NOT be applied)
n_boots = 0

# The first and last years of observations used in the calculation of probability distributions
base1_year = 1901
base2_year = 2024

# The observed temperature is within -50 ... +40 C
valmin = -50.0
valmax = 40.0
nbins = 1601

# Temperature range array
T_range = np.linspace(valmin, valmax, nbins)

# Quantiles 0.01...0.99 for quantile regression (QR)
quantiles = np.arange(0.01, 1.00, 0.01)

# Paths to directories for reading data and saving the output figures
input_data_dir = "/home/aetoropa/tas_attribution_tool/input_data"
path2figures = "/home/aetoropa/tas_attribution_tool/figures/"


"""
############################################################################
Construct dictionaries for the attribution cases and some input parameters
############################################################################
"""

# A dictionary for attribution cases
attribution_cases = {
    "preind":preind_year,
    "target":target_year,
}

# Future year is added, if it's not None
if future_year is not None:
    attribution_cases["future"] = future_year

# Observations are added, if 'use_obs' is not None
if use_obs is not None:
    attribution_cases["obs"] = "obs"

# Save initialization variables to a dictionary, so it's easier to pass them as arguments to functions
config_vars = {
    "obs_source": obs_source,
    "station_id": station_id,
    "ssp": ssp,
    "clim_var": clim_var,
    "doy_index": doy_index,
    "n_days": n_days,
    "base1_year": base1_year,
    "base2_year": base2_year,
    "quantiles": quantiles
    }

"""
################################################################
Read local daily temperature observations and calculate their
2 to 31-day moving averages (if needed). 
################################################################
"""

# Read the observational data 
# SMHI: remember to provide 'station2_id' if needed
# FROST: remember to specify 'frost_client_id'
daily_temp_obs_df, station_meta = functions.read_daily_obs(obs_source, clim_var, station_id)

# Compute n-day running mean of the observations if needed
if n_days > 1:
    obs_df = functions.compute_running_mean(daily_temp_obs_df, n_days)
else:
    obs_df = daily_temp_obs_df

# Check the validity of observations
functions.check_obs_validity(obs_df, doy_index, n_days, target_year)

"""
################################################################
Read observed and simulated global temperatures, merge them and
smooth with 11-year moving average. In addition, read the
regression coefficients for mean and variability.
################################################################
"""

# Read observed global mean temp and merge it with the simulated temperature for the model mean (mm) and single model (sm) cases
glob_temp_mm_df = functions.read_sim_temp_model_mean(input_data_dir, clim_var, ssp)
glob_temp_sm_df = functions.read_sim_temp_single_models(input_data_dir, clim_var, ssp)

# Read model mean (mm) and single model (sm) regression coefficients
coeffs_mm_ds = functions.read_coeffs_model_mean(input_data_dir, clim_var, ssp, station_meta["latitude"], station_meta["longitude"], n_days)
coeffs_sm_ds = functions.read_coeffs_single_models(input_data_dir, clim_var, ssp, station_meta["latitude"], station_meta["longitude"], n_days)

"""
###############################################################################
Compute model-mean and single-model pseudo-observations for all attribution cases.
For the model-mean case, QR is applied to all cases if it hasn't been applied
previously.
For the single-mode case, QR is applied (or pre-computed results are used) if
uncertainty estimate is calculated by bootstrapping ('n_boots > 0').
################################################################################
"""

# Compute model-mean pseudo-observations for the selected time-period and and get OR compute the corresponding quantiles
pseudo_obs_mm_dict, qr_obs_mm_dict = functions.get_pseudo_obs_and_qr_mm(attribution_cases, config_vars, input_data_dir, obs_df, coeffs_mm_ds, glob_temp_mm_df)
pseudo_obs_sm_dict, qr_sm_dict = functions.get_pseudo_obs_and_qr_sm(attribution_cases, config_vars, input_data_dir, obs_df, coeffs_sm_ds, glob_temp_sm_df, n_boots)

"""
################################################################
Fit a continuous CDF to the quantiles that correspond to the
selected time-period in all attribution cases.  
################################################################
"""

# Fit continuous probability distributions to the quantiles in the pre-industrial and present-day climates
# and optionally in the future and the observed climate
CDF_mm_dict, PDF_mm_dict = functions.get_mm_distributions(qr_obs_mm_dict, quantiles, T_range, doy_index)

"""
################################################################
If 'n_boots > 0' bootstrapping is applied to account for uncertainty
related to internal climate variability. 
################################################################
"""

# Apply bootstrapping to get confindence intervals
if n_boots > 0:
    bootstrap_3Darr_dict = functions.apply_bootstrapping(n_boots, doy_index, T_range, quantiles, qr_obs_mm_dict, qr_sm_dict)
# Continue without confidence intervals
else:
    bootstrap_3Darr_dict = None

"""
#######################################################################
Print probabilities of warmer/colder temperatures together with
the corresponding temperatures. In addition, print intensity change,
and the corresponding probability ratio.
5-95 % intervals from bootstrapping are shown in parentheses.
#######################################################################
"""

# The observed temperature corresponding to the target year and doy_index
target_value = obs_df.loc[target_year][doy_index]

# The corresponding index in the T_range array
target_index = functions.get_element_index(T_range, target_value)

# Percentile in today's climate
prob = CDF_mm_dict["target"][target_index]

# Calculate probabilites in warmer/colder climates
probabilities = functions.calculate_probabilities(pwarm, target_index, CDF_mm_dict, bootstrap_3Darr_dict)

# Get the temperature corresponding to the percentile in today's climate in the pre-industrial
# and optioanlly future and observed climates
percentile_temps = functions.get_percentile_temp_in_climates(prob, T_range, CDF_mm_dict)

# Calculate intensity change intervals if bootstrapping was appplied
if bootstrap_3Darr_dict is not None:
    dI_intervals = functions.calculate_dI_intervals(target_index, T_range, bootstrap_3Darr_dict)
else:
    dI_intervals = None

# Print the final attribution results
functions.print_attribution_results(attribution_cases, obs_source, station_meta, clim_var, doy_index, n_days, target_value, probabilities, percentile_temps, dI_intervals)

"""
################################################################
Plot the results:
First, plot a time-series of observations and pseudo-observations.
Second, plot all observations and selected quantile functions.
Third, plot PDFs corresponding to different climates.
Fourth, plot model statistics.
################################################################
"""

# Plot the time-series
functions.plot_time_series(path2figures, clim_var, station_meta["name"], target_year, obs_df, pseudo_obs_mm_dict["target"], pseudo_obs_sm_dict["target"], doy_index, n_days)

# Plot observations and quantile functions
functions.plot_observations(path2figures, clim_var, station_meta["name"], target_year, obs_df, qr_obs_mm_dict["obs"], doy_index, n_days)

# Plot distributions
functions.plot_distributions(path2figures, attribution_cases, obs_source, station_meta, ssp, T_range, doy_index, n_days, pwarm, target_value, pseudo_obs_mm_dict["target"], PDF_mm_dict, probabilities, percentile_temps, dI_intervals)

# Plot model statistics onyl if bootstrapping is applied
if n_boots > 0:
    model_names = qr_sm_dict["preind"].columns.tolist()
    functions.plot_model_statistics(path2figures, obs_source, station_meta, doy_index, n_days, target_year, model_names, n_boots, target_index, target_value, pwarm, bootstrap_3Darr_dict,\
                                    probabilities, percentile_temps, dI_intervals)
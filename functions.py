#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Antti Toropainen, Mika Rantanen
"""

# Import needed libraries and tools
import numpy as np
import pandas as pd
import xarray as xr
import os, glob, re
from pathlib import Path
#from datetime import date, datetime, timedelta
import datetime as dt
import xml.etree.ElementTree as ET
from fmiopendata.wfs import download_stored_query
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import sys
import concurrent.futures
import requests
from io import StringIO
from tqdm import tqdm



############################################################### Functions for handling dates ###################################################################

# Check that the target date is valid
def check_date(month:int, day:int) -> None:
    """
    Check that the target date is valid. Abort the program if the date is invalid.
    
    Parameters:
        month (int): The target month (1-12).
        day (int): The target day (1-31, depending on the month).
    
    Returns:
        None
    """
    
    try:
        # Construct a date for a non-leap year
        date_obj = dt.date(2001,month,day)

    except ValueError:
        # If the date is not valid, the program aborts
        print(f"Error: Invalid date {month:02d}-{day:02d}")
        sys.exit(1)

    # Check if the date is leap day
    if month == 2 and day == 29:
        print("Error: Leap days (Feb 29) are not allowed.")
        sys.exit(1)

def doy(month:int, day:int) -> int:
    """ 
    Returns day of year for a given month and day in the case of non-leap year
    See: Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7
    Parameters:
        month (int): 1-12
        day (int): 1-31
    Returns:
        doy (int): day of year
    """
    K = 2 # Only non-leap years are considered
    doy = int((275 * month) / 9.0) - K * int((month + 9) / 12.0) + day - 30
    return doy

def get_doy_index_and_ndays(start_month:int, start_day:int, end_month:int|None=None, end_day:int|None=None) -> tuple[int,int]:

    """ 
    Calculates the day of year index 'doy_index' (center day for off number days and last day for even number days) of the time period
    AND the number of days between two dates (start date and end date). Works for all 1-31 day long periods.
    Parameters:
        start_month (int): 1-12
        start_day (int): 1-31
        end_month (int): 1-12 (Optional)
        end_day (int): 1-31 (Optional)
    Returns:
        doy (int), n_days (int): day of year index, and number of days in the time period
    """


    # Single-day attribution
    if end_month is None and end_day is None:
        # Check that the given date is valid
        check_date(start_month, start_day)
        
        # Determine the index (day of year)
        doy_index = doy(start_month,start_day)
        n_days = 1

    # Multi-day attribution
    else:
        # Check that the given dates are valid
        check_date(start_month, start_day)
        check_date(end_month, end_day)

        # Start and end days of year
        start_doy = doy(start_month,start_day)
        end_doy = doy(end_month,end_day)

        # A range of days that extends from Dec to Jan
        if start_doy > end_doy:
            n_days = (365 - start_doy + 1) + end_doy
            
            # Odd days
            if n_days % 2 == 1:
                doy_index = end_doy - n_days // 2
            # Even days
            else:
                doy_index = end_doy

        # Other ranges of days within the calendar year
        else:
            n_days = end_doy - start_doy + 1 
            
            # Odd days
            if n_days % 2 == 1:
                doy_index = end_doy - n_days // 2
            # Even days
            else:
                doy_index = end_doy
        
    return doy_index, n_days

def doy_to_mm_dd(day_of_year: int) -> str:
    """Converts day of year to MM-DD form for a non-leap year."""
    return (dt.datetime(2001, 1, 1) + dt.timedelta(days=int(day_of_year) - 1)).strftime("%m-%d")

def get_date_strings(doy_index:int, n_days:int) -> tuple[str,str]:

    """Constructs a date string for figures and figure names
    
    Parameters:
        doy_index: int
            day of year index
        n_days: int
            Number of days in the time-period
    
    Returns:
        tuple[str, str]
            - Time-period string, displayed in the figure (e.g. 1-5 Jan)
            - Time-period string, for figure name (e.g. MMDD_MMDD)

    """
    
    months = {
        "01": "Jan",
        "02": "Feb",
        "03": "Mar",
        "04": "Apr",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "Aug",
        "09": "Sep",
        "10": "Aug",
        "11": "Nov",
        "12": "Dec"
        }
    
    # n-day mean case
    if n_days > 1:
        if n_days % 2 == 1:
            half_window = n_days // 2
            start_doy = doy_index - half_window
            end_doy = doy_index + half_window
        else:
            start_doy = doy_index - n_days + 1
            end_doy = doy_index

        # Convert days of year to "MM-DD" dates
        start_doy = (start_doy - 1) % 365 + 1
        end_doy = (end_doy - 1) % 365 + 1
        
        # Convert doys to "MM-DD" form
        start_mmdd = doy_to_mm_dd(start_doy)
        end_mmdd = doy_to_mm_dd(end_doy)

        # Extract months and days
        start_month = start_mmdd[:2]
        end_month = end_mmdd[:2]
        start_day = start_mmdd[-2:].lstrip("0")
        end_day = end_mmdd[-2:].lstrip("0")

        # Construct a datestring for the figure name
        figname_date_string = f"{start_mmdd.replace('-', '')}-{end_mmdd.replace('-', '')}"

        # Return the time period 
        if start_month == end_month:
            return f"{start_day} – {end_day} {months[start_month]}", figname_date_string
        else:
            return f"{start_day} {months[start_month]} – {end_day} {months[end_month]}", figname_date_string
    
    # Single-day case
    else:
        mmdd = doy_to_mm_dd(doy_index)
        month = mmdd[:2]
        day = mmdd[-2:].lstrip("0")

        figname_date_string = f"{mmdd.replace('-', '')}"

        return f"{day} {months[month]}", figname_date_string

#################################################################################################################################################################

############################################## Functions for reading and manipulating the input data ############################################################

# Regression coefficients for multi-model mean (all the days of a calendar year) 
def read_coeffs_model_mean(input_data_dir: str, clim_var:str, ssp:str, obs_lat:float, obs_lon:float, n:int) -> xr.Dataset:
    
    """
    Read multi-model mean coefficients corresponding to a station's coordinates (latitude, longitude) and a given time-period.

    Parameters:
        input_data_dir: str 
            location of the different subdirectorties which contain input data
        clim_var: str 
            Climate variable (tas, tasmax, tasmin).
        ssp: str
            Emission scenario.
        obs_lat: float
          Latitude of the weather station.

        obs_lon: float
            Longitude of the weather station.
        n: int
            Running mean of n days

    Returns:
        xr.Dataset: Coefficients for the specified location from the multi-model mean dataset.
    """

    # Search for the multi-model mean file corresponding to climate variable and ssp
    pattern = os.path.join(input_data_dir, "regression_coefficients", "model_mean", f"a_{clim_var}_multi-modelmean_*_{ssp}_nday_means.nc")
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    
    # If Exract filename
    file_path = matches[0]
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")
    
    # Open the multi-model-mean dataset
    ds_mm = xr.open_dataset(file_path)
    
    # Extract the list corresponding to all the cases of n-day running means
    ndays = ds_mm.coords["ndays"].values

    # Extract local coefficients for the case where n is in ndays
    if n in ndays:
        ds_local_coeffs = ds_mm.sel(lat=obs_lat, lon=obs_lon, ndays=n, method='nearest')

    # Interpolate local coefficients for the case where n is not in ndays 
    else:
        ds_local_coeffs = ds_mm.sel(lat=obs_lat, lon=obs_lon, method='nearest')
        ds_local_coeffs = ds_local_coeffs.interp(ndays=n)

    # Coordinates are shifted to the right so that they match a trailing running mean of n
    if n % 2 == 0:
        ds_local_coeffs = ds_local_coeffs.roll(time = n//2, roll_coords=True)

    return ds_local_coeffs

# Coefficients for single models (all the days of a calendar year)
def read_coeffs_single_models(input_data_dir: str, clim_var:str, ssp:str, obs_lat:float, obs_lon:float, n:int) -> xr.Dataset:
    
    """
    Read single-model coefficients corresponding to a station's coordinates (latitude, longitude) and a given time-period.

    Parameters:
        input_data_dir: str 
            location of the different subdirectorties which contain input data
        clim_var: str 
            Climate variable (tas, tasmax, tasmin).
        ssp: str
            Emission scenario.
        obs_lat: float
          Latitude of the weather station.

        obs_lon: float
            Longitude of the weather station.
        n: int
            Running mean of n days

    Returns:
        xr.Dataset: Coefficients for the specified location from the multi-model mean dataset.
    """

    # Search for the multi-model mean file corresponding to climate variable and ssp
    files = glob.glob(os.path.join(input_data_dir, "regression_coefficients", "single_models", f'a_{clim_var}_mean_var_*_{ssp}_*days.nc'))
    #matches = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No files found for climate variable: {clim_var} and emission scenario: {ssp} in {input_data_dir}")

    # Extract ndays from filenames
    ndays_map = {}
    for f in files:
        m = re.search(r"_(\d+)days\.nc", f)
        if m:
            ndays_map[int(m.group(1))] = f

    available_ndays = sorted(ndays_map.keys())

    # If n correspond to a ndays in the filename
    if n in available_ndays:
        return xr.open_dataset(ndays_map[n]).sel(lat=obs_lat, lon=obs_lon, method="nearest")

    # Interpolate between closest values
    else:
        
        lower = max([d for d in available_ndays if d < n], default=None)
        upper = min([d for d in available_ndays if d > n], default=None)

        if lower is None or upper is None:
            raise ValueError(f"Number of days must be in 1...31 range.")

        # Open the two datasets
        ds_lower = xr.open_dataset(ndays_map[lower]).sel(lat=obs_lat, lon=obs_lon, method="nearest")
        ds_upper = xr.open_dataset(ndays_map[upper]).sel(lat=obs_lat, lon=obs_lon, method="nearest")

        # Add ndays coordinate
        ds_lower = ds_lower.expand_dims(ndays=[lower])
        ds_upper = ds_upper.expand_dims(ndays=[upper])

        # Concatenate and interpolate
        ds_both = xr.concat([ds_lower, ds_upper], dim="ndays")
        ds = ds_both.interp(ndays=n)

        # Coordinates are shifted to the right so that they match a trailing running mean of n
        if n % 2 == 0:
            ds = ds.roll(time=n//2, roll_coords=True)

        return ds


# Read the simulated temperature for multi-model mean
def read_sim_temp_model_mean(input_data_dir: str, clim_var:str, ssp:str) -> pd.DataFrame:

    """
    Reads simulated global temperature data for multi-model mean and combines it with the historically observed global mean temperature. The combined
    time-series is smoothed with a 11-year running mean.
    
    Parameters:
        input_data_dir: str 
            Path to the directory where subdirectories of different input data is stored
        clim_var: str 
            Climate variable (tas, tasmax, tasmin).
        ssp: str
            The Shared Socioeconomic Pathway emission scenario (e.g., "ssp245").
        
    Returns:
        sm_df: pd.DataFrame 
            Smoothed global multi-model mean temperature time-series (rows are years, columns are models), adjusted by baseline year 2000.
    """

    try:

        # Create for paths to files
        pattern_simulated_ts = os.path.join(input_data_dir, "simulated_temp_time_series", f'{clim_var}_Tglob_{ssp}_multi-model_mean_*.nc')
        pattern_obs_ts = os.path.join(input_data_dir, "HadCRUT5_global_annual.nc")
        
        # Search for matches
        matches_simulated_ts = glob.glob(pattern_simulated_ts)
        matches_obs_ts = glob.glob(pattern_obs_ts)
        
        # Check if the matches exits
        if not matches_simulated_ts:
            raise FileNotFoundError(f"No files found for pattern: {pattern_simulated_ts}")
        
        if not matches_obs_ts:
            raise FileNotFoundError(f"No files found for pattern: {pattern_obs_ts}")
        
        # If Exract filename
        path2simulated_ts = matches_simulated_ts[0]
        path2obs_ts = matches_obs_ts[0]
        
        # Check that the files exists
        if not os.path.exists(path2simulated_ts):
            raise FileNotFoundError(f"Error: The file {path2simulated_ts} was not found.")
    
        if not os.path.exists(path2simulated_ts):
            raise FileNotFoundError(f"Error: The file {path2obs_ts} was not found.")
    
        # Open the simulated temperature file with xarray and rename the variable "dt" to "t" 
        tglob_sim_ds = xr.open_dataset(path2simulated_ts)
        tglob_sim_temp_ds = tglob_sim_ds.dt.squeeze().rename('t')
        tglob_sim_temp_ds['time'] = tglob_sim_ds.time.dt.year
        
        # Smooth the simulated temperatures with 11-year running mean
        tglob_sim_smooth_ds = tglob_sim_temp_ds.rolling(time=11, center=True, min_periods=1).mean()
        
        # Open the observed global mean temperature file
        glob_obs_temp_ds = xr.open_dataset(path2obs_ts)

        # Extract the mean temperature from the dataset from 1850 to the present day
        glob_obs_temp = glob_obs_temp_ds['tas_mean'].squeeze().sel(time=slice('1850-01-01','2024-12-31'))
        glob_obs_temp['time'] = glob_obs_temp.time.dt.year

        # Compute 11-year moving average of the global mean temperature
        glob_obs_smooth_ds = glob_obs_temp.rolling(time=11, center=True, min_periods=1).mean()
        
        # Find the last valid year. The last year is the last index minus 5 as that's half of 11
        last_valid_idx = glob_obs_smooth_ds.time.values[-1] - 5
            
        # Merge observed and simulated global temperatures
        diff = glob_obs_smooth_ds.sel(time=last_valid_idx) - tglob_sim_smooth_ds.sel(time=last_valid_idx)        
        glob_temp_smooth_ds = xr.concat([glob_obs_smooth_ds.sel(time=slice(None, last_valid_idx)), tglob_sim_smooth_ds.sel(time=slice(last_valid_idx + 1, None))+diff], dim='time')

        # Select year 2000 as baseline
        glob_temp_smooth_ds = glob_temp_smooth_ds - glob_temp_smooth_ds.sel(time=2000)

        # Extract the observations to a dataframe
        glob_temp_smooth_df = glob_temp_smooth_ds.astype(float).drop_vars(('lat','lon')).to_pandas()

        return glob_temp_smooth_df
    
    # Return None with other exceptions
    except Exception as e:
        print(f"An unexected error occurred: {e}")
        return None

def read_sim_temp_single_models(input_data_dir: str, clim_var:str, ssp:str) -> pd.DataFrame:

    """
    Reads simulated global temperature data from single models and combines it with the historically observed global mean temperature. The combined
    time-series is smoothed with a 11-year running mean.
    
    Parameters:
        input_data_dir: str 
            Path to the directory where subdirectories of different input data is stored
        clim_var: str 
            Climate variable (tas, tasmax, tasmin).
        ssp: str
            The Shared Socioeconomic Pathway emission scenario (e.g., "ssp245").
        
    Returns:
        sm_df: pd.DataFrame 
            Smoothed global temperature time-series of individual models (rows are years, columns are models), adjusted by baseline year 2000.
    """

    try:

        # Create unix-style wild card patterns 
        pattern_simulated_ts = os.path.join(input_data_dir, "simulated_temp_time_series", f"{clim_var}_Tglob_{ssp}_[0-9]*mod_*.nc")
        pattern_obs_ts = os.path.join(input_data_dir, "HadCRUT5_global_annual.nc")
        
        # Search the patterns
        matches_simulated_ts = glob.glob(pattern_simulated_ts)
        matches_obs_ts = glob.glob(pattern_obs_ts)
        
        # Check if the matches exits
        if not matches_simulated_ts:
            raise FileNotFoundError(f"No files found for pattern: {pattern_simulated_ts}")
        
        if not matches_obs_ts:
            raise FileNotFoundError(f"No files found for pattern: {pattern_obs_ts}")
        
        # Form paths to files
        path2simulated_ts = matches_simulated_ts[0]
        path2obs_ts = matches_obs_ts[0]
        
        # Check that the files exists
        if not os.path.exists(path2simulated_ts):
            raise FileNotFoundError(f"Error: The file {path2simulated_ts} was not found.")
    
        if not os.path.exists(path2simulated_ts):
            raise FileNotFoundError(f"Error: The file {path2obs_ts} was not found.")

        # Open the files with xarray
        single_models_ds = xr.open_dataset(path2simulated_ts)
        glob_obs_temp_ds = xr.open_dataset(path2obs_ts)

        # Store the single-models DataFrames here
        sm_dict = {}

        # Take the mean temperature from the dataset from pre-industrial time to the present day
        glob_obs_temp = glob_obs_temp_ds['tas_mean'].squeeze().sel(time=slice('1850-01-01','2024-12-31'))
        glob_obs_temp['time'] = glob_obs_temp.time.dt.year

        # Smooth the observed temperature with 11-year rolling mean
        glob_obs_smooth_ds = glob_obs_temp.rolling(time=11, center=True, min_periods=1).mean()
        
        # The 5th last year  
        last_valid_idx = glob_obs_smooth_ds.time.values[-1] - 5

        # Loop through each model
        for var_name, da in single_models_ds.data_vars.items():
            
            if var_name.startswith("time"):
                continue
            else:
            
                # Read the simulated temperature
                tglob_single_ds = da.squeeze().rename("t")
                tglob_single_ds["time"] = tglob_single_ds.time.dt.year

                # Take an 11-year running mean of the simulated temperature
                tglob_single_smooth = tglob_single_ds.rolling(time=11, center=True, min_periods=1).mean()

                # Merge observed and simulated global temperatures
                diff = glob_obs_smooth_ds.sel(time=last_valid_idx) - tglob_single_smooth.sel(time=last_valid_idx)        
                glob_temp_smooth_ds = xr.concat([glob_obs_smooth_ds.sel(time=slice(None, last_valid_idx)), tglob_single_smooth.sel(time=slice(last_valid_idx + 1, None))+diff], dim='time')
                
                # Select year 2000 as baseline
                glob_temp_smooth_ds = glob_temp_smooth_ds - glob_temp_smooth_ds.sel(time=2000)

                # Convert to pandas DataFrame
                glob_temp_smooth_df = glob_temp_smooth_ds.astype(float).drop_vars(('lat','lon')).to_pandas()

                # Store the resulting DataFrame in the dictionary
                sm_dict[var_name] = glob_temp_smooth_df

        # Combine all variables into a single DataFrame
        combined_df = pd.concat(sm_dict, axis=1)

        return combined_df
    
    # Return None if the file is not found
    except FileNotFoundError as e:
        print(e)
        return None 
    
    # Return None with all other possible errors
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return None 

def read_daily_obs_from_FMI(station_id: int, clim_var: str):

    """
    Retrieve daily temperature observations (mean, max, or min) from the FMI API for a single FMI station from the earliest available date up to today.
    Retrieve the station's metadata (name, lat, lon).
    Requires a Python Interface fmiopendata. See the description in Github: https://github.com/pnuu/fmiopendata

    Parameters
        station_id : int or str
            FMI station ID (e.g., 100971 for Helsinki Kaisaniemi)
        clim_var : str
            Climate variable (tas, tasmax, tasmin)

    Returns    
        pandas.DataFrame
            Pivot table where years are rows and days of year are columns.
        dict
            Contains station metadata
    """

    clim_var_map = {"tas": "tday", "tasmax": "tmax", "tasmin": "tmin"}
    if clim_var not in clim_var_map:
        raise ValueError("clim_var must be one of 'tas', 'tasmax', or 'tasmin'")
    parameter = clim_var_map[clim_var]

    url = "https://opendata.fmi.fi/wfs"

    # Take observations up to yesterday
    end_date = dt.date.today() - dt.timedelta(days=1)

    # --------------------------------------------------
    # STEP 0: Fetch station metadata
    # --------------------------------------------------
    meta_params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::ef::stations",
        "fmisid": station_id,
    }

    r_meta = requests.get(url, params=meta_params)
    if r_meta.status_code != 200:
        raise RuntimeError("Failed to fetch station metadata.")

    root_meta = ET.fromstring(r_meta.content)

    ns = {
        "gml": "http://www.opengis.net/gml/3.2",
        "ef": "http://inspire.ec.europa.eu/schemas/ef/4.0",
        "base": "http://inspire.ec.europa.eu/schemas/base/3.3",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    # Station name
    name_el = root_meta.find(".//gml:name", ns)
    station_name = name_el.text if name_el is not None else None

    # Coordinates
    pos = root_meta.find(".//gml:pos", ns)
    
    lat = lon = None
    if pos is not None:
        lat, lon = map(float, pos.text.split())

    station_meta = {
        "station_id": station_id,
        "name": station_name,
        "latitude": lat,
        "longitude": lon
    }

    # --- Step 1: Detect earliest available date ---
    #print(f"Finding the earliest available date for {clim_var}...")
    found = False
    test_year = 1850
    while test_year <= end_date.year:
        test_params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::observations::weather::daily::timevaluepair",
            "fmisid": station_id,
            "parameters": parameter,
            "starttime": f"{test_year}-01-01T00:00:00Z",
            "endtime": f"{test_year+1}-01-01T00:00:00Z",
        }
        r = requests.get(url, params=test_params)
        if r.status_code == 200 and b"<wml2:value>" in r.content:
            root = ET.fromstring(r.content)
            ns = {"wml2": "http://www.opengis.net/waterml/2.0"}
            times = [el.text for el in root.findall(".//wml2:time", ns)]
            if times:
                start_date = dt.datetime.fromisoformat(times[0].replace("Z", "+00:00")).date()
                #print(f"Earliest available date: {start_date}")
                found = True
                break
        test_year += 1

    if not found:
        raise RuntimeError("Could not determine earliest available date automatically.")

    # --- Step 2: Fetch data year by year ---
    #print(f"Downloading {clim_var} data from {start_date} to {end_date} for station {station_id}...")
    all_data = []
    year = start_date.year
    while year <= end_date.year:
        y1 = dt.date(year, 1, 1)
        y2 = dt.date(year + 1, 1, 1)
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "getFeature",
            "storedquery_id": "fmi::observations::weather::daily::timevaluepair",
            "fmisid": station_id,
            "parameters": parameter,
            "starttime": y1.strftime("%Y-%m-%dT00:00:00Z"),
            "endtime": y2.strftime("%Y-%m-%dT00:00:00Z"),
        }
        r = requests.get(url, params=params)
        if r.status_code != 200:
            print(f"Skipping year {year} (HTTP {r.status_code})")
            year += 1
            continue

        root = ET.fromstring(r.content)
        ns = {"wml2": "http://www.opengis.net/waterml/2.0"}
        times = [el.text for el in root.findall(".//wml2:time", ns)]
        values = [el.text for el in root.findall(".//wml2:value", ns)]
        for t, v in zip(times, values):
            try:
                date = dt.datetime.fromisoformat(t.replace("Z", "+00:00")).date()
                all_data.append((date, float(v)))
            except Exception:
                continue
        year += 1

    # --- Step 3: Process and pivot ---
    obs = pd.DataFrame(all_data, columns=["date", "temperature"])
    if obs.empty:
        raise RuntimeError("No valid data returned by FMI.")

    obs["date"] = pd.to_datetime(obs["date"])

    # Remove duplicates
    obs = (obs.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True))

    # Remove leap days
    obs = obs[~((obs["date"].dt.month == 2) & (obs["date"].dt.day == 29))]

    # Set a non-leap reference year and convert all dates to days of year
    ref_year = 2001
    obs["month_day"] = obs["date"].dt.strftime("%m-%d")
    obs["ref_date"] = pd.to_datetime(f"{ref_year}-" + obs["month_day"])
    obs["day_of_year"] = obs["ref_date"].dt.dayofyear
    obs["year"] = obs["date"].dt.year

    # Pivot (year x day_of_year)
    pivot_df = obs.pivot(index="year", columns="day_of_year", values="temperature")
    pivot_df = pivot_df.reindex(columns=range(1, 366))  # ensure full 365 days

    return pivot_df, station_meta

def read_daily_obs_from_FROST(frost_client_id: str, metnosid: str, clim_var:str, homogenised=True):
    
    """
    Retrieve daily temperature data (mean, max, or min) from the Frost API.

    @author: Amalie Skålevåg (amalie.skalevag@met.no), Herman F. Fuglestvedt

    Parameters
        frost_client_id : str
            Client ID for Frost API authentication (see https://frost.met.no/howto.html).
        metnosid : str
            Station ID for a Norwegian weather station (e.g. 'SN18700').
        clim_var : {'mean', 'max', 'min'}, optional
            Which daily temperature to retrieve:
            - 'tas' → daily mean temperature
            - 'tasmax' → daily maximum temperature
            - 'tasmin' → daily minimum temperature
        homogenised : bool, optional
            Whether to use homogenised temperature series (default=True).
            Homogenised data are often longer and quality controlled.

    Returns
        pandas.DataFrame
            A pivoted DataFrame of daily temperature time series (°C) with year as a row and day_of_year as columns.
    """

    # Determine the correct Frost variable
    if clim_var == "tas":
        frost_var = "mean"
    elif clim_var == "tasmax":
        frost_var = "max"
    elif clim_var == "tasmin":
        frost_var = "min"
    else:
        raise ValueError("clim_var has to be 'tas', 'tasmax' or 'tasmin'.")

    # Validate variable
    valid_vars = {"mean", "max", "min"}
    if frost_var not in valid_vars:
        raise ValueError(f"variable must be one of {valid_vars}, got '{frost_var}' which corresponds to '{clim_var}'")

    # determine variable name
    if homogenised:
        var_name = f"best_estimate_{frost_var}(air_temperature P1D)"
    else:
        var_name = f"{frost_var}(air_temperature P1D)"

    # Define endpoint and parameters
    endpoint = "https://frost.met.no/observations/v0.jsonld"
    parameters = {
        "sources": metnosid,
        "elements": var_name,
        "referencetime": f"1850-01-01/{pd.Timestamp.utcnow().strftime('%Y-%m-%d')}",
        "timeoffsets": "default",
        "levels": "default",
        "qualities": "0,1,2,3,4",
    }

    r = requests.get(endpoint, params=parameters, auth=(frost_client_id, ""))

    #print("Status code:", r.status_code)
    #print("Response text preview:")
    #print(r.text[:500])

    if r.status_code != 200:
        print("Error! Returned status code %s" % r.status_code)
        print("Message:", r.text)
        raise RuntimeError("Frost API request failed.")

    json_data = r.json()
    data = json_data["data"]

    # Extract JSON data
#    json = r.json()

    # Check if the request worked, print out any errors
 #   if r.status_code == 200:
  #      data = json["data"]
   # else:
    #    print("Error! Returned status code %s" % r.status_code)
     #   print("Message: %s" % json["error"]["message"])
      #  print("Reason: %s" % json["error"]["reason"])
       # raise RuntimeError(f'{json["error"]["message"]}. {json["error"]["reason"]}')

    # Create DataFrame from list of dictionaries
    df = pd.concat([pd.DataFrame(data[i]["observations"], index=[pd.to_datetime(data[i]["referenceTime"])]) for i in range(len(data))])
    # sort chronologically
    df.sort_index(inplace=True)
    # index to dates
    df.index = pd.to_datetime(df.index.date)
    df.rename(columns={"value": "temperature"}, inplace=True)
    df["date"] = df.index

    # Remove Feb 29th (non-leap-year alignment)
    df = df[~((df.index.month == 2) & (df.index.day == 29))]

    # Non-leap year for calendar
    ref_year = 2001

    # Format month and day as strings like 'MM-DD'
    df['month_day'] = df['date'].dt.strftime('%m-%d')

    # Construct a new reference date using string concatenation
    df['ref_date'] = pd.to_datetime(f"{ref_year}-" + df['month_day'])

    # Extract day-of-year from the reference date
    df['day_of_year'] = df['ref_date'].dt.dayofyear

    # Sort the observations by date
    df = df.sort_values("date")

    # Add year and day_of_year columns
    df["year"] = df.index.year

    # Create pivot table with year as index and day_of_year as columns
    pivot_df = df.pivot(index='year', columns='day_of_year', values="temperature")

    return pivot_df

def get_FROST_station_metadata(frost_client_id: str, metnosid: str):
    
    """
    Fetch station metadata (name, latitude, longitude, and  the first year of available observations) from the Frost API.

    Parameters:
        frost_client_id: str
            Client ID for Frost API authentication (see https://frost.met.no/howto.html).
        metnosid: int
            Station ID for a Norwegian weather station
        
    Returns: dict
        Contains station metadata
    """

    
    endpoint = "https://frost.met.no/sources/v0.jsonld"
    params = {"ids": metnosid}

    r = requests.get(endpoint, params=params, auth=(frost_client_id, ""))
    js = r.json()

    if r.status_code != 200:
        raise RuntimeError(
            f"Error {r.status_code}: {js.get('error', {}).get('message')}"
        )

    item = js["data"][0]

    name = item.get("name")

    # Frost gives coordinates in GeoJSON format: [lon, lat]
    lon, lat = item.get("geometry", {}).get("coordinates", [None, None])

    # --- Extract start year (validFrom) ---
    valid_from_str = item.get("validFrom")
    if valid_from_str:
        try:
            start_year = dt.datetime.fromisoformat(valid_from_str.replace("Z", "")).year
        except Exception:
            start_year = None
    else:
        start_year = None

    return {
        "station_id": metnosid,
        "name": name,
        "latitude": lat,
        "longitude": lon,
        "start_year": start_year,
    }

def read_daily_obs_from_SMHI(station_id: int, clim_var: str, period: str) -> pd.DataFrame:
    """
    Download local daily temperature observations from SMHI API.
    
    Parameters:
        station_id: int
            ID-code for a Swedish station (e.g. 161790 for Piteå)
        clim_car: str
            climate_variable (tas, tasmax, tasmin)
        period: str
            'hist' or 'recent': specifies whether quality controlled historical or more recent observations are downloaded
    
    Returns:
        daily_obs: pd.DataFrame
            Contains daily temperature observations with columns ['date', 'temperature']
    """

    period_map = {"hist":"corrected-archive","recent":"latest-months"}
    if period not in period_map:
        raise ValueError("source must be 'hist' or 'recent'.")
    period_id = period_map[period]

    var_map = {"tas": 2, "tasmax": 20, "tasmin": 19}
    if clim_var not in var_map:
        raise ValueError("clim_var must be one of 'tas', 'tasmax', or 'tasmin'")
    parameter_id = var_map[clim_var]

    url = (
        f"https://opendata-download-metobs.smhi.se/api/version/1.0/"
        f"parameter/{parameter_id}/station/{station_id}/period/{period_id}/data.csv"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    lines = response.text.splitlines()
    header_index = next(i for i, line in enumerate(lines) if "Datum" in line)
    csv_data = "\n".join(lines[header_index:])
    df = pd.read_csv(StringIO(csv_data), sep=";", engine="python", on_bad_lines="skip")
    df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]

    date_col = [c for c in df.columns if "datum" in c.lower()][0]
    temp_col = [c for c in df.columns if "temperatur" in c.lower()][0]
    df.rename(columns={date_col: "date", temp_col: "temperature"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df.dropna(subset=["date", "temperature"], inplace=True)

    daily_obs = df.groupby(df["date"].dt.date)["temperature"].mean().reset_index()
    daily_obs["date"] = pd.to_datetime(daily_obs["date"])
    return daily_obs

def get_SMHI_station_metadata(
    station_id: int,
    clim_var: str,
    strict: bool = False,
) -> dict:
    """
    Fetch SMHI station metadata (name, lat, lon) from SMHI API.

    Parameters:
        station_id: int
            ID-code for a Swedish station (e.g. 161790 for Piteå)
        clim_var: str
            climate variable (tas, tasmax, or tasmin)
        strict: bool,
            If station is not available for this parameter, returns None values, unless strict=True.
    
    Returns: dict
        Contains station metadata
    """

    clim_var_map = {"tas": 2, "tasmax": 20, "tasmin": 19}
    if clim_var not in clim_var_map:
        raise ValueError("clim_var must be 'tas', 'tasmax', or 'tasmin'")

    parameter_id = clim_var_map[clim_var]

    url = (
        f"https://opendata-download-metobs.smhi.se/api/version/1.0/"
        f"parameter/{parameter_id}.json"
    )

    r = requests.get(url)
    r.raise_for_status()
    stations = r.json().get("station", [])

    station = next(
        (s for s in stations if int(s["id"]) == int(station_id)),
        None,
    )

    if station is None:
        if strict:
            raise ValueError(
                f"Station {station_id} not found for parameter {clim_var}"
            )
        else:
            return {
                "station_id": station_id,
                "name": None,
                "latitude": None,
                "longitude": None
            }

    return {
        "station_id": station_id,
        "name": station.get("name"),
        "latitude": station.get("latitude"),
        "longitude": station.get("longitude")
    }

def concatenate_SMHI_observations(obs1:pd.DataFrame, obs2:pd.DataFrame)->pd.DataFrame:

    """
    Concatenates the time-series of observations from two nearby weather stations to a single time-series. 
    
    Parameters:
        obs1: pd.DataFrame
            Contains the OLDER temperature observations with columns ['date', 'temperature']
        obs2: pd.DataFrame
            Contains the NEWER temperature observations with columns ['date', 'temperature']
    Returns:
        obs_combined: pd.DataFrame
            Combined time-series of observations from two nearby stations with columns ['date', 'temperature']
    """

    # Get the last date from the OLDER observations time-series
    last_date = obs1["date"].max()

    # Include only observation after the last date in the 2nd observation time series
    obs2_included = obs2[obs2["date"] > last_date]

    # Concatenate the observation time series
    obs_combined = pd.concat([obs1, obs2_included], ignore_index=True)

    return obs_combined

def pivot_SMHI_obs(SMHI_obs: pd.DataFrame) -> pd.DataFrame:
    
    """
    Convert SMHI observation DataFrame (columns ['date', 'temperature']) into a pivot table where years are rows and day_of_year are coluns. 
    Lead day observations are removed.
    
    Parameters:
        SMHI_obs: pd.DataFrame
            Contains temperature observations in the form ['date', 'temperature']
    Returns:
        pivot_df: pd.DataFrame
            Contains the same observations in a DataFrame where rows are years, and columns are day_of_year.
    """
    

    # Set the date to datetime form and sort the observations
    SMHI_obs = SMHI_obs.copy()
    SMHI_obs["date"] = pd.to_datetime(SMHI_obs["date"])
    SMHI_obs = SMHI_obs.sort_values("date").reset_index(drop=True)

    # Remove leap day observations
    SMHI_obs = SMHI_obs[~((SMHI_obs["date"].dt.month == 2) & (SMHI_obs["date"].dt.day == 29))]

    # Convert dates to days of year 
    ref_year = 2001  # non-leap reference year
    SMHI_obs["month_day"] = SMHI_obs["date"].dt.strftime("%m-%d")
    SMHI_obs["ref_date"] = pd.to_datetime(f"{ref_year}-" + SMHI_obs["month_day"])
    SMHI_obs["day_of_year"] = SMHI_obs["ref_date"].dt.dayofyear
    SMHI_obs["year"] = SMHI_obs["date"].dt.year

    # Create a pivot table (rows=yeras columns=day_of_year)
    pivot_df = SMHI_obs.pivot(index="year", columns="day_of_year", values="temperature")

    return pivot_df

def read_daily_obs(obs_source:str, clim_var:str, station_id:str, station2_id:str|None=None, frost_client_id:str|None=None)->tuple[pd.DataFrame, dict]:
    
    """
    Downloads local daily temperature (mean, max or min) observations of a given weather station from the FMI, SMHI or FROST API.

    Parameters:
        obs_source: str
            Acronym for observation source (FMI, SMHI or FROST)
        frost_client_id: str
            Client ID for Frost API authentication (see https://frost.met.no/howto.html).
        clim_var: str
            climate variable (tas, tasmax or tasmin)
        station_id: str
            Id-code for the weather station (e.g. 101932 for Sodankylä, Finland)
        station2_id: str
            Id-code of a second, optional weather station. This only applies to SMHI observations as constructing a full time-series requires concatenating observations from
            old weather stations with nearby newer ones. You should exercise caution when doing this. 
            
    Returns:
        daily_temp_obs_df: pd.DataFrame
            Contains daily temperature observations. Rows are years, and columns are 'day-of-year'.
        station_meta: dict
            Contains metadata of the weather station (name, lat, lon)
    """

    clim_var_map = {
        "tas": "daily mean temperature",
        "tasmax": "daily maximum temperature",
        "tasmin": "daily minimum temperature"
    }

    if station2_id is not None:
        station_str = f"stations {station_id} and {station2_id}"
    else:
        station_str = f"station {station_id}"

    print(f"Downloading {clim_var_map[clim_var]} observations for {obs_source} {station_str}...\n")

    # Read FMI observations
    if obs_source == "FMI":
        try:
            daily_temp_obs_df, station_meta = read_daily_obs_from_FMI(station_id, clim_var)
        except Exception as e:
            print(f"Downloading temperature observations from {obs_source} failed due to {e}.")
            print("Aborting...")
            sys.exit(1)

    # Read FROST observations
    elif obs_source == "FROST":
        try:
            daily_temp_obs_df = read_daily_obs_from_FROST(frost_client_id, station_id, clim_var, False)
            station_meta = get_FROST_station_metadata(frost_client_id,station_id)
        except Exception as e:
            print(f"Downloading temperature observations from {obs_source} failed due to {e}.")
            print("Aborting...")
            sys.exit(1)

    # Read SMHI observations
    elif obs_source == "SMHI":
        smhi_obs1_hist = read_daily_obs_from_SMHI(station_id, clim_var, "hist")
        station_meta = get_SMHI_station_metadata(station_id,clim_var,True)
        daily_temp_obs_df = pivot_SMHI_obs(smhi_obs1_hist)
        
        if station2_id:
            smhi_obs2_hist = read_daily_obs_from_SMHI(station2_id, clim_var, "hist")
            station_meta = get_SMHI_station_metadata(station2_id,clim_var,True)
            smhi_obs_combined = concatenate_SMHI_observations(smhi_obs1_hist,smhi_obs2_hist)
            daily_temp_obs_df = pivot_SMHI_obs(smhi_obs_combined)
    else:
        print("Observation source must be FMI, SMHI or FROST.")
        print("Aborting...")
        sys.exit(1)
        raise ValueError("Observation source must be FMI, SMHI or FROST.")
    
    # Get the first and last years of observations
    first_year = daily_temp_obs_df.index.min()
    last_year = daily_temp_obs_df.index.max()

    # Get the first and last days of year
    doy_first = daily_temp_obs_df.loc[first_year].first_valid_index()
    doy_last = daily_temp_obs_df.loc[last_year].last_valid_index()

    # Get the corresponding date strings
    date_first_string = doy_to_mm_dd(doy_first)
    date_last_string = doy_to_mm_dd(doy_last)

    # Pring the time-period over which observations are downloaded
    print(f"{daily_temp_obs_df.size} observations were downloaded from {date_first_string}-{first_year} to {date_last_string}-{last_year}.\n")

    return daily_temp_obs_df, station_meta


def check_obs_validity(obs_df:pd.DataFrame, doy_index:int, n_days:int, target_year:int)-> None:

    """
    Checks that the observation corresponding to a given doy_index and target_year is not NaN. If it is NaN, the program is aborted.

    Parameters:
        obs_df: pd.DataFrame
            Contains daily temperature observations, or their 2 to 31-day running means. Rows are years, and columns are 'day-of-year'.
        doy_index: int
            day_of_year index (day_of year for single days, centered day for odd-day running means and the last day of even-day running means)
        n_days: int
            Number of days in the time-period
        target_year: int
            Year of observations
            
    Returns:
        None
    """

    # Get the time period string for printing
    time_period, _ = get_date_strings(doy_index, n_days)

    if n_days == 1:
        period_str = "day"
    else:
        period_str = "period"

    if np.isnan(obs_df.loc[target_year][doy_index]):
        print(f"Observation for the {period_str} {time_period} {target_year} is NaN, attribution cannot be done.")
        print("Aborting...")
        sys.exit(1)
    else:
        print(f"Observation for the {period_str} {time_period} {target_year} is valid.")
        print("Proceeding with attribution...\n")
    
def melt_obs_df(daily_temp_obs_df:pd.DataFrame) -> pd.DataFrame:
    
    """
    Reshapes daily temperature data from wide format (years as rows, day_of_year as columns)  to long format with year, day_of_year, and temperature columns.

    Parameters:
        daily_temp_obs_df: pd.DataFrame
            Contains daily temperature observations. Rows are years, and columns are 'day-of-year'.

    Returns:
        melted_obs_df: pd.DataFrame
    """

    # Reset the index to include year as a column
    daily_temp_obs_df = daily_temp_obs_df.reset_index()
    
    # Melt the observation DataFrame
    melted_obs_df = daily_temp_obs_df.melt(id_vars=["year"], var_name="day_of_year", value_name="temperature")
    
    # Set day_of_year as an integer 
    melted_obs_df["day_of_year"] = melted_obs_df["day_of_year"].astype(int)

    # Drop NaN temperature observations
    melted_obs_df = melted_obs_df.dropna(subset=["temperature"])

    # Set 3 columns
    melted_obs_df = melted_obs_df[["year", "day_of_year", "temperature"]]

    # Sort by year and day_of_year
    melted_obs_df = melted_obs_df.sort_values(by=["year", "day_of_year"]).reset_index(drop=True)
    
    return melted_obs_df

def unmelt_obs_df(melted_obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverses the melt_obs() transformation by converting a long-format temperature DataFrame ('year', 'day_of_year', 'temperature') back to
    wide format with years as rows and days (1-365) as columns.

    Parameters
        long_df : pd.DataFrame
            Long-format DataFrame with columns year, day_of_year, temperature
        
    Returns
        unmelt_obs_df: pd.DataFrame
            Wide-format DataFrame with index: years, columns: day_of_year values (1-365), values: daily mean temperatures
    """

    # Convert years and day_of_years to int
    melted_obs_df = melted_obs_df.copy()
    melted_obs_df["year"] = melted_obs_df["year"].astype(int)
    melted_obs_df["day_of_year"] = melted_obs_df["day_of_year"].astype(int)

    # Convert the DataFrame to a pivoted form
    unmelted_obs_df = (melted_obs_df.pivot(index="year", columns="day_of_year", values="temperature").sort_index())

    # Sort the day_of_year columns (1..365)
    unmelted_obs_df = unmelted_obs_df.reindex(sorted(unmelted_obs_df.columns), axis=1)

    # Name the index and columns
    unmelted_obs_df.index.name = "year"
    unmelted_obs_df.columns.name = "day_of_year"

    return unmelted_obs_df

###########################################################################################################################################################################################

################################################### Modify observations to pseudo-observations ############################################################################################

def compute_running_mean(obs_df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    Compute an n-day running mean for daily temperature observations. Centered (trailing) running mean is applied for odd (even) days.
    n-day periods which contain NaNs are ignored.

    Arguments:
        obs_df: pd.DataFrame 
            Contains daily temperature observations. Rows are years, columns are "day_of_year".
        n_days: int
            Number of days in the running mean
    
    Returns:
        obs_nday_mean_df: pd.DataFrame
            Contains n-day means of daily temperature observations. Rows are years, columns are the corresponding day_of_year indices.
    """

    # Melt the pivoted DataFrame
    melted = (obs_df.reset_index().melt(id_vars="year", var_name="day_of_year", value_name="temperature"))
    melted["day_of_year"] = melted["day_of_year"].astype(int)

    # Create a continuous time index
    melted = melted.sort_values(["year", "day_of_year"]).reset_index(drop=True)

    # Centered running mean for odd days
    if n_days % 2 == 1:
        melted["temperature_rm"] = (melted["temperature"].rolling(window=n_days, center=True, min_periods=n_days).mean())

    # Trailing running mean for even days
    if n_days % 2 == 0:
        melted["temperature_rm"] = (melted["temperature"].rolling(window=n_days, center=False, min_periods=n_days).mean())

    # Replace original temperature with running mean
    melted["temperature"] = melted["temperature_rm"]
    melted = melted.drop(columns=["temperature_rm"])

    # Convert the melted DataFrame to a pivoted DataFrame
    obs_nday_mean_df = melted.pivot(index="year", columns="day_of_year", values="temperature")

    # Reindex to original structure
    return obs_nday_mean_df.reindex(index=obs_df.index, columns=obs_df.columns)

# A function that converts observations to pseudo-observations for single models
def modify_obs_single_models(
        obs_df:pd.DataFrame, 
        coeffs_sm_ds:xr.Dataset,
        glob_temp_sm_df:pd.DataFrame,
        target_year:int,
        doy_index:int|None=None)->pd.DataFrame: 
    
    """
    Modify observed temperatures for days corresponding to a given 'doy_index' OR for all days of the year using model-specific regression coefficients
    and save results in a Pandas DataFrame.

    Parameters:
        obs_df: pd.DataFrame 
            Contains observed daily temperatures or their 2 to 31-day running means. Rows are years, columns are "day_of_year".
        coeffs_sm_ds: xr.Dataset 
            Contains the model-specific regression coefficients B and D for each 'day_of_year' or for each 2 to 31-day running mean period in a year.
        glob_temp_sm_df: pd.DataFrame 
            Contains the observed (past) and model-specific simulated (future) 11-year running mean of the global mean temperature for each year.
        target_year: int 
            Year for which pseudo-observations are calculated.
        doy_index: int None|None
            day-of-year index. If given, pseudo-observations are only calculated for the observations corresponding to the given doy_index. Otherwise they're calculated for all
            days/n-day periods 
    
    Returns:
        pseudo_obs_df: pd.DataFrame 
            MultiIndex DataFrame with pseudo-observations for the n-day period corresponding to the given doy-index (other days are NaN values).
            MultiIndex DataFrame with pseudo-observations for all n-day periods of the year. 
            Outer level = model names, Inner level = years. Columns = day_of_year.
    """

    # A list to store data
    pseudo_obs_list = []

    # Get model names
    models = [var.split('_')[1] for var in coeffs_sm_ds.data_vars if var.startswith('B_')]
    glob_temp_sm_df.columns = models

    # Loop through models
    for model in models:

        # Get the global mean temperature for this model
        g = glob_temp_sm_df[model]

        # Smoothed global mean temperature for the target year
        gg = g.loc[target_year]
    
        # Extract coefficients B and D for a single model
        B = coeffs_sm_ds[f"B_{model}"]
        D = coeffs_sm_ds[f"D_{model}"]

        # Compute pseudo-observations only for the given doy_index
        if doy_index:
            
            # Select observations (single-day or running mean) corresponding to a day of year
            obs = obs_df[doy_index]

            # Model-specific regression coefficients corresponding to the given doy_index
            B_day = B.values[doy_index-1]
            D_day = D.values[doy_index-1]

            # Calculate intermediate values: z_obs(t0, t)
            z_obs = obs + (B_day * (g.loc[target_year] - g.loc[obs.index]))

            # Mean value, Z(t), for the reference period
            Z = z_obs.loc["1901":"2024"].mean() 

            # Calculate sqrt(R(t0, t))
            ratio = (1. + gg * D_day) / (1. + g * D_day)
            ratio = np.maximum(ratio, 0)  # Ensure ratio is non-negative
            R = np.sqrt(ratio)

            # Calculate pseudo-observations: y_obs
            y_obs = Z + (z_obs - Z) * R

            # Collect data as records
            for year, value in y_obs.loc[obs.index].items():
                pseudo_obs_list.append((year, doy_index, model, value))

        else:
            # Loop through each day of the year
            for day in obs_df.columns:

                # Select observations (single-day or running mean) corresponding to a day of year
                obs = obs_df[day]

                # Model-specific values of regression coefficients    
                B_day = B.values[day-1]
                D_day = D.values[day-1]

                # Calculate intermediate values: z_obs(t0, t)
                z_obs = obs + (B_day * (g.loc[target_year] - g.loc[obs.index]))

                # Mean value, Z(t), for the reference period
                Z = z_obs.loc["1901":"2024"].mean() 

                # Calculate sqrt(R(t0, t))
                ratio = (1. + gg * D_day) / (1. + g * D_day)
                ratio = np.maximum(ratio, 0)  # Ensure ratio is non-negative
                R = np.sqrt(ratio)

                # Calculate pseudo-observations: y_obs
                y_obs = Z + (z_obs - Z) * R

                # Append the data to a list
                for year, value in y_obs.loc[obs.index].items():
                    pseudo_obs_list.append((year, day, model, value))

    # Construct a Dataframe from the model-specific pseudo-obseravations
    pseudo_obs_df = pd.DataFrame(pseudo_obs_list, columns=["year", "day_of_year", "model", "pseudo_observation"])

    # Reshape the DataFrame to match the desired structure
    # Move Model to the MultiIndex and pivot DayOfYear to columns
    pseudo_obs_df = (
        pseudo_obs_df
        .set_index(["model", "year"])
        .pivot(columns="day_of_year", values="pseudo_observation")
    )

    return pseudo_obs_df

def melt_pseudo_obs_single_models(pseudo_obs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a dataset of pseudo-observations with MultiIndex (Model, Year) 
    and day_of_year as columns into a long-format DataFrame with columns:
    'model', 'year', 'day_of_year', 'temperature'.
    
    Parameters:
        pseudo_obs_df: pd.DataFrame
            Input dataset with MultiIndex (Model, Year) and columns as day_of_year.
        
    Returns:
        pd.DataFrame:
            Transformed DataFrame with columns 'model', 'year', 'day_of_year', 'temperature'.
    """
    # Reset the MultiIndex to make 'Model' and 'Year' regular columns
    pseudo_obs_df = pseudo_obs_df.reset_index()
    
    # Melt the DataFrame to long format
    melted_df = pseudo_obs_df.melt(
        id_vars=["Model", "Year"],
        var_name="day_of_year",
        value_name="temperature"
    )
    
    # Set 'day_of_year' column values integers
    melted_df["day_of_year"] = melted_df["day_of_year"].astype(int)

    # Drop NaN values in the temperature column
    melted_df = melted_df.dropna(subset=["temperature"])

    # Rename columns for clarity
    melted_df = melted_df.rename(columns={"Model": "model", "Year": "year"})

    # Sort by model, year, and day_of_year
    melted_df = melted_df.sort_values(by=["model", "year", "day_of_year"]).reset_index(drop=True)

    return melted_df

# A function that converts observations to pseudo-observations for a multi-model mean case
def modify_obs_model_mean(
    obs_df:pd.DataFrame,
    coeffs_mean_ds:xr.Dataset,
    glob_temp_mm_df:pd.DataFrame,
    target_year:int,
    doy_index:int|None=None)->pd.DataFrame:
    """
    Modify observed temperatures for all days of year OR the days corresponding to a given 'doy_index' using multi-model mean regression coefficients.
    
    Parameters:
        obs_df: pd.DataFrame
            Contains observed daily temperatures or their 2 to 31-day running means. Rows are years, columns are "day_of_year".
        coeffs_mean_ds: xr.Dataset
            Contains regression coefficients B and D for each day_of_year or for each 2 to 31-day running mean period in a year.
        glob_temp_mm_df: pd.DataFrame 
            Contains the observed (past) and multi-model mean simulated (future) 11-year running mean of the global mean temperature for each year.
        target_year: int 
            Year for which pseudo-observations are calculated.
        doy_index: int None | None
            day-of-year index. If given, pseudo-observations are only calculated for that observations corresponding to it. Otherwise they're calculated for all
            days/n-day periods 
    
    Returns:
        pseudo_obs: pd.DataFrame 
            Contains pseudo-observations for the n-day period that corresponds to the given doy_index (other days are NaN values)
            Contains pseudo-observations for all n-day periods of the year (if computed for all days of year)
    """

    # Smoothed global mean temperature (Relative to year 2000)
    g = glob_temp_mm_df.squeeze()
    gg = g.loc[target_year]

    # Initialize DataFrame to store pseudo-observations
    pseudo_obs = pd.DataFrame(index=obs_df.index,columns=obs_df.columns)

    # Compute pseudo-observations corresponding only to the given doy_index
    if doy_index:

        # Extract observations for a day or an n-day running mean to a Pd.Series
        obs_series = obs_df[doy_index]

        # Regression coefficients
        B = coeffs_mean_ds["B"].values[doy_index-1]
        D = coeffs_mean_ds["D"].values[doy_index-1]

        # Calculate intermediate values: z_obs(t0, t)
        z_obs = obs_series + (B * (g.loc[target_year] - g.loc[obs_series.index]))

        # Mean value, Z(t), for the reference period
        Z = z_obs.loc["1901":"2024"].mean()

        # sqrt(R(t0, t))
        ratio = (1. + gg * D) / (1. + g * D)
        ratio = np.maximum(ratio, 0)
        R = np.sqrt(ratio)

        # Compute final pseudo-observations
        y_obs = Z + (z_obs - Z) * R

        # Save the results to a DataFrame
        pseudo_obs[doy_index] = y_obs  
    
    else:
        # Iterate over all days
        for day in obs_df.columns:
            
            # Extract observations for a day or an n-day running mean to a Pd.Series
            obs_series = obs_df[day]

            # Regression coefficients
            B = coeffs_mean_ds["B"].values[day-1]
            D = coeffs_mean_ds["D"].values[day-1]

            # Calculate intermediate values: z_obs(t0, t)
            z_obs = obs_series + (B * (g.loc[target_year] - g.loc[obs_series.index]))

            # Mean value, Z(t), for the reference period
            Z = z_obs.loc["1901":"2024"].mean()

            # sqrt(R(t0, t))
            ratio = (1. + gg * D) / (1. + g * D)
            ratio = np.maximum(ratio, 0)
            R = np.sqrt(ratio)

            # Compute final pseudo-observations
            y_obs = Z + (z_obs - Z) * R

            # Save the results to a DataFrame
            pseudo_obs[day] = y_obs  
    
    return pseudo_obs

def get_pseudo_obs_and_qr_mm(
    attribution_cases:dict,
    config_vars:dict,
    input_data_dir:str,
    obs_df:pd.DataFrame,
    coeffs_mean_ds:xr.Dataset,
    glob_temp_mean_df:pd.DataFrame
    )->tuple[dict,dict]:

    """
    Computes model-mean pseudo-observations for all attribution cases and EITHER reads pre-computed values of quantiles OR applies QR to observations and model-specific mean-observations.

    Parameters:
        attribution_cases: dict 
            Contains information of all the cases to which attribution is applied (e.g. preind, target, future, obs)
        config_vars: dict
            Contains all the variables that are needed for computing pseudo-obsrevations and applying QR (e.g. station_id, clim_var, n_days, etc.)
        input_data_dir: str
            Path to base directory
        obs_df: pd.DataFrame
            A pivoted DataFrame (rows = year, columns = day_of_year) containing daily temperature observations or their 2 to 31-day running means.
        coeffs_mm_ds: xr.Dataset
            A dataset which contains the model mean regression coefficients
        glob_temp_mm_df: pd.DataFrame
            A DataFrame which contains the model mean simulated global 11-year running mean temperature 
        
    Returns:
        tuple: A tuple which contains the model-mean pseudo-observations and quantiles in separate dictionaries. 
    """

    # Get the needed variables from the config_vars dictionary
    obs_source = config_vars["obs_source"]
    station_id = config_vars["station_id"]
    clim_var = config_vars["clim_var"]
    ssp = config_vars["ssp"]
    n_days = config_vars["n_days"]
    doy_index = config_vars["doy_index"]
    base1_year = config_vars["base1_year"]
    base2_year = config_vars["base2_year"]
    quantiles = config_vars["quantiles"]

    # Dictionary of long names for climate variables
    clim_var_map={
        "tas": "daily mean temperature",
        "tasmax": "daily maximum temperature",
        "tasmin": "daily minimum temperature"
    }

    # Save the results to dictionaries
    quantiles_mm_dict = {}
    pseudo_obs_mm_dict = {}

    # Use the earliest year for observations if they start later than 'base1_year'
    base1_year = np.maximum(base1_year, obs_df.index.min())
    
    # Calculate pseudo-observations for all attribution cases and read already existing qr-files or apply qr is the files are not found
    for name, case in attribution_cases.items():
        
        # Observation case
        if case == "obs":
            # Read the QR-file
            quantiles_mm_dict[name] = read_qr_mm_obs(input_data_dir, obs_source, station_id, clim_var, ssp, scenario="obs", n_days=n_days)
        
            # Check if QR has been done previously
            qr_exists = quantiles_mm_dict[name] is not None
            
            # Apply QR to observations
            if not qr_exists:
                print(f"QR-file not found for the {n_days}-day moving average of {clim_var_map[clim_var]} observations.") 
                print("QR will be applied now...")
                
                melted_obs_df = melt_obs_df(obs_df.loc[base1_year:base2_year]) # Use observations only from the baseline period
                qr_obs_dict = apply_quantile_regression(melted_obs_df["day_of_year"], melted_obs_df["temperature"], quantiles=quantiles, n_harmonics=6, max_iter=10000, num_workers=4)
                qr_obs_df = transfer_qr_obs_mm(qr_obs_dict)
                quantiles_mm_dict[name] = qr_obs_df

                # Save the results of QR so that it doesn't need to be run again in the future
                save_qr_obs_and_mm(input_data_dir, obs_source, station_id, clim_var, ssp, n_days, case, base1_year, base2_year, qr_obs_df)
        
        # Pseudo-observation case
        else:
            # Read the QR-file
            quantiles_mm_dict[name] = read_qr_mm_obs(input_data_dir, obs_source, station_id, clim_var, ssp, case, n_days)
            
            # Check if QR has been done previously
            qr_exists = quantiles_mm_dict[name] is not None

            if qr_exists:
                # Calculate a time-series of pseudo-observations only for the selected time-period (e.g. 1 Jan - 5 Jan)
                pseudo_obs_mm_dict[name] = modify_obs_model_mean(obs_df, coeffs_mean_ds, glob_temp_mean_df, case, doy_index)

            else:
                # Calculate pseudo-observations for all n_day periods
                psudo_obs_mm_df = modify_obs_model_mean(obs_df, coeffs_mean_ds, glob_temp_mean_df, case)
                pseudo_obs_mm_dict[name] = psudo_obs_mm_df
                
                print(f"QR-file not found for the {n_days}-day moving average of {clim_var_map[clim_var]} model-mean pseudo-observations in the year {case}.")
                print("QR will be applied now...")
                
                melted_pseudo_obs_df = melt_obs_df(psudo_obs_mm_df.loc[base1_year:base2_year]) # Use pseudo-observations only from the baseline period
                qr_pseudo_obs_dict = apply_quantile_regression(melted_pseudo_obs_df["day_of_year"], melted_pseudo_obs_df["temperature"], quantiles=quantiles, n_harmonics=6, max_iter=10000, num_workers=4)
                qr_pseudo_obs_df = transfer_qr_obs_mm(qr_pseudo_obs_dict)
                quantiles_mm_dict[name] = qr_pseudo_obs_df

                # Save the results of QR so that it doesn't need to be run again in the future
                save_qr_obs_and_mm(input_data_dir, obs_source, station_id, clim_var, ssp, n_days, case, base1_year, base2_year, qr_pseudo_obs_df)
    
    return pseudo_obs_mm_dict, quantiles_mm_dict

def get_pseudo_obs_and_qr_sm(
    attribution_cases:dict,
    config_vars:dict,
    input_data_dir:str,
    obs_df:pd.DataFrame,
    coeffs_sm_ds:xr.Dataset,
    glob_temp_sm_df:pd.DataFrame,
    n_boots:int) -> tuple[dict,dict]:

    """
    Computes model-specific pseudo-observations for all attribution cases and EITHER reads pre-computed values of quantiles OR applies QR to model-specific pseudo-observations.

    Parameters:
        attribution_cases:dict 
            Contains information for all the cases to which attribution is applied (e.g. preind, target, future, obs)
        config_vars:dict
            Contains all the variables that are needed for computing pseudo-obsrevations and applying QR (e.g. station_id, clim_var, n_days, etc.)
        input_data_dir:str
            Path to base directory
        obs_df: pd.DataFrame
            A pivoted DataFrame (rows = year, columns = day_of_year) containing daily temperature observations or their 2 to 31-day running means.
        coeffs_sm_ds: xr.Dataset
            A dataset which contains the single model regression coefficients
        glob_temp_sm_df: pd.DataFrame
            A DataFrame which contains the single model simulated global 11-year running mean temperature 
        n_boots:int
            Number of bootstrap iterations
        
    Returns:
        tuple: A tuple which contains the model-specific pseudo-observations and quantiles in separate dictionaries. 
    """

    # Dictionary of long names for climate variables
    clim_var_map={
        "tas": "daily mean temperature",
        "tasmax": "daily maximum temperature",
        "tasmin": "daily minimum temperature"
    }

    # Get the needed variables from the config_vars dictionary
    obs_source = config_vars["obs_source"]
    station_id = config_vars["station_id"]
    clim_var = config_vars["clim_var"]
    ssp = config_vars["ssp"]
    n_days = config_vars["n_days"]
    doy_index = config_vars["doy_index"]
    base1_year = config_vars["base1_year"]
    base2_year = config_vars["base2_year"]
    quantiles = config_vars["quantiles"]

    # Number of models
    n_models = len(glob_temp_sm_df.columns)

    # Save the results to dictionaries
    quantiles_sm_dict = {}
    pseudo_obs_sm_dict = {}

    # Use the earliest year for observations if they start later than 'base1_year'
    base1_year = np.maximum(base1_year, obs_df.index.min())

    for name, case in attribution_cases.items():
        
        # Observation case
        if case == "obs":
            continue
        
        if n_boots > 0:
        
            # Read the QR-file
            quantiles_sm_dict[name] = read_qr_sm_obs(input_data_dir, obs_source, station_id, clim_var, ssp, case, doy_index, n_days)
            
            # Check if QR has been done previously
            qr_exists = quantiles_sm_dict[name] is not None

            if qr_exists:
                # Calculate a time-series of pseudo-observations only for the selected time-period (e.g. 1 Jan - 5 Jan)
                pseudo_obs_sm_dict[name] = modify_obs_single_models(obs_df, coeffs_sm_ds, glob_temp_sm_df, case, doy_index)
            else:
                print(f"QR-file not found for the single-model {n_days}-day moving average of {clim_var_map[clim_var]} pseudo-observations in the year {case}.")
                print(f"QR is applied to {n_models} model-specific pseudo-observations in the year {case}.")
                print("This will take approximately 20 to 45 minutes...")
            
                # Calculate pseudo-observations for all n_day periods
                pseudo_obs_sm = modify_obs_single_models(obs_df, coeffs_sm_ds, glob_temp_sm_df, case)
                pseudo_obs_sm[name] = pseudo_obs_sm

                # Apply QR to all pseudo-observations from the baseline period. The results go to a dictionary 
                qr_sm_dict = apply_qr2sm_pseudo_obs(pseudo_obs_sm.loc[pd.IndexSlice[:,base1_year:base2_year], :], quantiles, n_harmonics=6, max_iter=10000, num_workers=4)
                quantiles_sm_dict[name] = transfer_qr_obs_sm(qr_sm_dict)

                # Save the QR-results to a NetCDF-file so that QR does not have to be applied again in the future
                save_qr_sm(input_data_dir, obs_source, station_id, clim_var, ssp, n_days, case, base1_year, base2_year, qr_sm_dict)
                #input_data_dir:str, obs_source:str, station_id:str, clim_var:str, ssp:str, n_days:int, case:int, base1_year:int, base2_year:int, qr_sm_dict:dict

        else:
            # Don't apply QR to single models (or read the corresponding files) is n_boots = 0
            pseudo_obs_sm_dict[name] = modify_obs_single_models(obs_df, coeffs_sm_ds, glob_temp_sm_df, case, doy_index)

    return pseudo_obs_sm_dict, quantiles_sm_dict


#######################################################################################################################################################################################################################

############################################################ Functions for performing quantile regression using a Fourier series as a predictor #######################################################################

def fourier_basis(doy_array:np.ndarray, n_harmonics:int) -> np.ndarray:
    """
    A Fourier basis matrix for an array which contains days of year and a specified number of harmonics.

    Parameters:
        doy_array (np.ndarray): A numpy array of days of year (e.g. 1-365).
        n_harmonics (int): Number of harmonics in a Fourier Series.

    Returns:
        np.ndarray: A 2D array a row corresponds to a specific doy and a column represents a Fourier Series term (sin or cos).
                    The number of columns is 2 * n_harmonics.
    """
    
    fourier_components = []

    for i in range(1, n_harmonics + 1):
        
        # Compute the sine and cosine terms for the current harmonic
        fourier_components.append(np.sin(2 * np.pi * i * doy_array / 365))
        fourier_components.append(np.cos(2 * np.pi * i * doy_array / 365))
    
    return np.column_stack(fourier_components)

def apply_quantile_regression(doy_data:np.ndarray, temp_data:np.ndarray, quantiles:np.ndarray, n_harmonics:int, max_iter=10000, num_workers=4):
    """
    Perform quantile regression in parallel for each quantile.

    Parameters:
        doy_data: np.ndarray (day-of-year data).
        temp_data: np.ndarray (temperature data).
        quantiles: np.ndarray Array of quantiles (e.g. 0.01...0.99).
        n_harmonics: Number of harmonics for Fourier basis (default is 6).
        max_iter: Maximum number of iterations for each quantile regression.
        num_workers: Number of parallel workers.

    Returns:
        quantile_models: Dictionary of fitted models for each quantile.
    """
    
    # Fourier basis
    X_fourier = fourier_basis(doy_data, n_harmonics=n_harmonics)

    # Add a constant term for the intercept
    X = sm.add_constant(X_fourier)

    # Function to fit a single quantile regression model
    def fit_single_quantile(q):
        try:
            model = QuantReg(temp_data, X)
            result = model.fit(q=q, max_iter=max_iter)
            return round(q, 3), result
        except Exception as e:
            print(f"Error fitting quantile {q:.3f}: {e}")
            return round(q, 3), None

    # Parallel execution using ThreadPoolExecutor
    quantile_models = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fit_single_quantile, q): q for q in quantiles}

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),):
            
            q = futures[future]
            try:
                rounded_q, result = future.result()
                if result:
                    quantile_models[rounded_q] = result
            except Exception as e:
                print(f"Error processing quantile {q:.3f}: {e}")

    return quantile_models

def apply_qr2sm_pseudo_obs(pseudo_obs_sm_df:pd.DataFrame, quantiles:np.ndarray, n_harmonics=6, max_iter=10000, num_workers=4):
    """
    Apply quantile regression for model-specific pseudo-observations in the pseudo_obs_sm_df DataFrame.

    Parameters:
        pseudo_obs_sm_df: Input DataFrame with columns ['model', 'year', 'day_of_year', 'temperature'].
        quantiles: np.ndarray Array of quantiles (e.g. 0.01...0.99).
        n_harmonics: Number of harmonics for Fourier basis (default is 6).
        max_iter: Maximum number of iterations for each quantile regression.
        num_workers: Number of parallel workers.

    Returns:
        results_by_model: Dictionary where keys are model names, and values are QR results.
    """
    qr_sm_dict = {}

    # Melt the MultiIndex Dataframe so that QR can be applied
    melted_obs_df = melt_pseudo_obs_single_models(pseudo_obs_sm_df)

    # Group the DataFrame by 'model'
    grouped = melted_obs_df.groupby('model')

    # Loop over all models
    for model, group in grouped:
        print(f"Processing: {model}.")
        
        # Extract day_of_year and temperature data for this model
        doy_data = group['day_of_year'].values
        temp_data = group['temperature'].values
        
        # Apply quantile regression for this model's pseudo-observations
        quantile_models = apply_quantile_regression(
            doy_data,
            temp_data,
            quantiles=quantiles,
            n_harmonics=n_harmonics,
            max_iter=max_iter,
            num_workers=num_workers
        )

        qr_sm_dict[model] = quantile_models

    return qr_sm_dict

def transfer_qr_obs_mm(quantile_regressed_obs_dict:dict[float, sm.OLS]) -> pd.DataFrame:

    """
    Generates a DataFrame containing predicted values from quantile regressions 
    for each quantile, using a Fourier series basis.

    Parameters:
    observations_df : pd.DataFrame
        A DataFrame containing the original observation data. The columns 
        define the variables of interest.

    quantile_regressed_obs_dict : dict[float, sm.OLS]
        A dictionary where keys are quantiles (e.g., 0.01, 0.02, ..., 0.99) 
        and values are statsmodels OLS objects fitted for each quantile.

    Returns:
    pd.DataFrame
        A DataFrame indexed by quantiles (from 0.01 to 0.99), with columns 
        matching those in `observations_df`. Each entry contains the 
        predicted values for the corresponding quantile.
    """

    # List of quantiles
    quantiles = np.array(list(quantile_regressed_obs_dict.keys()))

    # Days of year
    doy= np.arange(1,366,1)

    # Initialize an empty DataFrame that has the same dimensions as the observations_df DataFrame
    quantile_regressed_obs_df = pd.DataFrame(index=quantiles, columns=doy)
    quantile_regressed_obs_df.index.name = "quantile"

    # Initialize a day of year array and fit a 6-component Fourier series to it
    doy_Fourier = fourier_basis(doy,n_harmonics=6)
    X_fit = sm.add_constant(doy_Fourier)  # Add constant to match the training data structure

    # Loop over each quantile and extract the regressed values to the dataframe
    for q in quantile_regressed_obs_dict:
        quantile_regressed_obs_df.loc[q] = quantile_regressed_obs_dict[q].predict(X_fit)

    return quantile_regressed_obs_df.sort_index().astype(np.float64)


def transfer_qr_obs_sm(qr_single_models_dict:dict[str, dict[float, sm.OLS]]) -> dict:

    """
    Generates a DataFrame containing predicted values from QR 
    for each quantile function from a Fourier series basis.

    Parameters:
    qr_single_models_dict : dict[str, dict[float,sm.OLS]]
        A dictionary where keys are quantile functions (e.g., 0.01, 0.02, ..., 0.99) 
        and values are statsmodels OLS objects fitted for each quantile.

    Returns:
    pd.DataFrame
        A DataFrame indexed by quantiles (from 0.01 to 0.99), with columns 
        matching those in `observations_df`. Each entry contains the 
        predicted values for the corresponding quantile.
    """
    
    # Initialize an empty dictionary for storing the results
    results_dict = {}

    day_of_year = np.arange(1,366,1)
    
    # Loop through the keys (models) and values (dictionaries)
    for model, dict in qr_single_models_dict.items():

        # List of quantiles in each model
        quantiles = np.array(list(dict.keys()))

        # Initialize an empty DataFrame that has the same dimensions as the observations_df DataFrame
        quantile_regressed_obs_df = pd.DataFrame(index=quantiles, columns=day_of_year)
        quantile_regressed_obs_df.index.name = "quantile"

        # Initialize a day of year array and fit a 6-component Fourier series to it
        day_of_year_Fourier = fourier_basis(day_of_year,n_harmonics=6)
        X_fit = sm.add_constant(day_of_year_Fourier)

        # Loop over each quantile and extract the regressed values to the dataframe
        for q in dict:
            quantile_regressed_obs_df.loc[q] = dict[q].predict(X_fit)

        # Save the results to a dictionary
        results_dict[model] = quantile_regressed_obs_df

    return results_dict
    

def save_qr_obs_and_mm(input_data_dir:str, obs_source:str, station_id:str, clim_var:str, ssp:str, n_days:int, case:str, base1_year:int, base2_year:int, quantiles_df:pd.DataFrame):
    """
    Saves the given DataFrame containing quantile-functions obtained from applying QR to observations or model-mean pseudo-observations to a NetCDF file.
    
    Parameters:
    ----------
    input_data_dir : str
        Base data directory path, which defines the path to directory where the NetCDF-file is saved.
    obs_source: str
        Source of the observation (FMI, SMHI or FROST)
    station_id: str
        Weather-station ID. (e.g. 101932 for Sodankylä)
    clim_var : str
        Climate variable name (e.g. tasmax, tas, tasmin).
    ssp: str
        Emission scenario (ssp119, ssp126, ssp245, ssp370, ssp585)
    case : str
        Attribution case (e.g. obs, 1900, 2025, 2050)
    base1_year: int
        Start year of the baseline period (e.g. 1901)
    base2_year: int
        End year of the baseline period (e.g. 2024)
    qr_sm_dict : dict
        Nested dict: {model: DataFrame with quantile as index, day_of_year as columns}.
    n_days : int
        Number of days for running mean window.
    ----------

    Returns: None
    """

    if case == "obs":

        # Path to qr-files of observations
        path2dir = Path(input_data_dir) / "qr_files" / "observations"
    else:
         # Path to qr-files for model-mean pseudo-observations
        path2dir = Path(input_data_dir) / "qr_files" / "model_mean" / ssp

    os.makedirs(path2dir, exist_ok=True)

    # Extract the quantiles and the number of days
    quantiles = quantiles_df.index.to_numpy()
    days = quantiles_df.columns.astype(int)

    # Save the results from the DataFrame to a NetCDF file
    ds = xr.Dataset(
            {"temperature": (("quantile","day_of_year"), quantiles_df.values)},
            coords={
                "quantile": quantiles,
                "day_of_year": days},
            attrs={
                "description": "Quantile-functions of model-mean pseudo-observations",
                "obs_source": obs_source,
                "station_id": str(station_id),
                "variable": clim_var,
                "case": case,
                "baseline_period": f"{str(base1_year)}-{str(base2_year)}",
                "window": f"{n_days}-day running mean window",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
    )
    
     # Add n_days as a dimension
    ds = ds.expand_dims(dim={"n_days": [n_days]})
    ds["n_days"] = ("n_days", [n_days])

    if case != "obs":
        ds.attrs["ssp"] = ssp

    # Define the output file path
    if case == "obs":
        path2output_file = os.path.join(path2dir,f"{obs_source}_{station_id}_{clim_var}_qr_{case}_{n_days}days.nc")
    else:
        path2output_file = os.path.join(path2dir,f"{obs_source}_{station_id}_{clim_var}_qr_mm_{ssp}_{case}_{n_days}days.nc")

    # Save to NetCDF
    ds.to_netcdf(path2output_file)

def save_qr_sm(input_data_dir:str, obs_source:str, station_id:str, clim_var:str, ssp:str, n_days:int, case:int, base1_year:int, base2_year:int, qr_sm_dict:dict):
    
    """
    Saves model-specific quantile-functions, obtained from applying QR to model-specific pseudo-observations, to a NetCDF-file.

    Parameters:
    ----------
    input_data_dir : str
        Base data directory path, which defines the path to directory where the NetCDF-file is saved.
    obs_source: str
        Source of the observation (FMI, SMHI or FROST)
    station_id: str
        Weather-station ID. (e.g. 101932 for Sodankylä)
    clim_var : str
        Climate variable name (e.g. tasmax, tas, tasmin).
    ssp: str
        Emission scenario (ssp119, ssp126, ssp245, ssp370, ssp585)
    case : int
        Attribution case (e.g. 1900, 2025, 2050)
    base1_year: int
        Start year of the baseline period (e.g. 1901)
    base2_year: int
        End year of the baseline period (e.g. 2024)
    qr_sm_dict : dict
        Nested dict: {model: DataFrame with quantile as index, day_of_year as columns}.
    n_days : int
        Number of days for running mean window.
    ----------

    Returns: None
    """

    # Path to model-mean quantile regression files
    path2dir = Path(input_data_dir) / "qr_files" / "single_models" / ssp

    os.makedirs(path2dir, exist_ok=True)

    # Prepare flattened data
    qr_sm_list = []
    for model, quantiles in qr_sm_dict.items():
        for quantile, day_of_year_data in quantiles.iterrows():
            for day_of_year, temperature in day_of_year_data.items():
                qr_sm_list.append({
                    "model": model,
                    "quantile": quantile,
                    "day_of_year": int(day_of_year),
                    "temperature": temperature
                })

    qr_sm_list_df = pd.DataFrame(qr_sm_list)

    # Prepare coordinates
    models = sorted(qr_sm_list_df["model"].unique())
    quantiles = np.sort(qr_sm_list_df["quantile"].unique())
    day_of_year = np.sort(qr_sm_list_df["day_of_year"].unique())

    # Pivot into 3D structure: [model, quantile, day_of_year]
    temp_values = (
        qr_sm_list_df
        .pivot_table(
            index=["model", "quantile"],
            columns="day_of_year",
            values="temperature"
        )
        .reindex(index=pd.MultiIndex.from_product([models, quantiles], names=["model", "quantile"]))
        .to_numpy()
        .reshape(len(models), len(quantiles), len(day_of_year))
    )

    # Create Dataset
    ds = xr.Dataset(
        {
            "temperature": (["model", "quantile", "day_of_year", "n_days"], temp_values[:,:,:,np.newaxis])
        },
        coords={
            "model": ("model", models),
            "quantile": ("quantile", quantiles),
            "day_of_year": ("day_of_year", day_of_year),
            "n_days": ("n_days", [n_days])
        },
        attrs={
            "description": "Quantile-functions of model-specific pseudo-observations",
            "obs_source": obs_source,
            "station_id": str(station_id),
            "variable": clim_var,
            "ssp": ssp,
            "year": str(case),
            "baseline_period": f"{str(base1_year)}-{str(base2_year)}",
            "window": f"{n_days}-day running mean",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    # Save to NetCDF
    nc_path = os.path.join(path2dir,f"{obs_source}_{station_id}_{clim_var}_qr_pseudo_obs_sm_{ssp}_{case}_{n_days}days.nc")
    ds.to_netcdf(nc_path)

    print(f"Saved NetCDF: {nc_path}")


def newest_qr_file(all_days: Path, nday: Path) -> Path:
    
    """
    Return the newest existing QR file for a case.
    Raises FileNotFoundError if neither file exists.
    """

    paths = [p for p in (all_days, nday) if p.exists()]
    if not paths:
        raise FileNotFoundError
    return max(paths, key=lambda p: p.stat().st_mtime)

def read_qr_mm_obs(input_dir:str, obs_source:str, station_id:int, clim_var:str, ssp:str, scenario:str, n_days:int)->pd.DataFrame:

    """
    Load multi-model-mean quantile-regressed temperature data for a given station
    and return it as a pivoted DataFrame (quantile x day_of_year).

    Parameters
    ----------
    input_dir : str
        Base directory containing `qr_obs/model_mean`.
    obs_source : str
        Observation source identifier (e.g., "FMI").
    station_id : int
        Weather-station ID. (e.g. 101932 for Sodankylä)
    clim_var : str
        Climate variable (e.g., "tas", "tasmax", "tasmin").
    ssp: str
        Emission scenerio (e.g. ssp119, ssp126, ssp245, ssp370, ssp585)
    scenario : str
        "obs" or pseudo-observation year (e.g., "2025").
    n_days : int
        Selected `n_days` value in the dataset.

    Returns
    -------
    pandas.DataFrame or None
        Pivoted DataFrame with quantiles as rows and day_of_year as columns,
        or None if the file is missing or an error occurs.
    """

    try:
        # Path to input file
        if scenario == "obs":
            path2dir = Path(input_dir) / "qr_files" / "observations" 
            all_days_file = path2dir / f"{obs_source}_{station_id}_{clim_var}_qr_{scenario}.nc"
            nday_file = path2dir / f"{obs_source}_{station_id}_{clim_var}_qr_{scenario}_{n_days}days.nc"
        else:
            path2dir = Path(input_dir) / "qr_files" / "model_mean" / ssp
            all_days_file = path2dir / f"{obs_source}_{station_id}_{clim_var}_qr_mm_{ssp}_{scenario}.nc"
            nday_file = path2dir / f"{obs_source}_{station_id}_{clim_var}_qr_mm_{ssp}_{scenario}_{n_days}days.nc"
        
        #print("all_days:", all_days_file, all_days_file.exists())
        #print("nday    :", nday_file.exists())
        
        # Use the newest file
        path2file = newest_qr_file(all_days_file, nday_file)
        
        # Open data NetCDF file and read quantiles for all the days of year for a given "n_days" number of days
        qr_ds = xr.open_dataset(path2file)

        # Check for errors
        if "n_days" in qr_ds.dims:
            qr_ds = qr_ds.sel(n_days=n_days)

        if "quantile" not in qr_ds.dims or qr_ds.dims["quantile"] == 0:
            raise ValueError(
                f"QR file exists but contains no quantiles: {path2file}"
            )

        qr_df = qr_ds["temperature"].to_dataframe().reset_index()
        qr_pivot_df = qr_df.pivot(index="quantile", columns="day_of_year", values="temperature")
        
        return qr_pivot_df
    
    # Return None if the file is not found
    except FileNotFoundError as e:
        print(e)
        return None 
    
    # Return None with all other possible errors
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return None 

def read_qr_sm_obs(input_dir:str, obs_source:str, station_id:int, clim_var:str, ssp:str, case:int, doy_index:int, n_days:int)->pd.DataFrame:

    """
    Read model-specific quantile functions computed from pseudo-observation temperature data
    for a specific station, climate variable, attribution case, n_days value, and
    day-of-year, and return it as a pivoted DataFrame.

    Parameters
    ----------
    input_dir : str
        Base directory containing `qr_files/single_models`.
    obs_source : str
        Observation source identifier (e.g., "FMI").
    station_id : int
        Weather-station ID.
    clim_var : str
        Climate variable (e.g., "tas", "tasmax", "tasmin").
    scenario : str
        Pseudo-observation scenario year (e.g., "2025").
    doy_index : int
        Day-of-year to extract (1-365).
    n_days : int
        Number of days.
    
    Returns
    -------
    pandas.DataFrame or None
        DataFrame indexed by quantile with model names as columns and
        temperature values as entries, or None if the file is missing or
        an error occurs.
    """
    try:
        # Path to single-model QR directory
        path2dir = Path(input_dir) / "qr_files" / "single_models" / ssp

        # Filenames
        all_days_file = path2dir / f"{obs_source}_{station_id}_{clim_var}_qr_pseudo_obs_sm_{ssp}_{case}.nc"
        nday_file = path2dir / f"{obs_source}_{station_id}_{clim_var}_qr_pseudo_obs_sm_{ssp}_{case}_{n_days}days.nc"

        # Pick the newest existing file
        path2file = newest_qr_file(all_days_file, nday_file)
        
        # Open data NetCDF file and read quantiles for all the days of year for a given "n_days" number of days
        qr_ds = xr.open_dataset(path2file).sel(n_days=n_days,day_of_year=doy_index)
        qr_df = qr_ds["temperature"].to_dataframe().reset_index()
        qr_pivot_df = qr_df.pivot(index="quantile",columns="model",values="temperature")
        
        return qr_pivot_df

    # Return None if the file is not found
    except FileNotFoundError as e:
        print(e)
        return None 
    
    # Return None with all other possible errors
    except Exception as e:
        print(f"An unexpected error occured: {e}")
        return None

############################################################################################################################################################################################################################

####################################################################################### Functions for constructing the CDFs and PDFs for daily pseudo-observations #########################################################

def frsgs(y, valmax, valmin, nbins):
    
    # This function converts a sample of (original or modified) observations (y)
    # to a continuous SGS probability distribution (f). The corresponding
    # cumulative distribution (cub_prob) is also calculated.
    
        
    # Calculation of mean, standard deviation, skewness and excess kurtosis
    # (using wikipedia formulas; estimate for skewness is only unbiased
    # for symmetric distributions)  
    
    EPS=1e-3
    resol=(valmax-valmin)/(nbins-1)

        
    n = len(y)
    f = np.zeros((nbins))
    cum_prob = np.zeros((nbins))
    
    m1=0
    m2=0
    m3=0
    m4=0
    ndata=0
    for i in np.arange(0, n):
        if np.isfinite(y[i]):
           m1=m1+y[i]
           ndata=ndata+1
        
     
    m1=m1/ndata        
    for i in np.arange(0,n):
        if np.isfinite(y[i]):    
           m2=m2+((y[i]-m1)**2.)/ndata
           m3=m3+((y[i]-m1)**3.)/ndata
           m4=m4+((y[i]-m1)**4.)/ndata

    std=np.sqrt((ndata-0.)/(ndata-1.)*m2)
    skew=m3/(std**3.)
    variance = std**2
    kurt=(ndata+1.)*ndata/((ndata-1.)*(ndata-2.)*(ndata-3.))*ndata*m4/(std**4.) -3*(ndata-1.)*(ndata-1.)/((ndata-2.)*(ndata-3.))
    
    if kurt < 3./2.*(skew**2.):
          kurt=3./2.*(skew**2.)+EPS
  
    
    #  SGS parameters using Eqs. 8a-8c in Sardesmukh et al. 2015
    # (J. Climate, 28, 9166-9187)
    
    # e2=np.maximum(2./3.*(kurt-3./2.*(skew**2.)/(kurt+2-(skew**2.))),
    #               1.-1./np.sqrt(1+(skew**2.)/4.)+EPS) 

    e2=np.maximum(np.maximum(2./3.*(kurt-3./2.*(skew**2.)/(kurt+2-(skew**2.))),\
           1.-1./np.sqrt(1+(skew**2.)/4.)+EPS),EPS)

    if (e2 > 2./3.):
          e2=2./3.*(1.-EPS)

    
    g=skew*std*(1.-e2)/(2*np.sqrt(e2))
    b2=2*(std**2.)*(1-e2/2.-((1-e2)**2.)/(8.*e2)*(skew**2.))
    
    if b2 < 0:
        f[:]=np.nan
        cum_prob[:]=np.nan    
        
        return f, cum_prob, (m1, variance, skew, kurt)
    
    # Calculation of the probability density function, first unnormalized.
    # Note that it is assumed that there is no probability mass beyond the range 
    # fmin...fmax -> these need to be put far enough in the tails.    
   
    
    for ind in np.arange(0, nbins):
        x=valmin+(ind-1.)/(nbins-1.)*(valmax-valmin)-m1
        f[ind]=np.log((np.sqrt(e2)*x+g)**2.+b2)*(-1.-1./e2) +(2*g/(e2*np.sqrt(b2))*np.arctan((np.sqrt(e2)*x+g)/np.sqrt(b2)))
    
    fmax=f[0]
    for ind in np.arange(1,nbins):
        if f[ind] > fmax:
            fmax=f[ind]
 
    
    sumf=0.
    for ind in np.arange(0,nbins):  
        f[ind]=np.exp(f[ind]-fmax)
  
    for ind in np.arange(0,nbins):
        sumf=sumf+resol*f[ind]
  
    for ind in np.arange(1,nbins):
        f[ind]=f[ind]/sumf

    
    cum_prob[0] = 0. 
    for ind in np.arange(1,nbins):
        cum_prob[ind]=cum_prob[ind-1]+resol*(f[ind]+f[ind-1])/2.
  
    for ind in np.arange(1,nbins):
        cum_prob[ind]=cum_prob[ind]/cum_prob[nbins-1] 
  
    
    return f, cum_prob, (m1, variance, skew, kurt)

def calculate_sgs(obs_df, valmax, valmin, nbins):

    
    obs_df = pd.DataFrame(obs_df)
    
    n_mod = obs_df.shape[1]
    
    resol=(valmax-valmin)/(nbins-1)
    index = np.arange(valmin, valmax+resol, resol).round(3)

    
    f_arr = np.zeros((len(index), n_mod))
    cp_arr = np.zeros((len(index), n_mod))
    

        
    # loop over all models (if there are many models)
    for m in np.arange(0,n_mod):
        
        # if there is only one realization
        if n_mod>1:
            temp = list(obs_df[m+1].values.squeeze())
        else:
            temp = list(obs_df.values.squeeze())
        
        
        f_arr[:,m], cp_arr[:,m],moments = frsgs(temp, valmax, valmin, nbins)
    
    return np.squeeze(f_arr), np.squeeze(cp_arr), moments

# Find the index of an element in a given array
def get_element_index(array:np.ndarray, element:float) -> int:
    """
    Returns the index of the array element closest to the given value.
    """
    return np.nanargmin(np.abs(array - element))

def log5p(T:np.ndarray, A:float, D:float, B:float, C:float, S:float)-> np.ndarray:

    """
    5-parameter logistic function
    T  : Temperature data
    A  : Lower asymptote as x -> -inf
    D  : Upper asymptote as x -> inf
    B  : Growth rate parameter
    C  : Location parameter
    S  : Asymmetry parameter
    More detailed description: https://www.r-bloggers.com/2019/11/five-parameters-logistic-regression/
    """

    denominator = 1 + np.exp(B*(C-T))

    return A + (D - A) / denominator**S

def d_log5p(T:np.ndarray, A:float, D:float, B:float, C:float, S:float)-> np.ndarray:

    """
    Derivative of the 5-parameter logistic function.
    """

    exp_term = np.exp(B*(C-T))
    denom = 1 + exp_term
    return (D - A) * S * B * exp_term / (denom**(S+1))


def initial_guess_log5pl(quantiles:np.ndarray, T_quantiles:np.ndarray)-> np.ndarray:
    
    """Estimate initial guess parameters for the 5-parameter logistic fitting.
    
    Parameters:
        quantiles: np.ndarray
            Contains quantiles (usually 0.01...0.99)
        T_quantiles: np.ndarray
            Contains temperatures that correspond to quantiles (obtained from quantile regression) in the range 0.01...0.99
    Returns:
        np.ndarray: Initial guess of the 5-parameter logistc function
    
    """

    # Remove any possible NaNs
    mask = ~np.isnan(T_quantiles) & ~np.isnan(quantiles)
    T_quantiles = T_quantiles[mask]
    quantiles = quantiles[mask]

    # The lower and upper asymptotes
    A0 = np.min(quantiles)
    D0 = np.max(quantiles)

   # Midpoint
    mid_quantile = (A0 + D0) / 2.0
    index_mid = get_element_index(quantiles, mid_quantile)
    C0 = T_quantiles[index_mid]

    # Set the slope as None at first
    slope = None

    # Estimate the slope around midpoint (the width, over which slope is estimated is increased 
    # if the difference between successive Temperature quantiles is zero)
    for i in [1, 5, 10]:
        i1 = index_mid - i
        i2 = index_mid + i
        
        # Indices must be within the range 0...99 as there are 99 quantiles
        if i1 < 0 or i2 >= len(T_quantiles):
            continue
        
        # Slope denominator
        denominator = T_quantiles[i2] - T_quantiles[i1]

        # Denominator must not be zero
        if not np.isclose(denominator, 0.0):
            slope = (quantiles[i2] - quantiles[i1]) / denominator
            break

    # Estimate slope using endpoints if the above fails
    if slope is None:
        slope = (D0 - A0) / (T_quantiles.max() - T_quantiles.min())

    # Estimate the growth rate parameter
    B0 = slope / (D0 - A0 + 1e-9)
    
    # Symmetric distribution as an initial guess
    S0 = 1.0

    return np.array([A0, D0, B0, C0, S0])

def fit_log5pl(quantiles:np.ndarray, T_quantiles:np.ndarray, T_range:np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    
    """Fit 5-parameter logistic function to a set of temperature quantiles.
    
    Parameters:
        quantiles: np.ndarray
            Contains quantiles (usually 0.01...0.99)
        T_quantiles: np.ndarray
            Contains temperatures that correspond to the quantiles in the quantiles array
        T_range: np.ndarray
            The range of temperatures to which probability distribution is fitted.
        
    Returns:
        tuple[np.ndarray, np.ndarray]
            The logistic function CDF and PDF in the range of quantiles 0.01...0.99
    """

    # Sort the temperature quantiles
    sorted_indices = np.argsort(T_quantiles)
    T_quantiles = T_quantiles[sorted_indices]
    
    # Initial guss of log5p parameters
    initial_guess = initial_guess_log5pl(quantiles, T_quantiles)
    
    # Parameter bounds (if needed)
    bounds=([-0.1, 0.8, 0, T_quantiles.min(), 0.001],
            [0.05, 1.0, 5, T_quantiles.max(), 10])
    
    # Apply curve_fit function to estimate the logistic function parameters
    popt, _ = curve_fit(log5p, T_quantiles, quantiles, p0=initial_guess, method='trf', bounds=bounds, maxfev=int(1e5))
    
    # Get the indices corresponding to minimum and maximum temperatures
    Tmin_idx = get_element_index(T_range, T_quantiles[0])
    Tmax_idx = get_element_index(T_range, T_quantiles[-1])

    # Initialize full-length arrays with NaNs
    CDF_log5p_full = np.full_like(T_range, np.nan)
    PDF_log5p_full = np.full_like(T_range, np.nan)

    # Temperature range of the logistic function
    T_range_log5p = T_range[Tmin_idx:Tmax_idx]

    # Construct CDF and PDF from the fitted parameters
    CDF_log5p_in_range = log5p(T_range_log5p, *popt)
    PDF_log5p_in_range = d_log5p(T_range_log5p, *popt)
    CDF_log5p_in_range = np.clip(CDF_log5p_in_range, 1e-12, 1 - 1e-12)

    # Replace NaNs with numbers in the correct range
    CDF_log5p_full[Tmin_idx:Tmax_idx] = CDF_log5p_in_range
    PDF_log5p_full[Tmin_idx:Tmax_idx] = PDF_log5p_in_range
    
    return CDF_log5p_full, PDF_log5p_full

# Extract constants for the gumbel functions
def get_gumbel_parameters(
    CDF_logistic: np.ndarray,
    PDF_logistic: np.ndarray,
    T_range: np.ndarray,
    merging_quantiles:tuple) -> np.ndarray:

    """Estimates the parameters of the Gumbel-distribution fitted to the tails by requiring that the Gumbel distribution CDF and PDF
    must be equal to the ones of the logistic function at some merging quantiles (e.g. 0.05 and 0.95)
    
    Parameters:
        CDF_logistic: np.ndarray
            Logistic function CDF
        PDF_logistic: np.ndarray
            Logistic function PDF
        T_range: np.ndarray
            The range of temperatures to which the probability distribution is fitted.
        merging_quantiles: np.ndarray
            Boundary quantiles where CDF and PDF of the logistic function and Gumbel distribution must be equal.
            
    Returns:
        np.ndarray:
            - mu_left: Location parameter for the left tail.
            - beta_left: Scale parameter for the left tail.
            - mu_right: Location parameter for the right tail.
            - beta_right: Scale parameter for the right tail.
    """

    # Find the temperatures corresponding to the merging quantiles
    T_left = get_T_from_CDF_level(CDF_logistic, T_range, merging_quantiles[0])
    T_right = get_T_from_CDF_level(CDF_logistic, T_range, merging_quantiles[1])

    # Indices of the merging quantile temperatures in the T_range array (0.05 and 0.95 quantile temperatures)
    T_left_idx = get_element_index(T_range, T_left)
    T_right_idx = get_element_index(T_range, T_right)

    # Corresponding values of CDF and PDF
    CDF_left, PDF_left = CDF_logistic[T_left_idx], PDF_logistic[T_left_idx]
    CDF_right, PDF_right = CDF_logistic[T_right_idx], PDF_logistic[T_right_idx]
    
    # Check that the CDF and PDF values correponding to the 0.05th and 0.95th quantiles are finite
    epsilon = 1e-12
    if (
        not np.isfinite(CDF_left) or not np.isfinite(PDF_left) or
        not np.isfinite(CDF_right) or not np.isfinite(PDF_right) or
        CDF_left <= epsilon or CDF_left >= 1 - epsilon or
        CDF_right <= epsilon or CDF_right >= 1 - epsilon or
        PDF_left <= epsilon or PDF_right <= epsilon
    ):
        return np.full(4, np.nan)

    # Solve left tail parameters (Gumbel min)
    z_left = -np.log(1 - CDF_left)
    beta_left = (1 - CDF_left) * z_left / PDF_left
    mu_left = T_left - beta_left * np.log(z_left)

    # Solve right tail parameters (Gumbel max)
    z_right = -np.log(CDF_right)
    beta_right = CDF_right * z_right / PDF_right
    mu_right = T_right + beta_right * np.log(z_right)

    return np.array([mu_left, beta_left, mu_right, beta_right])

def get_T_from_CDF_level(CDF_logistic:np.ndarray, T_range:np.ndarray, level:float) -> float:
    """
    Find T such that CDF(T) = level, ignoring NaNs.
    """
    mask = np.isfinite(CDF_logistic)
    if mask.sum() < 2:
        raise ValueError("Not enough valid CDF points to invert.")

    return np.interp(level, CDF_logistic[mask], T_range[mask])


# The CDF of the Gumbel distribution corresponding to max values (right tail)
def gumbel_max_cdf(T:np.ndarray, mu:float, beta:float) -> np.ndarray:
    """Cumulative Gumbel Distribution Function for max values.
    
    Parameters:
    T    : Temperature array
    mu   : Location parameter
    beta : Scale parameter
    
    """
    return np.exp(-np.exp(-(T - mu) / beta))

# The CDF of the Gumbel distribution corresponding to min values (left tail)
def gumbel_min_cdf(T:np.ndarray, mu:float, beta:float) -> np.ndarray:
    """Cumulative Gumbel Distribution Function for min values.
    
    Parameters:
    T    : Temperature array
    mu   : Location parameter
    beta : Scale parameter
    
    """
    return 1 - np.exp(-np.exp((T - mu) / beta))

# The PDF of the Gumbel distribution corresponding to max values (right tail)
def gumbel_max_pdf(T:np.ndarray, mu:float, beta:float):
    """
    Probability Density Distribution Functions of the Gumbel Distribution for max values.
    
    Parameters:
    T    : Temperature array
    mu   : Location parameter
    beta : Scale parameter
    """
    z = -(T - mu) / beta
    return (1 / beta) * np.exp(z - np.exp(z))

# The PDF of the Gumbel distribution corresponding to max values (right tail)
def gumbel_min_pdf(T:np.ndarray, mu:float, beta:float):
    """
    Probability Density Distribution Functions of the Gumbel Distribution for min values.
    
    Parameters:
    T    : Temperature array
    mu   : Location parameter
    beta : Scale parameter
    """
    z = (T - mu) / beta
    return (1 / beta) * np.exp(z - np.exp(z))

# Fit Gumbel functions for the tails (usually 0.01...0.10 and 0.90...0.99 quantiles) 
def fit_gumbel_functions(gumbel_params:np.ndarray, quantiles:np.ndarray, T_quantiles:np.ndarray, T_range:np.ndarray, transition_quantiles:tuple) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    """
    Fit Gumbel distribution functions for the left and right tails.

    Parameters:
        gumbel_params: Numpy array of Gumbel parameters:
            - mu_left: Location parameter for the left tail.
            - beta_left: Scale parameter for the left tail.
            - mu_right: Location parameter for the right tail.
            - beta_right: Scale parameter for the right tail.
        quantiles: np.ndarray 
            quantiles (usually 0.01...0.99)
        T_quantiles: 
            Temperature quantiles
        T_range np.ndarray:
            Array of temperatures to which the distributions are fitted.
        transition_quantiles: tuple
            quantiles at which the Gumbel-distribution end completely (usually 0.10 and 0.90)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            CDF_gumbel_left, CDF_gumbel_right, PDF_gumbel_left, PDF_gumbel_right
    """

    # Sort the temperature quantiles
    sorted_indices = np.argsort(T_quantiles)
    T_quantiles_sorted = T_quantiles[sorted_indices]

    # Get the indices of transition quantiles (9 and 89 in the standard case)
    index_left = get_element_index(quantiles, transition_quantiles[0])
    index_right = get_element_index(quantiles, transition_quantiles[1])

    # Get the corresponding quantile temperatures (10th and 90th quantiles usually)
    T_transition_left = T_quantiles_sorted[index_left]
    T_transition_right = T_quantiles_sorted[index_right]
    
    # Get the corresponding indices in the T_range array
    T_transition_left_idx = get_element_index(T_range, T_transition_left)
    T_transition_right_idx = get_element_index(T_range, T_transition_right)

    # Initialize full-length arrays with NaN values
    CDF_gumbel_left_full = np.full_like(T_range, np.nan)
    CDF_gumbel_right_full = np.full_like(T_range, np.nan)
    PDF_gumbel_left_full = np.full_like(T_range, np.nan)
    PDF_gumbel_right_full = np.full_like(T_range, np.nan)

    # Temperature range of the logistic function
    T_range_left = T_range[:T_transition_left_idx]
    T_range_right = T_range[T_transition_right_idx:]

    # Initial guesses for the left and right tail functions
    gumbel_left_guess = [gumbel_params[0], gumbel_params[1]]
    gumbel_right_guess = [gumbel_params[2], gumbel_params[3]]
    
    # Limits
    gumbel_bounds = ([-1e10, -1e10], [1e10, 1e10])

    # Get optimized parameters for the Gumbel functions
    popt_gumbel_left, _ = curve_fit(gumbel_min_cdf, T_quantiles_sorted[:index_left], quantiles[:index_left], p0=gumbel_left_guess, maxfev=5e5, method="trf", bounds=gumbel_bounds)
    popt_gumbel_right, _ = curve_fit(gumbel_max_cdf, T_quantiles_sorted[index_right+1:], quantiles[index_right+1:], p0=gumbel_right_guess, maxfev=5e5, method="trf", bounds=gumbel_bounds)

    # Compute CDFs for the left and right tails using the Gumbel function
    CDF_gumbel_left_full[:T_transition_left_idx] = gumbel_min_cdf(T_range_left, *popt_gumbel_left)
    CDF_gumbel_right_full[T_transition_right_idx:] = gumbel_max_cdf(T_range_right, *popt_gumbel_right)

    # Compute PDFs for the left and right tails using the derivative of the Gumbel function
    PDF_gumbel_left_full[:T_transition_left_idx] = gumbel_min_pdf(T_range_left, *popt_gumbel_left)
    PDF_gumbel_right_full[T_transition_right_idx:] = gumbel_max_pdf(T_range_right, *popt_gumbel_right)

    return CDF_gumbel_left_full, CDF_gumbel_right_full, PDF_gumbel_left_full, PDF_gumbel_right_full

# Calculate transition functions
def fit_transition_functions(logistic_function:np.ndarray,
    gumbel_left:np.ndarray,
    gumbel_right:np.ndarray,
    quantiles:np.ndarray,
    T_quantiles:np.ndarray,
    T_range:np.ndarray,
    target_quantiles:tuple) -> tuple[np.ndarray,np.ndarray]:

    """
    Fit smooth transition functions between the logistic function and Gumbel distributions.

    Parameters:
        logistic_function: np.ndarray
            The values of the logistic function across the temperature range.
        gumbel_left: np.ndarray
            The values of the left-tail Gumbel distribution across the temperature range.
        gumbel_right: np.ndarray
            The values of the right-tail Gumbel distribution across the temperature range.
        quantiles: np.ndarray
            Array of quantiles (usually 0.01...0.99) in ascending order.
        T_quantiles: np.ndarray
            Array of temperature quantiles, typically representing percentiles from 1st to 99th.
        T_range: np.ndarray
            The range of temperatures for which the transition functions are computed.
        target_quantiles: tuple
            The quantiles used to define transition regions. Defaults to (0.1, 0.9).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - `transition_left`: Transition function values for the left-tail (low temperatures).
            - `transition_right`: Transition function values for the right-tail (high temperatures).
    """

    # Sort the temperature quantiles
    sorted_indices = np.argsort(T_quantiles)
    sorted_T_quantiles = T_quantiles[sorted_indices]

    # Find the indices of the target quantiles
    index_left = get_element_index(quantiles, target_quantiles[0])
    index_right = get_element_index(quantiles, target_quantiles[1])
    
    # Get transition temperatures
    T1 = sorted_T_quantiles[0]
    T_transition_left = sorted_T_quantiles[index_left]
    T_transition_right = sorted_T_quantiles[index_right]
    T99 = sorted_T_quantiles[-1]

    # Find the indices corresponding to the transition temperatures in the T_range array          
    T1_idx = get_element_index(T_range, T1)
    T_transition_left_idx = get_element_index(T_range, T_transition_left)
    T_transition_right_idx = get_element_index(T_range, T_transition_right)
    T99_idx = get_element_index(T_range, T99)

    # Define the temperature ranges for the lower and upper transition regions
    T_lower = T_range[T1_idx:T_transition_left_idx]
    T_upper = T_range[T_transition_right_idx:T99_idx]
    
    # Normalize temperature ranges to values between 0 and 1 for the smoothstep function
    x_lower = (T_lower-T1) / (T_transition_left-T1)
    x_upper = (T_upper-T_transition_right) / (T99-T_transition_right)

    # Smooth step transition functions for lower and upper regions
    smooth_lower = smoothstep(x_lower)
    smooth_upper = smoothstep(x_upper)

    # Compute transition functions by blending logistic and Gumbel values for the left and right tails
    transition_left = transition_smooth(smooth_lower, logistic_function[T1_idx:T_transition_left_idx], gumbel_left[T1_idx:T_transition_left_idx], tail="left")
    transition_right = transition_smooth(smooth_upper, logistic_function[T_transition_right_idx:T99_idx], gumbel_right[T_transition_right_idx:T99_idx], tail="right")
    
    return transition_left, transition_right

# Smoothstep function
def smoothstep(x:np.ndarray) -> np.ndarray:
    """Smoothstep function"""
    return -2 * x**3 + 3 * x**2

# A function for blending the logistic function with the left or right-tailed Gumbel function.
def transition_smooth(transition_array: np.ndarray, logistic_function: np.ndarray, gumbel_function: np.ndarray, tail: str = "right") -> np.ndarray:
    
    """
    Compute a smooth transition between the logistic function and the Gumbel function 
    for either the left or the right tail.

    Parameters:
        transition_array: np.ndarray 
            A smooth weigth ranging from 0 to 1, with which the transition from one distribution to another is smoothed
        logistic_function: np.ndarray 
            Values of the logistic function over the temperature range.
        gumbel_function: np.ndarray 
            Values of the Gumbel function over the same temperature range (left or right tail).
        tail: str 
            Specifies whether the transition is for the "left" or "right" tail.

    Returns:
        np.ndarray: 
            The smoothly blended transition function values between the logistic and Gumbel functions.
    """

    # Check for which tail transition is performed
    if tail not in {"left", "right"}:
        raise ValueError("Parameter 'tail' must be either 'left' or 'right'.")

    # Check that the arrays have the same size
    if logistic_function.shape != gumbel_function.shape or logistic_function.shape != transition_array.shape:
        raise ValueError("Input arrays must all have the same shape.")

    if tail == "right":
        # Smooth transition for the right tail
        return (1 - transition_array) * logistic_function + transition_array * gumbel_function
    else:
        # Smooth transition for the left tail
        return (1 - transition_array) * gumbel_function + transition_array * logistic_function

# A function for combinning the logistic function with the left and right tailed Gumbel functions together with the respective transition functions
def combine_functions(logistic_function:np.ndarray,
    gumbel_left_function:np.ndarray,
    gumbel_right_function:np.ndarray,
    transition_left:np.ndarray,
    transition_right:np.ndarray,
    quantiles:np.ndarray,
    T_quantiles:np.ndarray,
    T_range:np.ndarray,
    transition_quantiles:tuple) -> np.ndarray:
    
    """
    Combine the logistic function with the left and right-tailed Gumbel functions using the transition functions to construct a smooth, unified function.

    Parameters:
        logistic_function: np.ndarray 
            Values of the logistic function over the temperature range.
        gumbel_left_function: np.ndarray 
            Values of the left-tail Gumbel function over the temperature range.
        gumbel_right_function: np.ndarray 
            Values of the right-tail Gumbel function over the temperature range.
        transition_left: np.ndarray
            Smooth transition values between the logistic and left-tail Gumbel function.
        transition_right: np.ndarray 
            Smooth transition values between the logistic and right-tail Gumbel function.
        quantiles: np.ndarray
            Array of quantiles (usually 0.01...0.99) in ascending order
        T_quantiles: np.ndarray 
            Array of temperature quantiles, e.g., 1st, 10th, 90th, and 99th percentiles.
        T_range: np.ndarray
            Array of temperature values representing the range over which the functions are defined.
        target_quantiles: tuple
            Quantiles at which transition from Gumbel function to logistic function ends.

    Returns:
        np.ndarray: 
            Combined function values over the entire temperature range
    """

    # Sort the random quantiles and the corresponding temperature quantiles
    sorted_indices = np.argsort(T_quantiles)
    T_quantiles_sorted = T_quantiles[sorted_indices]

    # Find the indices of the nearest quantiles
    index_left = get_element_index(quantiles, transition_quantiles[0])
    index_right = get_element_index(quantiles, transition_quantiles[1])

    # Create an empty numpy array for the combined function
    combined_function = np.full_like(T_range, np.nan)
    
    # Define the transition temperatures
    T1 = T_quantiles_sorted[0]
    T_transition_left = T_quantiles_sorted[index_left]
    T_transition_right = T_quantiles_sorted[index_right]
    T99 = T_quantiles_sorted[-1]

    # Find the corresponding indices for the transition temperatures in the T_range array           
    T1_idx = get_element_index(T_range, T1)
    T_transition_left_idx = get_element_index(T_range, T_transition_left)
    T_transition_right_idx = get_element_index(T_range, T_transition_right)
    T99_idx = get_element_index(T_range, T99)
    
    #--- Merge the left, center, and right parts of the function ---#

    # Left tail Gumbel function up to the 1st quantile
    combined_function[:T1_idx] = gumbel_left_function[:T1_idx]

    # Check for the shape of the left tail transition function
    assert transition_left.shape[0] == T_transition_left_idx - T1_idx, \
        f"Mismatch in left transition shape: expected {T_transition_left_idx - T1_idx}, got {transition_left.shape[0]}"

    # Transition from left-tail Gumbel function to the logistic function between the 1st and the 20th quantiles
    combined_function[T1_idx:T_transition_left_idx] = transition_left
    
    # Logistic function between the two transition points
    combined_function[T_transition_left_idx:T_transition_right_idx] = logistic_function[T_transition_left_idx:T_transition_right_idx]
    
    # Check for the shape of the right tail transition function
    assert transition_right.shape[0] == T99_idx - T_transition_right_idx, \
        f"Mismatch in right transition shape: expected {T99_idx - T_transition_right_idx}, got {transition_right.shape[0]}"

    # Transition from the logistic function to the rigth-tail Gumbel function between the 80th and the 99th quantiles
    combined_function[T_transition_right_idx:T99_idx] = transition_right

    # Right tail Gumbel function above the 99th quantile
    combined_function[T99_idx:] = gumbel_right_function[T99_idx:]

    return combined_function

# Fit Gumbel-functions for the tails and the Generalized logistic funtion to the middle
def estimate_CDF_and_PDF_from_quantiles(quantiles:np.ndarray, T_quantiles:np.ndarray, T_range:np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    """
    Fit Gumbel functions for the tails and a 5-parameter logistic function to the middle 
    of the temperature quantiles to estimate the CDF and PDF.

    Parameters:
        quantiles: np.ndarray
            Array of quantiles (usually 0.01...0.99)
        T_quantiles: np.ndarray
            Array of temperature quantiles
        T_range: np.ndarray
            Range of temperatures to which the distribution is fitted
        
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - CDF_combined: Combined cumulative distribution function.
            - PDF_combined: Combined probability density function.
    """

    # Define quantile levels at which logistic function is merged with Gumbel distributions
    merging_quantiles = (0.05, 0.95)

    # Define quantile levels at which the CDF and PDF fully transition from Gumbel-distribution to logistic distribution
    transition_quantiles = (0.10, 0.90)

    # Fit a 5-parameter logistic function to the quantiles 0.01...0.99
    CDF_logistic, PDF_logistic = fit_log5pl(quantiles, T_quantiles, T_range)

    # Get Gumbel-distribution parameters for both tails
    gumbel_params = get_gumbel_parameters(CDF_logistic, PDF_logistic, T_range, merging_quantiles)

    # Get gumbel parameters for the tail distributions
    #gumbel_params = get_gumbel_parameters_from_quantiles(CDF_logistic, PDF_logistic, T_quantiles, T_range)

    # Compute the CDF and PDF of the Gumbel-function for the left and right tails
    CDF_gumbel_left, CDF_gumbel_right, PDF_gumbel_left, PDF_gumbel_right = fit_gumbel_functions(gumbel_params, quantiles, T_quantiles, T_range, transition_quantiles)

    # Get the transition functions
    CDF_transition_left, CDF_transition_right = fit_transition_functions(CDF_logistic, CDF_gumbel_left, CDF_gumbel_right, quantiles, T_quantiles, T_range, transition_quantiles)
    PDF_transition_left, PDF_transition_right = fit_transition_functions(PDF_logistic, PDF_gumbel_left, PDF_gumbel_right, quantiles, T_quantiles, T_range, transition_quantiles)
    
    # Combine CDF and PDF by merging the logistic and Gumbel functions
    CDF_combined = combine_functions(CDF_logistic, CDF_gumbel_left, CDF_gumbel_right, CDF_transition_left, CDF_transition_right, quantiles, T_quantiles, T_range, transition_quantiles)
    PDF_combined = combine_functions(PDF_logistic, PDF_gumbel_left, PDF_gumbel_right, PDF_transition_left, PDF_transition_right, quantiles, T_quantiles, T_range, transition_quantiles)

    return CDF_combined, PDF_combined

def get_mm_distributions(qr_obs_mm_dict:dict, quantiles:np.ndarray, T_range:np.ndarray, doy_index:int)->tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    
    """
    Fits a continuous probability distribution to sets of quantiles that were obtained by applying quantile regression to:
        - model mean (mm) pseudo-observations in pre-industrial climate
        - model mean (mm) pseudo-observations in present-day climate
        - model mean (mm) pseudo-observations in future climate (Optional)
        - plain observations (Optional)

    Parameters:
        qr_obs_mm_dict:dict
            Dictionary contains the temperature quantiles of at least the 'pre-industrial' and 'present-day' climates and possibly 'future' climate and the 'observed' climate.
        quantiles: np.ndarray
            Array of quantiles (usually 0.01...0.99)
        T_range: np.ndarray
            Range of temperatures to which the distribution is fitted
        doy_index: int
            day of year index
        
    Returns:
        tuple[dict, dict]: 
            - CDF_mm_dict: Contains continuous CDFs
            - PDF_mm_dict: Contains continuous CDFs
    """

    # Fit probability distributions for quantile distributions corresponding to obsrvations and pseudo-observations in pre-industrial, present-day and future climate
    CDF_mm_dict = {}
    PDF_mm_dict = {}

    for name, T_quantiles in qr_obs_mm_dict.items():

        # Get CDF and PDF
        CDF, PDF = estimate_CDF_and_PDF_from_quantiles(quantiles, T_quantiles[doy_index].values, T_range)
        CDF_mm_dict[name], PDF_mm_dict[name] = CDF, PDF

    return CDF_mm_dict, PDF_mm_dict

def get_CDF_single_models(qr_sm: pd.DataFrame, quantiles:np.ndarray, T_range:np.ndarray) -> np.ndarray:

    """
    Fits a continuous probability distribution to sets of quantiles that were obtained by applying quantile regression to:
        - single model (sm) pseudo-observations in pre-industrial climate
        - single model (sm) pseudo-observations in present-day climate
        - single model (sm) pseudo-observations in future climate (Optional)
        
    Parameters:
        qr_sm_dict:dict
            Dictionary contains the temperature quantiles of at least the 'pre-industrial' and 'present-day' climates and possibly 'future' climate.
        quantiles: np.ndarray
            Array of quantiles (usually 0.01...0.99)
        T_range: np.ndarray
            Range of temperatures to which the distribution is fitted
        
    Returns:
        CDF_sm: dict 
            Contains continuous single-model CDFs
            
    """

    CDF_sm = []

    # Loop over models
    for model in qr_sm.columns:
        
        # Get T quantiles
        T_quantiles = qr_sm[model].values

        # Fit a cumulative distribution function
        CDF, _ = estimate_CDF_and_PDF_from_quantiles(quantiles, T_quantiles, T_range)
        CDF_sm.append(CDF)

    return np.asarray(CDF_sm)

def monotonic(x):
    """Check monotonicity of an array"""
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

def monotonic_sm(CDF_sm):
    """Check monotonicity for all models."""
    n_models = CDF_sm.shape[0]
    for m in range(n_models):
        if not monotonic(CDF_sm[m,:]):
            return False
    return True

###########################################################################################################################################################################################

################################################################################ Functions for bootstrapping ##############################################################################

def apply_bootstrapping(
    n_boots: int,
    doy_index: int,
    T_range: np.ndarray,
    quantiles: np.ndarray,
    qr_obs_mm_dict:dict,
    qr_sm_dict:dict) -> dict:

    """
    Applies bootstrapping to single-model temperature quantiles, fits a continuous CDF and saves the results.
    Bootstrapping is applied to:
        - single model (sm) quantiles in pre-industrial climate
        - single model (sm) quantiles in present-day climate
        - single model (sm) quantiles in future climate (Optional)
        - observation quantiles (Optional)

    Parameters:
        n_boots: int
            The number of boostrap samples
        doy_index: int
            day of year index
        T_range: np.ndarray
            The range of temperature to which CDF is fitted
        quantiles: np.ndarray
            An array of quantiles (0.01...0.99 usually)
        qr_obs_mm_dict:dict,
            Dictionary which contains quantiles from observations
        qr_sm_dict:dict
            Dictionary contains the temperature quantiles of at least the 'pre-industrial' and 'present-day' climates and possibly 'future' climate.
        
    Returns:
        boostrap_dict: dict 
            Contains continuous single-model CDFs for all the bootstrap cases
            
    """

    # Get the QR-dataframes from the dictionaries
    quantiles_sm_preind = qr_sm_dict["preind"]
    quantiles_sm_target = qr_sm_dict["target"]
    quantiles_sm_future = qr_sm_dict.get("future", None)
    quantiles_obs = qr_obs_mm_dict.get("obs", None)

    # Number of models
    n_models = quantiles_sm_preind.shape[1]

    # Check if obs and future quantiles are used
    use_obs = quantiles_obs is not None
    use_future = quantiles_sm_future is not None

    # Initialize empty arrays
    CDF_sm_preind_array = np.empty((len(T_range), n_models, n_boots+1))
    CDF_sm_target_array = np.empty((len(T_range), n_models, n_boots+1))

    if use_obs:
        CDF_obs_array = np.empty((len(T_range), n_boots+1))
    
    if use_future:
        CDF_sm_future_array = np.empty((len(T_range), n_models, n_boots+1))

    #np.random.seed(42)
    pbar = tqdm(total=n_boots+1, desc="Bootstrapping", unit="bootstrap iteration")

    # Apply bootstrapping to single-model quantiles for 'n_boots' + 1 (deterministic case) times 
    b = 0
    while b < n_boots+1:
        
        if b == 0:

            # Deterministic bootstrap
            random_quantiles = quantiles_sm_preind.index

            bootstrap_sample_preind = quantiles_sm_preind.loc[random_quantiles]
            bootstrap_sample_target = quantiles_sm_target.loc[random_quantiles]
            
            if use_obs:
                bootstrap_sample_obs = quantiles_obs.loc[random_quantiles, doy_index]
            
            if use_future:
                bootstrap_sample_future = quantiles_sm_future.loc[random_quantiles]
            
        else:
            # Random bootstrap quantile sample (0.01,...,0.99) taken by replacement   
            random_quantiles = quantiles_sm_preind.index.to_series().sample(n=99,axis=0,replace=True)

            # Take the same random quantile sample from the pre-industrial and present-day climates
            bootstrap_sample_preind = quantiles_sm_preind.loc[random_quantiles]
            bootstrap_sample_target = quantiles_sm_target.loc[random_quantiles]

            # Take the "same random quantile sample" from observations if they are used
            if use_obs:
                bootstrap_sample_obs = quantiles_obs.loc[random_quantiles, doy_index]
            
            # Take the same random quantile sample from future climate if it is used
            if use_future:
                bootstrap_sample_future = quantiles_sm_future.loc[random_quantiles]
           
        try:
            # Estimate CDFs for observations and pseudo-observations
            CDF_sm_preind = get_CDF_single_models(bootstrap_sample_preind, quantiles, T_range)
            CDF_sm_target = get_CDF_single_models(bootstrap_sample_target, quantiles, T_range)
            
            if use_obs:
                CDF_obs, _ = estimate_CDF_and_PDF_from_quantiles(quantiles, bootstrap_sample_obs.values, T_range)
            
            if use_future:
                CDF_sm_future = get_CDF_single_models(bootstrap_sample_future, quantiles, T_range)
            
        except ValueError as e:
            continue
        
        # Check for monotonicity as a sanity check
        if not monotonic_sm(CDF_sm_preind):
            continue

        if not monotonic_sm(CDF_sm_target):
            continue
        
        if use_obs and not monotonic(CDF_obs):
            continue

        if use_future and not monotonic_sm(CDF_sm_future):
            continue

        # Save the monotonic CDFs
        CDF_sm_preind_array[:, :, b] = CDF_sm_preind.T
        CDF_sm_target_array[:, :, b] = CDF_sm_target.T
        
        if use_obs:
            CDF_obs_array[:, b] = CDF_obs
                    
        if use_future:
            CDF_sm_future_array[:, :, b] = CDF_sm_future.T

        b+=1
        pbar.update(1)
    
    pbar.close()

    # Store results in dictionary
    bootstrap_dict = {
        "preind": CDF_sm_preind_array,
        "target": CDF_sm_target_array}
    
    if use_obs:
        bootstrap_dict["obs"] = CDF_obs_array
    
    if use_future:
        bootstrap_dict["future"] = CDF_sm_future_array

    return bootstrap_dict

# Reshape the 3D arrays within the 'bootstrap_dict' dictionary to 2D arrays
def reshape_bootstrap_arrays(bootstrap_dict:dict[str, np.ndarray])->dict[str, np.ndarray]:

    """
    Reshapes the 3D arrays containing single-model CDFs to 2D arrays
        
    Parameters:
        boostrap_dict: dict 
            Contains continuous 3D single-model CDFs for all the bootstrap cases
        
    Returns:
        bootstrap_2D_arrays: dict 
            Contains the reshaped CDF arrays (model dimension is dropped)
    """

    bootstrap_2D_arrays = {}

    # Save the CDFs of the pre-industrial and present-day climates
    for key in ("preind", "target"):
        arr_3d = bootstrap_dict[key]
        bootstrap_2D_arrays[key] = arr_3d.reshape(*arr_3d.shape[:-2], -1)

    # Save the CDFs of the observations (if they're in the input argument dictionary)
    if "obs" in bootstrap_dict:
        bootstrap_2D_arrays["obs"] = bootstrap_dict["obs"]

    # Save the CDFs of the future-climate (if they're in the input argument dictionary)
    if "future" in bootstrap_dict:
        arr_3d = bootstrap_dict["future"]
        bootstrap_2D_arrays["future"] = arr_3d.reshape(*arr_3d.shape[:-2], -1)

    return bootstrap_2D_arrays

############################################################################################################################################################################################

######################################################################## Functions for extracting diagnostics and plotting data ############################################################

def find_nearest(array:np.ndarray, value:float) -> float:
     
    """
    Find the nearest value in a given array to a specified target value
    """

    # Compute the absolute differences between each element and the target value
    # and find the index of the smallest difference
    array = np.asarray(array)
    diff = np.abs(array - value)
    idx = (diff).argmin()
    return array[idx]

def find_intensity_interval(T_range:np.ndarray, CDF_target_2Darray:np.ndarray, CDF_preind_2Darray:np.ndarray, index:int) -> tuple[float, float]:

    """
    Calculate the intensity interval of a target CDF compared to a pre-industrial CDF over all models.

    Parameters:
    T_range: np.ndarray
        The temperature range array used for computing the CDFs.
    CDF_target_2Darray: np.ndarray
        A 2D array representing the target CDF values, with dimensions [temperature, model].
    CDF_comparison_2Darray: np.ndarray
        A 2D array representing the comparison CDF values, with dimensions [temperature, model].
    index: int
        Index specifying the target temperature level in `CDF_target_2Darray`.

    Returns:
    interval : tuple[float, float]
        A tuple containing the 5th and 95th percentile of the intensity interval.
    test_list : list[np.ndarray]
        A list of temperature values corresponding to the target CDF probabilities across models.
    """

    temperatures = []

    for I in np.arange(0, np.shape(CDF_preind_2Darray)[1]):
        
        # Extract target CDF probability for the given temperature and model
        PROB = CDF_target_2Darray[index,I]
        if np.isnan(PROB):
            continue 
        # Find the nearest probability in the comparison CDF array
        nearest = find_nearest(CDF_preind_2Darray[:,I],PROB)
        
        if np.isnan(nearest):
            continue

        # Find the corresponding temperature index
        ind = np.where(CDF_preind_2Darray[:,I] == nearest)[0]
        # Extract the temperature corresponding to the nearest CDF value 
        TEMP = np.squeeze(T_range[ind])
        
        temperatures.append(TEMP)

    return (np.nanpercentile(temperatures, 5), np.nanpercentile(temperatures, 95))


def find_difference_interval(T_range:np.ndarray, CDF_target_2Darray:np.ndarray, CDF_preind_2Darray:np.ndarray, i:int) -> tuple[float, float, pd.Series]:
    
    """
    Calculate the difference in intensity between the target and pre-industrial CDFs for a given temperature level.

    Parameters:
    ----------
    T_range : np.ndarray
        The temperature range array used for computing the CDFs.
    CDF_target_2Darray : np.ndarray
        A 2D array representing the target CDF values, with dimensions [temperature, model].
    CDF_preind_2Darray : np.ndarray
        A 2D array representing the pre-industrial CDF values, with dimensions [temperature, model].
    i : int
        Index specifying the target temperature level in `CDF_target_2Darray`.

    Returns:
    -------
    interval : tuple[float, float]
        A tuple containing the 5th and 95th percentile of the intensity differences.
    intensity_diff_series : pd.Series
        A pandas Series of intensity differences across models.
    """

    intensity_difference_series = pd.Series(index=np.arange(0, np.shape(CDF_target_2Darray)[1]), dtype=float)
    for m in np.arange(0, np.shape(CDF_target_2Darray)[1]):
        
        # Calculate the probability in the present climate
        target_cdf = CDF_target_2Darray[:, m]
        preind_cdf = CDF_preind_2Darray[:, m]

        PROB = target_cdf[i]

        if np.isnan(PROB):
            continue
        
        #target_temp = np.round(np.squeeze(T_range[np.where(target_cdf == find_nearest(target_cdf,PROB))[0]]),1)
        target_temp = np.squeeze(T_range[np.where(target_cdf == find_nearest(target_cdf,PROB))[0]][0])
        
        
        #preind_temp = np.round(np.squeeze(T_range[np.where(preind_cdf == find_nearest(preind_cdf,PROB))[0]]),1)
        preind_temp = np.squeeze(T_range[np.where(preind_cdf == find_nearest(preind_cdf,PROB))[0]][0])

        # Skip this part if either array is empty
        if target_temp.size == 0 or preind_temp.size == 0:
            continue  

        if np.sum(np.isfinite(target_cdf)) > 0:
            intensity_difference_series[m] = target_temp - preind_temp
    
    return (np.nanpercentile(intensity_difference_series, 5), np.nanpercentile(intensity_difference_series, 95), intensity_difference_series)



def calculate_probabilities(pwarm: bool,
    index:int,
    CDFs_mm_dict:dict[str,np.ndarray],
    bootstrap_3Darr_dict:dict[str,np.ndarray] | None=None) -> dict:
    
    """
    Calculates probabilities of exceeding or being below a certain threshold based on CDFs,
    along with their 5-95% confidence intervals derived from bootstrap sampling.

    Parameters:
        pwarm: bool
            If True, computes exceedance probabilities (1 - CDF); if False, computes non-exceedance probabilities (CDF).
        index: int
            Index of the temperature value in the CDF arrays.
        CDFs_mm_dict: dict
            Dictionary which contains the model-mean CDFs for obs, preind, target and future climates
        bstrap_results_dict:
            Dictionary which contains the single model bootstrapped CDFs for obs, preind, target and future climates (Optional)

    Returns:
        dict: A dictionary containing probability estimates, and if 'bstrap_results_dict' is provided, the corresponding confidence intervals
            
    """
    def get_cdf(cdf:np.ndarray, pwarm:bool):
        if pwarm == True:
            return 1. - cdf
        else:
            return cdf

    def get_confidence_interval(array:np.ndarray,index:int):
        if pwarm:
            return (
                1. - np.nanpercentile(array, 95, axis=1)[index],
                1. - np.nanpercentile(array, 5, axis=1)[index]
            )
        else:
            return (
                np.nanpercentile(array, 95, axis=1)[index],
                np.nanpercentile(array, 5, axis=1)[index]
            )

    probabilities_dict = {}

    # Loop over the model-mean CDFs and calculate probabilities
    for key, cdf in CDFs_mm_dict.items():
        probabilities_dict[f"{key}"] = get_cdf(cdf,pwarm)[index]

    # Compute confidence intervals if bootstrapping results are provided as an argument     
    if bootstrap_3Darr_dict is not None:
        bootstrap_2Darr_dict = reshape_bootstrap_arrays(bootstrap_3Darr_dict)

        for key, cdf_array in bootstrap_2Darr_dict.items():
            low, up = get_confidence_interval(cdf_array,index)
            probabilities_dict[f"{key}_low"] = low
            probabilities_dict[f"{key}_up"] = up


    # Calculate probability ratios from the model-mean CDFs    
    if pwarm:
        pr_ratio = probabilities_dict["target"] / probabilities_dict["preind"]
    else:
        pr_ratio = probabilities_dict["preind"] / probabilities_dict["target"]
    
    probabilities_dict["pr_ratio"] = pr_ratio

    # Calculate confidence intervals for the probability ratio
    if bootstrap_3Darr_dict is not None:
        prob_preind_bstrap = bootstrap_2Darr_dict["preind"][index, :]
        prob_target_bstrap = bootstrap_2Darr_dict["target"][index, :]

        if pwarm:
            pr_ratio_bstrap = (1. - prob_target_bstrap) / (1. - prob_preind_bstrap)
        else:
            pr_ratio_bstrap = prob_preind_bstrap / prob_target_bstrap

        probabilities_dict["pr_ratio_low"] = np.nanpercentile(pr_ratio_bstrap, 5)
        probabilities_dict["pr_ratio_up"] = np.nanpercentile(pr_ratio_bstrap, 95)

    return probabilities_dict

def get_percentile_temp_in_climates(prob:float, T_range:np.ndarray, CDF_mm_dict:dict[str,np.ndarray])-> dict[str,float]:

    """
    Calculates temperatures corresponding to a given probability level in different climate periods.

    Parameters
    ----------
    prob: float
        Probability level (e.g. 0.9).
    T_range: np.ndarray
        Temperature values corresponding to the CDF grid.
    CDFs_mm_dict: dict[str, np.ndarray]
        Dictionary of CDFs (e.g. obs, pseudo_obs_preind, pseudo_obs_future).

    Returns
    -------
    dict[str, float]
        Temperatures at the given probability level.
    """

    def temp_at_prob(cdf:np.ndarray, T_range:np.ndarray, prob:float)->float:
        index = np.nanargmin(np.abs(cdf - prob))
        return float(np.round(T_range[index], 1))

    temperatures = {}

    # Find the temperatures that correspond to the given probability in pre-industrial, observed and future climates
    if "obs" in CDF_mm_dict:
        temperatures["t_y1base"] = temp_at_prob(CDF_mm_dict["obs"], T_range, prob)
        #temperatures["t_y1base"] = np.round(np.squeeze(T_range[np.where(CDFs_mm_dict["obs"] == get_element_index(CDFs_mm_dict["obs"], prob))[0]]), 1)
    
    temperatures["t_preind"] = temp_at_prob(CDF_mm_dict["preind"], T_range, prob)
    #t_preind = np.round(np.squeeze(T_range[np.where(CDFs_mm_dict["pseudo_obs_preind"] == get_element_index(CDFs_mm_dict["pseudo_obs_preind"], prob))[0]]), 1)
    
    if "future" in CDF_mm_dict:
        temperatures["t_future"] = temp_at_prob(CDF_mm_dict["future"], T_range, prob)
        #t_future = np.round(np.squeeze(T_range[np.where(CDFs_mm_dict["pseudo_obs_future"] == get_element_index(CDFs_mm_dict["pseudo_obs_future"], prob))[0]]), 1)

    return temperatures

def calculate_dI_intervals(index:int, T_range:np.ndarray, bootstrap_3Darr_dict:dict[str,np.ndarray]) -> dict:

    """
    Calculates intensity intervals by determining temperature ranges and differences 
    between different climate scenarios.

    Parameters:
        index: int
            The index corresponding to the probability level in the CDF arrays.
        T_range: np.ndarray
            Array of temperature values corresponding to CDF values.
        bstrap_results_dict: dictionary
            Dictionary of single-model bootstrapping results

    Returns:
        dict: A dictionary containing the lower and upper bounds of temperature intensities:
            - 't_preind_lower': Lower bound for the pre-industrial temperature interval.
            - 't_preind_upper': Upper bound for the pre-industrial temperature interval.
            - 't_future_lower': Lower bound for the future temperature interval.
            - 't_future_upper': Upper bound for the future temperature interval.
            - 'tdiff_lower': Lower bound for the temperature difference interval.
            - 'tdiff_upper': Upper bound for the temperature difference interval.
            - 'intensity_diff': DataFrame containing the intensity difference calculations.
    """

    intensity_intervals_dict = {}

    # Extract bootstrapping results to arrays
    bootstrap_2Darr_dict = reshape_bootstrap_arrays(bootstrap_3Darr_dict)

    # Intensity intervals in the pre-industrial climate
    intensity_intervals_dict["t_preind_lower"], intensity_intervals_dict["t_preind_upper"] = find_intensity_interval(T_range, bootstrap_2Darr_dict["target"], bootstrap_2Darr_dict["preind"], index)

    # Intensity intervals in the future climate
    if "future" in bootstrap_2Darr_dict:
        intensity_intervals_dict["t_future_lower"], intensity_intervals_dict["t_future_upper"] = find_intensity_interval(T_range, bootstrap_2Darr_dict["target"], bootstrap_2Darr_dict["future"], index)
    
    # Temperature differences between pre-industrial and present-day climates
    intensity_intervals_dict["tdiff_lower"], intensity_intervals_dict["tdiff_upper"], intensity_intervals_dict["intensity_diff"] = find_difference_interval(T_range, bootstrap_2Darr_dict["target"], bootstrap_2Darr_dict["preind"], index)
    
    return intensity_intervals_dict


def print_attribution_results(
    attribution_cases:dict,
    obs_source:str,
    station_meta:dict,
    clim_var:str,
    doy_index:int,
    n_days:int,
    target_value:float,
    probabilities:dict,
    percentile_temps:dict,
    dI_intervals:dict | None=None) -> None:

    """
    Prints annual probabilities, temperatures and probability ratios and intensity change between the pre-industrial and present-day climates.

    Parameters:
        attribution_cases: dict
            Contains all cases for which attribution is done:
                - Pre-industrial year
                - Present-day year
                - Future year (Optional)
                - Observations (Optional)
        obs_source: str
            Source of observation (e.g. FMI, SMHI, FROST)
        station_meta:dict
            Contains details about the station (e.g. name, lat, lon)
        clim_var: str
            Climate variable (tas, tasmax, tasmin)
        doy_index: int
            Day of year index
        n_days: int
            Number of days in the time-period
        target_value: int
            The observed temperature in the present-day climate for which attribution is done
        probabilities: dict
            Contains probability estimates in pre-industrial, present-day, future (Optional) and observed climate (Optional)
        percentile_temps: dict
            Contains temperature corresponding to a given percentile in pre-industrial, present-day, future (Optional) and observed climate (Optional)
        dI_intervals: dict
            Intensity change interval values between the pre-industrial and present-day climate
    
    Returns:
        None: Attribution results are printed for the given station
    """

    def format_var(var_type:str, var:float, var_low:float | None=None, var_up:float | None=None):
        if var_type == "prob":
            if var_low is None or var_up is None:
                return f"{var*100:.2f}%"
            else:
                return f"{var*100:.2f}% ({var_low*100:.2f}% - {var_up*100:.2f}%)"
        elif var_type == "temp":
            if var_low is None or var_up is None:
                return f"{var:.1f}°C"
            else:
                return f"{var:.1f}°C ({var_low:.1f}°C - {var_up:.1f}°C)"
        elif var_type == "pr_ratio":
            if var_low is None or var_up is None:
                return f"{var:.1f}"
            else:
                return f"{var:.1f} ({var_low:.1f} - {var_up:.1f})"
        else:
            return None


    clim_var_map={
        "tas": "daily mean temperature",
        "tasmax": "daily maximum temperature",
        "tasmin": "daily minimum temperature"
    }

    # Get a time-period date string
    time_period, _ = get_date_strings(doy_index, n_days)

    if n_days > 1:
        var_string = f"{str(n_days)}-day mean of {clim_var_map.get(clim_var, clim_var)}"
    else:
        
        # Labels for a single-day case
        var_string = f"{clim_var_map.get(clim_var, clim_var)}"
    
    # Station metadata
    station_id = station_meta["station_id"]
    name = station_meta["name"]
    lat = station_meta["latitude"]
    lon = station_meta["longitude"]

    # Get probabilities
    pr_preind = probabilities["preind"]
    pr_preind_low = probabilities.get("preind_low")
    pr_preind_up = probabilities.get("preind_up")

    pr_target = probabilities["target"]
    pr_target_low = probabilities.get("target_low")
    pr_target_up = probabilities.get("target_up")

    pr_ratio = probabilities["pr_ratio"]
    pr_ratio_low = probabilities.get("pr_ratio_low")
    pr_ratio_up = probabilities.get("pr_ratio_up")

    # Limits for intensity changes
    if dI_intervals is not None:
        t_preind_low = dI_intervals.get("t_preind_lower")
        t_preind_upper = dI_intervals.get("t_preind_upper")
        tdiff_lower = dI_intervals["tdiff_lower"]
        tdiff_upper = dI_intervals["tdiff_upper"]

    # Extract temperature
    t_preind = percentile_temps["t_preind"]
    
    # Intensity change
    dI = target_value - t_preind

    # Get years
    preind_year = attribution_cases.get("preind")
    target_year = attribution_cases.get("target")
    future_year = attribution_cases.get("future")

    # Print the results
    print(f"Attribution results for {obs_source} station {station_id} ({name.replace('_', ' ').title()}):\n")
    print(f"Location: ({np.round(lat,2)} °N, {np.round(lon,2)} °E)")
    print(f"Climate variable: {var_string}")
    print(f"Time-period: {time_period}")
    print("\nAnnual probabilities:")
    print(f"{preind_year}: {format_var("prob", pr_preind, pr_preind_low, pr_preind_up)}")
    print(f"{target_year}: {format_var("prob", pr_target, pr_target_low, pr_target_up)}")
    
    # Print probability for future year only if it's provided
    if future_year and "future" in probabilities:
        
        print(f"{future_year}: {format_var("prob", probabilities["future"], probabilities.get("future_low"), probabilities.get("future_up"))}")
        
    print("\nAnnual temperatures:")
    if dI_intervals is not None:
        print(f"{preind_year}: {format_var("temp", t_preind, t_preind_low, t_preind_upper)}")
    else:
        print(f"{preind_year}: {format_var("temp", t_preind, None, None)}")

    print(f"{target_year}: {np.round(target_value,1)}°C")
    
    # Print temperatures for future year only if it's provided
    if future_year and "t_future" in percentile_temps:
        if dI_intervals is not None:
            print(f"{future_year}: {format_var("temp", percentile_temps["t_future"], dI_intervals.get("t_future_lower"), dI_intervals.get("t_future_upper"))}")
        else:
            print(f"{future_year}: {format_var("temp", percentile_temps["t_future"], None, None)}")

    # Print probability ratios
    print(f"\nProbability ratio: {format_var("pr_ratio", pr_ratio, pr_ratio_low, pr_ratio_up)}")

    if dI_intervals is not None:
        print(f"Change in intensity: ({dI:.1f}°C) ({tdiff_lower:.1f}-{tdiff_upper:.1f})")
    else:
        print(f"Change in intensity: {dI:.1f}°C")
    
def plot_time_series(path2figures:str,
    clim_var:str,
    place:str,
    target_year:int,
    obs_df:pd.DataFrame,
    pseudo_obs_mm_target_df:pd.DataFrame,
    pseudo_obs_sm_target_df:pd.DataFrame,
    doy_index:int,
    n_days:int)-> None:

    import matplotlib.lines as mlines

    """
    Plots a time-series of observations, pseudo-observations and model uncertainty from the chosen baseline period (usually 1901-2024). 

    Parameters:
        path2figures: str 
            Directory to save the plot
        clim_var: str
            Climate variable (tas, tasmax, tasmin)
        place: str
            Name of the weather station
        target_year: int
            Year of target observation
        obs_df: pd.DataFrame
            Temperature observations (rows = years, day_of_year = columns)
        pseudo_obs_mm_target_df: pd.DataFrame
            Model-mean pseudo-observations (rows = years, day_of_year = columns)
        pseudo_obs_sm_target_df: pd.DataFrame
            Single-model pseudo-observations (rows = years, day_of_year = columns)
        doy_index: int
            Day of year index
        n_days: int
            Number of days in the time-period
    
    Returns:
        None: The figure is saved as a PNG file.
    """

    # Strings correponsing to the given climate variable
    clim_var_map={
        "tas": "daily mean temperature",
        "tasmax": "daily maximum temperature",
        "tasmin": "daily minimum temperature"
    }

    # Determine the first year in the time-series
    if pseudo_obs_mm_target_df.index.min() < 1901:
        start_year = 1901
    else:
        start_year = pseudo_obs_mm_target_df.index.min()

    # Construct a time-period string 
    time_period, figname_date_string = get_date_strings(doy_index, n_days)
    
    # Construct the y-axis label
    if n_days > 1:
        ylabel_string = f"{str(n_days)}-day mean of \n {clim_var_map.get(clim_var, clim_var)} [$^\\circ$C]"        
    else:
        ylabel_string = f"{clim_var_map.get(clim_var, clim_var)} [$^\\circ$C]"
        
    # Time series of daily temperature observations up to the target year
    obs_temp_plot = obs_df.loc[slice(str(start_year),str(target_year)),doy_index]

    # Time series of daily mean temperature pseudo-observations in the target year climate
    target_obs = pseudo_obs_mm_target_df.loc[slice(str(start_year),str(target_year))][doy_index]
    
    # Take 5-95 % of the model spread
    target_up = pseudo_obs_sm_target_df.groupby(level="year").quantile(0.95)[doy_index].loc[slice(start_year,target_year)]
    target_low = pseudo_obs_sm_target_df.groupby(level="year").quantile(0.05)[doy_index].loc[slice(start_year,target_year)]

    # Define error bars
    y_l = np.array(target_obs) - np.array(target_low)
    y_u =np.array(target_up) - np.array(target_obs)
    errors = [y_l, y_u]

    # Plot the results
    fig=plt.figure(figsize=(10,5))
    ax=plt.gca()

    # Plot observed temperatures
    ax.plot(obs_temp_plot.index, obs_temp_plot, color='k', linewidth=1.5, label='Observations')

    # Plot pseudo-observations with error bars
    ax.scatter(target_obs.index.values, target_obs, label=f'Pseudo-observations in {target_year}')
    ax.errorbar(target_obs.index.values, target_obs, yerr=errors, fmt='o', ecolor = 'red')

    # Highlight the target year with a horizontal line
    ax.axhline(y=obs_temp_plot.loc[target_year], linestyle='--')

    # Adjust axes, legend, etc.
    ax.set_title(f"{place.replace('_', ' ').title()}, {time_period}", fontsize=15, pad=25)
    ax.set_ylabel(ylabel_string,fontsize=14)
    ax.set_xlabel("Year",fontsize=14)
    ax.set_xlim(start_year-1, target_year+1)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.55, 1.09), ncol=3, fontsize=12)

    # Save the figure
    figure_name = f"time_series_plot_{clim_var}_{place}_{figname_date_string}.png"
    plt.savefig(os.path.join(path2figures,clim_var,figure_name),dpi=300,bbox_inches="tight")


def plot_observations(
        path2figures:str,
        clim_var:str,
        place:str,
        target_year:int,
        obs_df:pd.DataFrame,
        qr_obs_df:pd.DataFrame,
        doy_index:int,
        n_days:int)-> None:
    
    """
    Plots all observed daily temperature observations or their 2 to 31-day running means from a selected weather station and a few quantile functions (e.g. 0.01, 0.10, 0.50, 0.90, 0.99).

    Args:
        path2figures: str
            Directory to save the plot.
        clim_var: str
            Climate variable (e.g. tas, tasmax or tasmin)
        place: str 
            Name of the weather station
        target_year: int 
            Year of target observation
        obs_df: pd.DataFrame
            Temperature observations (rows = years, day_of_year = columns)
        qr_obs_df: pd.DataFrame 
            Quantile functions. Rows correspond to quantiles (e.g., 0.01, 0.05, ..., 0.99) and columns correspond to day-of-year.
        doy_index (int):
            Day of year index (1-365).
        n_days: int
            Number of days in the time-period.

    Returns:
        None: The figure is saved as a PNG file.
    """

    # For even number of n_days, doy_index is the last day of the period (e.g. for 1-4 Jan, doy_index = 4)
    # This function shifts the doy_index from the last day of the time-period to the latter center day for the purpose of plotting observations
    def shift_days(doy_array, n_days):
        shift = (n_days - 1) // 2
        return ((doy_array - 1 - shift) % 365) + 1
    
    def shift_quantiles(qr_obs:pd.Series, n_days:int)-> pd.Series:
        doy = qr_obs.index.to_numpy()
        shifted_doy = shift_days(doy, n_days)
        shifted_qr_obs = pd.Series(qr_obs.values,index=shifted_doy)

        return shifted_qr_obs.sort_index()
    
    # Strings corresponding to the given climate variable
    clim_var_map = {
        "tas": "mean temperature",
        "tasmax": "maximum temperature",
        "tasmin": "minimum temperature"
    }

    # Get time-period strings for legend and figure name
    time_period, figname_date_string = get_date_strings(doy_index, n_days)

    # Legend label for other than the target observation 
    if n_days > 1:
        obs_label = f"{n_days}-day mean of daily \n {clim_var_map.get(clim_var, clim_var)}"
    else:
        obs_label = f"{clim_var_map.get(clim_var, clim_var)}"
        
    # Melt pivoted observation DataFrame for plotting
    obs_melted_df = melt_obs_df(obs_df)

    # Get target observation
    target_obs = obs_df.loc[target_year][doy_index]

    # ---- Plot observations ----
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add shading to highlight the range
    
    if n_days % 2 == 1:
        # odd: doy_index is the center day
        half = (n_days - 1) // 2
        start = doy_index - half
        end = doy_index + half
    else:
        # even: doy_index is the end day (trailing window)
        start = doy_index - n_days + 1
        end = doy_index

    ax.axvspan(
        start,
        end,
        color="#d62728",
        alpha=0.15,
        zorder=0)

    # Days of year
    days = np.arange(1, 366, 1)

    # Colors for quantiles
    if qr_obs_df.index.min() < 0.01:
        colors = {
            qr_obs_df.index.min(): 'blue',
            0.01: 'cyan',
            0.1: 'lime',
            0.5: 'yellow',
            0.9: 'orange',
            0.99: 'crimson',
            qr_obs_df.index.max(): 'magenta'}
    else:
        colors = {
            0.01: 'blue',
            0.05: 'cyan',
            0.1: 'lime',
            0.5: 'yellow',
            0.9: 'orange',
            0.95: 'crimson',
            0.99: 'magenta'}
    
    # Plot the observations and store the handles
    if n_days % 2 == 0:
        # For even day case, shift the doy_index and the corresponding observations from the last day to the latter center day
        shifted_obs_doy = shift_days(obs_melted_df["day_of_year"],n_days)
        obs_handle = ax.scatter(shifted_obs_doy, obs_melted_df.loc[shifted_obs_doy.index]["temperature"], alpha=0.3, label=obs_label)
        target_handle = ax.scatter(shift_days(doy_index,n_days), target_obs, color="#d62728", label=f"{time_period} {target_year}", zorder=5)
    else:
        obs_handle = ax.scatter(obs_melted_df["day_of_year"], obs_melted_df["temperature"], alpha=0.3, label=obs_label)
        target_handle = ax.scatter(doy_index, target_obs, color="#d62728", label=f"{time_period} {target_year}", zorder=5)

    # Plot quantiles and store handles
    quantile_handles = []
    for q in colors.keys():
        
        # Get the number of decimals
        decimals = str(q).split(".")[-1]
        if len(decimals) >= 3:
            label = label=f"{q:.3f}"
        else:
            label = label=f"{q:.2f}"
        if n_days % 2 == 0:
            # For even day case, shift the doy_index and the corresponding quantile functions from the last day to the latter center day
            shifted_days = sorted(shift_days(days,n_days))
            shifted_quantiles = shift_quantiles(qr_obs_df.loc[q],n_days)
            line, = ax.plot(shifted_days, shifted_quantiles, label=label,color=colors[q])
        else:
            line, = ax.plot(days, qr_obs_df.loc[q], label=label,color=colors[q])
        quantile_handles.append(line)

    # ---- Legends ----
    
    # Observation legend
    obs_legend_ax = fig.add_axes([0.88, 0.5, 0.15, 0.35])  # left, bottom, width, height
    obs_legend_ax.axis("off")
    obs_legend_ax.legend(handles=[obs_handle, target_handle],
                        labels=[obs_label, f"{time_period} {target_year}"],
                        title=r"$\mathbf{Observations}$",
                        loc="center",
                        fontsize=11,
                        title_fontsize=12)

    # Quantile legend
    quantile_legend_ax = fig.add_axes([0.88, 0.25, 0.15, 0.35])
    quantile_legend_ax.axis("off")
    quantile_legend_ax.legend(handles=quantile_handles,
                            title=r"$\mathbf{Quantiles}$",
                            loc="center",
                            fontsize=12,
                            title_fontsize=13)

    # --- Adjust axes ---
    month_start_days = np.array([1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336, 365])
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June','July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' , 'Jan']
    
    ax.set_xticks(month_start_days)
    ax.set_xticklabels(month_labels, fontsize=13)

    # Set a grid on y-axis
    ax.grid(True, axis="y")

    # Draw vertical lines as a grid for every second month on the x-axis
    for x in month_start_days[::2]:
        ax.axvline(x, linestyle="-", linewidth=0.8, alpha=0.6, color="gray")
    
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month", fontsize=15)
    ax.set_ylabel(r"Temperature [$^{\circ}$C]", fontsize=15)
    ax.set_title(f"{place.replace('_', ' ').title()}", fontsize=20)

    # Make room for legend
    plt.subplots_adjust(left=0.07, right=0.80, top=0.95, bottom=0.10)
    
    # ---- Save the figure ----
    figure_name = f"observation_plot_{clim_var}_{place}_{figname_date_string}.png"
    save_path = os.path.join(path2figures, clim_var, figure_name)
    #plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_distributions(
    path2figures:str,
    attribution_cases:dict,
    obs_source:str,
    station_meta:dict,
    ssp:str,
    T_range:np.ndarray,
    doy_index:int,
    n_days:int,
    pwarm:bool,
    target_value:float,
    pseudo_obs_mm_target:pd.DataFrame,
    PDF_mm_dict:dict[str,np.ndarray],
    probabilities:dict[str,float],
    percentile_temps:dict,
    intensity_intervals:dict,
    )-> None:

    """
    Plots probability distributions for the pre-industrial, present-day and future climates and prints a summary of the attribution statistics.

    Args:
        path2figures: str 
            Directory path to save the resulting plot.
        attribution_cases: dict
            Contains information of all attribution cases (e.g. pre-industrial, present-day and future years)
        obs_source: str
            Source of observations (FMI, SMHI, FROST)
        station_meta: dict
            Station metadata
        ssp: str
            Emission scenario (e.g. spp245)
        T_range: np.ndarray
            Range of temperatures to which the probability distributions are fitted
        doy_index: int
            day_of_year index
        n_days: int
            Number of days in the time-period
        pwarm: bool
            Probabilty for warmer (True)/colder (False) temperatures
        target_value:float
            Target observation
        pseudo_obs_mm_target: pd.DataFrame
            Contains a time-series of model-mean pseudo-observations in the target (present-day) climate. Years = rows, columns=day_of_year.
        PDF_mm_dict: dict
            Contains probability density functions in the pre-industrial, present-day and optionally future and observed climates.
        probabilities: dict
            Contains probability estimates for:
                - Pre-industrial and present-day climates
                - Future and observed climates (Optional)
                - Confidence intervals (Optional)
        percentile_temps: dict
            Contains temperatures corresponding to a given percentile in:
                 - Observed climate over the baseline period (usually 1901-preceding year)
                 - Pre-industrial year
                 - Current climate
                 - Future climate (Optional)
        intensity_intervals (dict):
            Contains the lower and uppoer bounds for temperature intensities in:
                - Pre-industrial climate
                - Future climate (Optional) 
    Returns:
        None: Generates a plot comparing temperature distributions and prints probabilities, return periods and intensities in each climate. The plot in save as a PNG-figure.
    """

    def format_prob(probs: dict, key: str) -> str:
        p = probs[key] * 100

        up_key = f"{key}_low"
        low_key = f"{key}_up"

        if low_key in probs and up_key in probs:
            return f"{p:.2f} ({probs[up_key]*100:.2f}-{probs[low_key]*100:.2f}) %"
        else:
            return f"{p:.2f} %"
        
    def format_recurrence(probs: dict, key: str) -> str:
        p = probs[key]

        up_key = f"{key}_low"
        low_key = f"{key}_up"

        if low_key in probs and up_key in probs:
            return (
                f"{1/p:.0f} "
                f"({1/probs[low_key]:.0f}-{1/probs[up_key]:.0f}) years"
            )
        else:
            return f"{1/p:.0f} years"

    # Extract years
    preind_year = attribution_cases["preind"]
    target_year = attribution_cases["target"]
    if "future" in attribution_cases:
        future_year = attribution_cases["future"]

    # Get the y1base
    if pseudo_obs_mm_target.index.min() < 1901:
        y1base = 1901
    else:
        y1base = pseudo_obs_mm_target.index.min()

    import matplotlib

    # Define the default font
    font = {'weight' : 'normal',
            'size'   : 13}
    matplotlib.rc('font', **font)

        # ---- dates, labels and titles ---
    time_period, fig_name_date_string = get_date_strings(doy_index, n_days)

    # Plot two figures side by side
    fig, axlist=plt.subplots(nrows=1, ncols=2, figsize=(18,7), sharey=False)
    ax1=axlist[0]
    ax2=axlist[1]

    # Initialize a histogram of pseudo-observations in the present-day climate
    hist, bin_edges = np.histogram(pseudo_obs_mm_target.loc[y1base:target_year][doy_index].values, density=True, bins=np.arange(min(T_range), max(T_range) + 1.0, 1.0) + 0.25)
    
    # -- Plot probability distributions of pseudo-observations --
    ax1.plot(T_range, PDF_mm_dict["target"], color="royalblue", zorder=7, linewidth=2.5)

    ax2.plot(T_range, PDF_mm_dict["preind"], color="#1b9e77",label=f"\'{preind_year}\'", zorder=9, linewidth=2.5)
    ax2.plot(T_range, PDF_mm_dict["target"], color="royalblue",label=f"\'{target_year}\'", zorder=8, linewidth=2.5)
    
    if "future" in PDF_mm_dict:
        ax2.plot(T_range, PDF_mm_dict["future"], color="#e7298a",label=f"\'{future_year}\'", zorder=9, linewidth=2.5)

    # Plot histogram of pseudo-observations to the left-hand side subfigure
    ax1.bar(bin_edges[:-1]+0.25,hist, width=1.0, edgecolor="k", facecolor="skyblue", zorder=2, label=f"Pseudo-observations \'{target_year}\'", alpha=0.7)

    # Add a vertical line that marks the daily mean temperature of the target day
    ax2.axvline(x=target_value, zorder=10, color="k")
    ax2.annotate(f"${target_value:.1f}\\, \\degree$C", xy=(target_value-0.1, ax2.get_ylim()[1]), xycoords="data", rotation=90, ha="right", va="top")

    # -- Fill the area below the PDF and x-axis for temperatures warmer or colder than the target temperature --
    
    # Warmer than the target temperature   
    if pwarm:
        ax2.fill_between(T_range, PDF_mm_dict["preind"], where = T_range >= target_value, color="skyblue", zorder=5, alpha=0.7)
        ax2.fill_between(T_range, PDF_mm_dict["target"], where = T_range >= target_value, color="bisque", zorder=4, alpha=0.7)
        
        if "future" in PDF_mm_dict:
            ax2.fill_between(T_range, PDF_mm_dict["future"], where = T_range >= target_value, color="coral", zorder=3, alpha=0.7)
        
        probtext = rf"Probability\ of\ T\ \geq\ {target_value:.1f} °C"
        returntext = rf"Return\ period\ of\ T\ \geq\ {target_value:.1f} °C"
    
    # Colder than the target temperature
    else:
        ax2.fill_between(T_range, PDF_mm_dict["preind"], where = T_range <= target_value, color='lightblue', zorder=3, alpha=0.7)
        ax2.fill_between(T_range, PDF_mm_dict["target"], where = T_range <= target_value, color='coral', zorder=4, alpha=0.7)
        
        if "future" in PDF_mm_dict:
            ax2.fill_between(T_range, PDF_mm_dict["target"], where = T_range <= target_value, color='lightgrey', zorder=5, alpha=0.7)
        
        probtext = rf"Probability\ of\ T\ \leq\ {target_value:.1f} °C"
        returntext = rf"Return\ period\ of\ T\ \leq\ {target_value:.1f} °C"

    # Set axis tick-marks, labels and ranges and adjust the legend
    for ax in axlist:
        ax.set_xticks(np.arange(min(T_range), max(T_range), 4))
        ax.set_xlim(np.floor(np.nanmin(pseudo_obs_mm_target.loc[y1base:target_year][doy_index].values.squeeze()))-10,
                    np.ceil(np.nanmax((pseudo_obs_mm_target.loc[y1base:target_year][doy_index].values.squeeze())))+10)
        ax.set_xlabel(r"Temperature [$^{ \circ}$C]")
        ax.grid(True, zorder=1)
        ax.set_ylim(0, None)
    ax1.set_ylabel(r"Relative frequency / probability density [1/$^{\circ}$C]")
    ax1.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(0.92, 1.1))
    ax2.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.1))

    # Intensity change
    #dI_text = "Change\\ in\\ intensity"

    # -- Get text variables for printing --
    probabilities_preind = format_prob(probabilities, "preind")
    probabilities_target = format_prob(probabilities, "target")

    recurrence_preind = format_recurrence(probabilities, "preind")
    recurrence_target = format_recurrence(probabilities, "target")

    if intensity_intervals is not None:

        temps_preind = (
            f"{percentile_temps['t_preind']:.1f} °C "
            f"({intensity_intervals['t_preind_lower']:.1f}-"
            f"{intensity_intervals['t_preind_upper']:.1f}) °C")
    else:
        temps_preind = (f"{percentile_temps['t_preind']:.1f} °C ")


    if "obs" in probabilities:
        probabilities_obs = format_prob(probabilities, "obs")
        recurrence_obs = format_recurrence(probabilities, "obs")
        temps_obs =  rf"{y1base}-{target_year}: {percentile_temps['t_y1base']:.1f} °C"
    else:
        probabilities_obs = None
        recurrence_obs = None
        temps_obs = None

    if "future" in probabilities:
        probabilities_future = format_prob(probabilities, "future")
        recurrence_future = format_recurrence(probabilities, "future")
        
        if intensity_intervals is not None:
            temps_future = (
                f"{percentile_temps['t_future']:.1f} °C "
                f"({intensity_intervals['t_future_lower']:.1f}-"
                f"{intensity_intervals['t_future_upper']:.1f}) °C")
        else:
            temps_future = (f"{percentile_temps['t_future']:.1f} °C ")

    else:
        probabilities_future = None
        recurrence_future = None
        temps_future = None

    # -- Save the printed text in a list --
    text = []

    # Add place and time period to the text
    name = station_meta["name"].replace("_", " ").title()
    name_tex = name.replace(" ", r"\ ")
    text.append(rf"$\bf{{{name_tex}}}$")
    text.append(f"({station_meta["latitude"]:.2f} °N, {station_meta["longitude"]:.2f} °E)")
    text.append(time_period)
    text.append("")
    
    # -- Add probabilities to the text --
    text.append(rf"$\bf{{{probtext}}}$")
    
    if probabilities_obs is not None:
        text.append(f"{y1base}-{target_year}: {probabilities_obs}")

    text.append("")
    text.append(f"'{preind_year}': {probabilities_preind}")
    text.append(f"'{target_year}': {probabilities_target}")

    if probabilities_future is not None:
        text.append(f"'{future_year}': {probabilities_future}")

    text.append("")
    
    # -- Add return periods to the text --
    text.append(rf"$\bf{{{returntext}}}$")

    if recurrence_obs is not None:
        text.append(f"{y1base}-{target_year}: {recurrence_obs}")

    text.append("")
    text.append(f"'{preind_year}': {recurrence_preind}")
    text.append(f"'{target_year}': {recurrence_target}")
    if recurrence_future is not None:
        text.append(f"'{future_year}': {recurrence_future}")

    text.append("")

    # -- Add intensities to the text --
    text.append(rf"$\mathbf{{{"Change\\ in\\ intensity"}}}$")
    
    if temps_obs is not None:
        text.append(temps_obs)
    text.append("")
    text.append(f"'{preind_year}': {temps_preind}")
    text.append(rf"'{target_year}': {target_value:.1f} °C")

    if temps_future is not None:
        text.append(f"'{future_year}': {temps_future}")
        text.append("")
        text.append(f"Scenario for future climate:\n{ssp}")

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    # place a text box in upper left in axes coords
    ax2.text(1.04, 0.99, "\n".join(text), transform=ax2.transAxes, fontsize=13, verticalalignment="top", bbox=props)

    # Get statistical parameters of the present-day observations
    mean_val = np.mean(pseudo_obs_mm_target[doy_index])
    variance_val = np.var(pseudo_obs_mm_target[doy_index])
    skewness_val = skew(pseudo_obs_mm_target[doy_index],nan_policy='omit')
    kurtosis_val = kurtosis(pseudo_obs_mm_target[doy_index],nan_policy='omit')

    # Display statistical parameters in the left-hand side figure
    ax1.annotate(rf'$\mu$={mean_val:.2f}', (0.03, 0.87), xycoords="axes fraction")
    ax1.annotate(rf'$\sigma^{{\rm 2}}$={variance_val:.2f}', (0.03, 0.82), xycoords="axes fraction")
    ax1.annotate(rf'$\gamma$={skewness_val:.2f}', (0.03, 0.78), xycoords="axes fraction")
    ax1.annotate(rf'$\kappa$={kurtosis_val:.2f}', (0.03, 0.74), xycoords="axes fraction")

    ax1.annotate("a", (0.03, 0.95), xycoords="axes fraction", fontweight="bold", fontsize=18)
    ax2.annotate("b", (0.03, 0.95), xycoords="axes fraction", fontweight="bold", fontsize=18)

    # Save the figure
    figure_name = f"distribution_plot_{obs_source}_{station_meta["station_id"]}_{fig_name_date_string}.png"
    plt.savefig(os.path.join(path2figures, figure_name), dpi=300, bbox_inches="tight")
    #plt.show()



def plot_model_statistics(
    path2figures:str,
    obs_source:str,
    station_meta:dict,
    doy_index:int,
    n_days:int,
    model_names:list,
    n_boots:int,
    index:int,
    target_value:float,
    pwarm:bool,
    bootstrapping_dict:dict[str, np.ndarray],
    probabilities:dict,
    percentile_temps:dict,
    dI_intervals:dict)-> None:

    """
    Plots probability distributions for the pre-industrial, present-day and future climates and prints a summary of the attribution statistics.

    Args:
        path2figures: str 
            Directory path to save the resulting plot.
        obs_source: str
            Source of observations (FMI, SMHI, FROST)
        station_meta: dict
            Station metadata
        doy_index: int
            day_of_year index
        n_days: int
            Number of days in the period
        model_names: list
            List of climate models
        n_boots: int
            Number of bootstrap iterations
        target_value: float
            Target observation
        pwarm: bool
            Probabilty for warmer (True)/colder (False) temperatures
        bootstrapping_dict:
            Contains 3D numby arrays of bootstrapped CDFs
        probabilities: dict
            Contains probability estimates for:
                - Pre-industrial and present-day climates
                - Future and observed climates (Optional)
                - Confidence intervals (Optional)
        percentile_temps: dict
            Contains temperatures corresponding to a given percentile in:
                 - Observed climate over the baseline period (usually 1901-preceding year)
                 - Pre-industrial year
                 - Current climate
                 - Future climate (Optional)
        dI_intervals (dict):
            Contains the lower and uppoer bounds for temperature intensities in:
                - Pre-industrial climate
                - Future climate (Optional) 
    Returns:
        None: The plot is saved as a PNG-figure.
    """

    # Number of models
    n_models = len(model_names)

    # Station name
    station_id = station_meta["station_id"]

    # Get bootstrap results to arrays
    preind_array = bootstrapping_dict["preind"]
    target_array = bootstrapping_dict["target"]

    # Convert date to day of year and date string
    _, fig_date_name_string = get_date_strings(doy_index, n_days)
    
    # Set properties for the box plots, including the line style, width, and color for the median
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    medianprops_mm = dict(linestyle=None, linewidth=0,) # Properties for the mean model
    boxprops=dict(facecolor='red', color='k') # Box properties (red with black edges)

    # Create a figure with two subplots (ax1 and ax2) to display both the probability ratio and intensity change
    fig, axlist=plt.subplots(nrows=1, ncols=2,figsize=(10,12), sharey=True)
    plt.subplots_adjust(wspace=0.7) # Adjust space between the two subplots

    ax=axlist[0] # First subplot for probability ratio

    # Compute the probability ratios for each model (probability of target period exceeding pre-industrial)
    if pwarm:
        PRratios = np.transpose((1-target_array[index,:,:]) / (1-preind_array[index,:,:]))
    else:
        PRratios = np.transpose(preind_array[index,:,:] / target_array[index,:,:])

    # Filter out any NaN values from the probability ratios array
    mask = ~np.isnan(PRratios)
    filtered_data = [d[m] for d, m in zip(PRratios.T, mask.T)]

    # Get the probability ratio for the multi-model mean (MMM)
    pr_ratio_bp = np.array(probabilities["pr_ratio"])
    pr_ratio_bp = pr_ratio_bp.reshape(pr_ratio_bp.shape + (1,))

    # Plot the probability ratio box plot for each model
    a = ax.boxplot(filtered_data, 
                   positions=np.arange(0,n_models),
                   labels=model_names,
                   vert=False,
                   showfliers=False,
                   medianprops=medianprops,
                   whis=(5,95),
                   widths=0.7)

    # Add a box plot for the multi-model mean (MMM) using the combined filtered data
    a2 = ax.boxplot([np.concatenate(filtered_data)],
                    positions=[n_models],
                    labels=['MMM'],
                    vert=False,
                    showfliers=False,  
                    patch_artist=True,
                    medianprops=medianprops_mm,
                    whis=(5,95),
                    widths=0.7,
                    boxprops=boxprops)

    # Add a box plot for the probability ratio for the multi-model mean (MMM)
    a2b = ax.boxplot(pr_ratio_bp,
                    positions=[n_models],
                    labels=[''],
                    vert=False,
                    showfliers=False,
                    medianprops=medianprops,
                    whis=(5,95),
                    widths=0.7)

    # Adjust axes
    ax.yaxis.tick_right()  # Move the y-axis ticks to the right
    ax.set_xlim(0.1, np.max(filtered_data))
    ax.set_xlabel("Probability ratio")
    ax.set_xscale('log')
    ax.invert_yaxis()

    # Second subplot for intensity change
    ax=axlist[1]

     # Compute the change in intensity (temperature difference) for each bootstrap sample
    deltaI = dI_intervals["intensity_diff"][::n_boots+1].values
    deltaI = np.transpose(deltaI.reshape(deltaI.shape + (1,)))

    # Calculate the intensity change for the multi-model mean (MMM) for the target value
    deltaI_mm = np.round(target_value - percentile_temps["t_preind"],1)
    deltaI_mm = deltaI_mm.reshape(deltaI_mm.shape + (1,))

    # Plot the change in intensity box plot for each model
    b = ax.boxplot(deltaI,
                   positions=np.arange(0,n_models),
                   labels=model_names,
                   vert=False,
                   showfliers=False,
                   medianprops=medianprops,
                   widths=0.7)
    
    # Add a box plot for the multi-model mean (MMM) intensity change
    b2 = ax.boxplot(deltaI_mm,
                    positions=[n_models],
                    labels=[''],
                    vert=False,
                    showfliers=False,
                    medianprops=medianprops,
                    widths=0.7)
    
    # Add a box plot for the combined intensity change data for the MMM
    b3 = ax.boxplot([np.concatenate(deltaI)],
                    positions=[n_models],
                    labels=['MMM'],
                    vert=False,
                    showfliers=False,  
                    patch_artist=True,
                    medianprops=medianprops_mm,
                    whis=(5,95),
                    widths=0.7,
                    boxprops=boxprops)

    # Adjust axes
    ax.set_xlim(0, np.max(deltaI)+1)
    ax.set_xlabel(r'Change in intensity [$^{\circ}$C]')

    # Save the figure
    figure_name = f"model_statistics_{obs_source}_{station_id}_{fig_date_name_string}.png"
    plt.savefig(os.path.join(path2figures, figure_name), dpi=300, bbox_inches="tight")
    #plt.show()


def plot_rank_histogram(path2figures:str, place:str, observations_df:pd.DataFrame, quantile_regressed_obs_df:pd.DataFrame)-> None:

    """
    Plots and saves a normalized rank histogram to evaluate the distribution of observed daily temperatures 
    relative to quantile-regressed predictions.

    Args:
        path2figures (str): Directory to save the rank histogram figure.
        place (str): Name of the location or station (used in the filename and title).
        observations_df (pd.DataFrame): Observed daily temperatures (rows: years, columns: days of the year).
        quantile_regressed_obs_df (pd.DataFrame): Quantile-regressed predictions (rows: quantiles, columns: days of the year).

    Returns:
        None: Saves the histogram as a PNG file and displays the plot.

    """        
    # Get the quantile values and observations
    quantiles = quantile_regressed_obs_df.index.values  # Quantile levels
    n_quantiles = len(quantiles)  # Number of quantiles
    observations = observations_df.values  # Observations (day_of_year x year matrix)
    quantile_values = quantile_regressed_obs_df.values  # Quantile estimates (quantile x day_of_year)

    # Initialize an empty list to store ranks
    ranks = []
    
    # Iterate over each day of the year
    for day_idx in range(observations.shape[1]):
        day_observations = observations[:, day_idx]  # Observations for the current day (all years)
        day_quantiles = quantile_values[:, day_idx]  # Quantile predictions for the current day

        # Remove NaN values from both observations and quantiles
        valid_mask = ~np.isnan(day_observations)
        valid_observations = day_observations[valid_mask]

        if valid_observations.size > 0:  # Skip if no valid observations
            day_ranks = np.sum(valid_observations[:, None] > day_quantiles, axis=1)  # Count quantiles exceeded
            ranks.extend(day_ranks)  # Append ranks to the overall list

    # Normalize the histogram
    rank_counts, bin_edges = np.histogram(ranks, bins=np.arange(n_quantiles + 2) - 0.5)
    normalized_counts = rank_counts / np.sum(rank_counts)  # Normalize by total count

    # Plot the normalized histogram
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(n_quantiles + 1), normalized_counts, width=1, edgecolor="black", alpha=0.7, color="skyblue")
    plt.xticks(np.arange(0,101,5), labels=np.arange(0,101,5))
    plt.xlabel("Percentile rank [%]")
    plt.ylabel("Frequency")
    plt.title(f"Normalized Rank Histogram of Observations")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    figure_name = f"rank_histogram_{place}_obs.png"
    plt.savefig(os.path.join(path2figures, figure_name), dpi=300, bbox_inches="tight")
    #plt.show()

###########################################################################################################################################################################################



# Import needed libraries and tools
import numpy as np
import xarray as xr
import os
from scipy.stats import linregress
from scipy.optimize import curve_fit
import calendar
import argparse
from datetime import datetime, timedelta
import cftime

#-------------------Calendar functions----------------------#
def get_dayofyear_index(day_str, year_length):
    """
    Convert "MM-DD" string to day-of-year index (1..year_length)

    Parameters
    ----------
    day_str : str
        Day in format "MM-DD", e.g., "01-01"
    year_length : int
        Length of the calendar year (360 or 365)

    Returns
    -------
    int
        Day-of-year index (1..year_length)
    """
    month, day = map(int, day_str.split("-"))

    if year_length == 360:
        # 360-day calendar: 12 months x 30 days
        return (month - 1) * 30 + day
    elif year_length == 365:
        # Standard Gregorian year: sum days in preceding months + current day
        # Use a fixed non-leap year (e.g., 2001) to get day-of-year
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return sum(days_in_month[:month - 1]) + day
    else:
        raise ValueError(f"Unsupported year length: {year_length}")

def is_360_day_calendar(ds):
    time_encoding = ds["time"].encoding
    return "calendar" in time_encoding and time_encoding["calendar"] == "360_day"


#-------------------------------------Raw regression coefficients------------------------------#

# A function for computing the regression coefficients B and D for all coordinates on a day or n-day mean
def get_regression_coefficients(ds_var, ds_g11, day_str, clim_var, n_days):

    # Extract start and end years and the index of the year 2000
    start_year = str(ds_g11.time.dt.year.min().values)
    end_year = str(ds_g11.time.dt.year.max().values)
    years = np.arange(int(start_year), int(end_year) + 1)
    year_2000_idx = np.where(years == 2000)[0][0]

    # 11-year global mean time series
    g11 = ds_g11["tas"].values.squeeze()

     # Check for calendar type and select data accordingly
    if is_360_day_calendar(ds_var):
        year_length = 360
        #daily_data = daily_data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-30"))
    else:
        year_length = 365
        #daily_data = daily_data.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

    # Compute target day index
    target_day = get_dayofyear_index(day_str, year_length)
    #target_day = pd.to_datetime(f"2001-{day_str}").dayofyear  # 2001 is arbitrary non-leap year

    # n-day cyclic (extends over year) running window
    half = n_days // 2
    window = [(target_day + offset - 1) % year_length + 1 for offset in range(-half, half + 1)]

    # Select data for all years in this window
    daily_data = ds_var.sel(time=ds_var.time.dt.dayofyear.isin(window))

    # Average across the n-day window for each year
    daily_data = daily_data.groupby("time.year").mean("time")

    # Extract data for the specific day
    #daily_data = ds_var.sel(time=ds_var.time.dt.strftime("%m-%d") == day_str)

    # Convert Kelvin to Celsius
    daily_array = daily_data[clim_var].values - 273.15

     # Initialize arrays for regression coefficients
    shape = daily_array.shape[1:] # Get the shape for latitude/longitude
    A, B, C, D = np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape)

    for lat in range(shape[0]):
        for lon in range(shape[1]):
            slope, intercept, _, _, _ = linregress(g11, daily_array[:, lat, lon])
            A[lat, lon] = intercept
            B[lat, lon] = slope

            regressed = intercept + slope * g11[:, None]
            residuals = daily_array[:, lat, lon] - regressed.squeeze()
            tas_var = residuals ** 2

            slope_var, intercept_var, _, _, _ = linregress(g11, tas_var)
            C[lat, lon] = intercept_var
            D[lat, lon] = slope_var

    var_regression_fit = C + D * g11[:, None, None]
    var2000 = var_regression_fit[year_2000_idx, :, :]
    D_final = D / np.squeeze(var2000)

    return B, D_final

# A function for computing regression coefficients for the entire year
def get_regression_coefficients_year(ds_ts, ds_g11, clim_var, n_days):

    # Initialize dictionaries to store coefficients for each day of the year
    B_dict = {}
    D_dict = {}

    # Loop through each month (1 to 12)
    for month in range(1, 13):

        # Determine the number of days in the current month
        num_days = calendar.monthrange(2001, month)[1]  # A random non-leap year: 2001

        # Loop through each day in the current month
        for day in range(1, num_days + 1):

            # Format the date as "MM-DD" for the current month and day
            day_str = f"{month:02d}-{day:02d}"
            print(f"Processing {day_str}")

            # Compute regression coefficients A and B for this specific day
            B, D = get_regression_coefficients(ds_ts, ds_g11, day_str, clim_var, n_days)

            # Store the coefficients in the dictionaries with day_str as the key
            B_dict[day_str] = B
            D_dict[day_str] = D

    return B_dict, D_dict

# Compute regression coefficients for a 360-day year
def get_regression_coefficients_360days(ds_tas, ds_g11, clim_var, n_days):

    # Initialize dictionaries to store coefficients for each day of the year
    B_dict = {}
    D_dict = {}

    # Loop through each month (1 to 12)
    for month in range(1, 13):

        # For a 360-day calendar, we assume each month has 30 days
        num_days = 30  # All months have 30 days in a 360-day calendar

        # Loop through each day in the current month
        for day in range(1, num_days + 1):

            # Format the date as "MM-DD" for the current month and day
            day_str = f"{month:02d}-{day:02d}"
            print(f"Processing {day_str}")

            # Compute regression coefficients A and B for this specific day
            B, D = get_regression_coefficients(ds_tas, ds_g11, day_str, clim_var, n_days)

            # Store the coefficients in the dictionaries with day_str as the key
            B_dict[day_str] = B
            D_dict[day_str] = D

    return B_dict, D_dict

#----------- Fit a Fourier-series and extend 360-day coeffs to 365 days  ------------#

# General Fourier series factory
def make_fourier_series(period):
    def fourier_func(x, *params):
        a0 = params[0]  # constant term
        result = a0
        num_harmonics = (len(params) - 1) // 2
        for i in range(1, num_harmonics + 1):
            result += params[2*i-1] * np.sin(2 * np.pi * i * x / period) \
                    + params[2*i]   * np.cos(2 * np.pi * i * x / period)
        return result
    return fourier_func

# Define a Fourier function for 365-day year
def fourier_series_365(x, *params):
    """
    Fourier series function up to n harmonics.
    The first term is the offset (mean value), then sine/cosine terms follow.
    """
    a0 = params[0]  # The constant term
    result = a0
    num_harmonics = (len(params) - 1) // 2  # Calculate number of harmonics

    # Loop through harmonics, using the correct range
    for i in range(1, num_harmonics + 1):
        result += params[2*i-1] * np.sin(2 * np.pi * i * x / 365) + params[2*i] * np.cos(2 * np.pi * i * x / 365)

    return result

# Define a Fourier function for 360-day year
def fourier_series_360(x, *params):
    """
    Fourier series function up to n harmonics.
    The first term is the offset (mean value), then sine/cosine terms follow.
    """
    a0 = params[0]  # The constant term
    result = a0
    num_harmonics = (len(params) - 1) // 2  # Calculate number of harmonics

    # Loop through harmonics, using the correct range
    for i in range(1, num_harmonics + 1):
        result += params[2*i-1] * np.sin(2 * np.pi * i * x / 360) + params[2*i] * np.cos(2 * np.pi * i * x / 360)

    return result

# A script for fitting Fourier series to the seasonal variation of B and D at all lats and lons
def fit_fourier_series_to_all_coeffs(B, D, num_harmonics):

    """
    Fit a Fourier series to seasonal variation of regression coefficients B and D.

    Parameters
    ----------
    B, D : dictionaries with "daysofyear as keys and arrays as values with shape (lats, lons)
        Arrays with shape (num_days, num_lats, num_lons).
    num_harmonics : int
        Number of Fourier harmonics to include.

    Returns
    -------
    B_fitted, D_fitted : np.ndarray
        Fourier-fitted coefficients with the same shape as B, D.
    """

    if isinstance(B, dict) and isinstance(D, dict):
        daysofyear = sorted(B.keys())
        B = np.stack([B[day] for day in daysofyear], axis=0)
        D = np.stack([D[day] for day in daysofyear], axis=0)

    num_days, num_lats, num_lons = B.shape

    # Choose Fourier function based on calendar length
    if num_days == 365:
        fourier_func = make_fourier_series(365)
    elif num_days == 360:
        fourier_func = make_fourier_series(360)
    else:
        raise ValueError(f"Unexpected num_days={num_days}. Expected 360 or 365.")

    # Initialize a numpy array for the results
    B_fitted_coeffs = np.zeros(B.shape)
    D_fitted_coeffs = np.zeros(D.shape)

    # Day of year
    dayofyear = np.arange(1,num_days+1,1)

    # Loop over all latitudes and longitudes and fit Fourier series for the seasonal variation of regression coefficients B and D
    for lat in range(num_lats):
        for lon in range(num_lons):

            # Extract regression coefficients for all the days of the year
            B_year = B[:,lat,lon]
            D_year = D[:,lat,lon]

            # Initial guess for Fourier components
            initial_guess_B = [np.mean(B[:,lat,lon])] + [1] * (2 * num_harmonics)
            initial_guess_D = [np.mean(D[:,lat,lon])] + [1] * (2 * num_harmonics)

            if num_days == 365:

                # Fit the Fourier series to time series of B and D
                params_B, _ = curve_fit(fourier_series_365, dayofyear, B_year, p0=initial_guess_B)
                params_D, _ = curve_fit(fourier_series_365, dayofyear, D_year, p0=initial_guess_D)

                # Save the fitted coefficients to numpy arrays
                B_fitted_coeffs[:,lat,lon] = fourier_series_365(dayofyear, *params_B)
                D_fitted_coeffs[:,lat,lon] = fourier_series_365(dayofyear, *params_D)

            if num_days == 360:

                # Fit the Fourier series to time series of B and D
                params_B, _ = curve_fit(fourier_series_360, dayofyear, B_year, p0=initial_guess_B)
                params_D, _ = curve_fit(fourier_series_360, dayofyear, D_year, p0=initial_guess_D)

                # Save the fitted coefficients to numpy arrays
                B_fitted_coeffs[:,lat,lon] = fourier_series_360(dayofyear, *params_B)
                D_fitted_coeffs[:,lat,lon] = fourier_series_360(dayofyear, *params_D)

    return B_fitted_coeffs, D_fitted_coeffs

# Function to fit Fourier coefficients to 365-day calendar using parameters from 360-day calendar
def extend_fourier_to_365_from_360(B_360, D_360, num_harmonics=6):
    num_days_360, num_lats, num_lons = B_360.shape

    # Initialize arrays for the 365-day fitted values
    B_fitted_365 = np.zeros((365, num_lats, num_lons))
    D_fitted_365 = np.zeros((365, num_lats, num_lons))

    # Day of year for 360 days and 365 days
    dayofyear_360 = np.arange(1, 361)
    dayofyear_365 = np.arange(1, 366)

    # Loop over each latitude and longitude
    for lat in range(num_lats):
        for lon in range(num_lons):
            # Get the 360-day Fourier-fitted data for each location
            B_series_360 = B_360[:, lat, lon]
            D_series_360 = D_360[:, lat, lon]

            # Fit Fourier series for B and D on 360-day calendar to get coefficients
            initial_guess_B = [np.mean(B_series_360)] + [0] * (2 * num_harmonics)
            initial_guess_D = [np.mean(D_series_360)] + [0] * (2 * num_harmonics)

            params_B, _ = curve_fit(fourier_series_360, dayofyear_360, B_series_360, p0=initial_guess_B)
            params_D, _ = curve_fit(fourier_series_360, dayofyear_360, D_series_360, p0=initial_guess_D)

            # Evaluate the adjusted Fourier series for 365 days using the fitted parameters
            B_fitted_365[:, lat, lon] = fourier_series_365(dayofyear_365, *params_B)
            D_fitted_365[:, lat, lon] = fourier_series_365(dayofyear_365, *params_D)

    return B_fitted_365, D_fitted_365

#------------------------------- Save the regression coefficients --------------------------------#

# A function for saving the regression coefficients to a netcdf file
def save_raw_coeffs_to_netcdf(ds_var, output_directory, model_name, lat, lon, B_dict, D_dict, clim_var, ssp, n_days):

    """
    Save regression coefficients to a NetCDF file, handling both 360-day and 365-day calendars.

    Parameters
    ----------
    ds_var : xarray.Dataset
        Original dataset (used to detect calendar type).
    output_directory : str
        Directory to save the NetCDF file.
    model_name : str
        Name of the climate model.
    lat : array-like
        Latitude coordinates.
    lon : array-like
        Longitude coordinates.
    B_dict, D_dict : dict
        Dictionaries with regression coefficients for each day ("MM-DD").
    clim_var : str
        Name of the climate variable.
    ssp : str
        Scenario name.
    n_days : int
        n-day running mean used to compute the coefficients.
    """

    # Detect calendar type
    if is_360_day_calendar(ds_var):
        calendar_type = "360_day"
        start_date = cftime.Datetime360Day(2001, 1, 1)
        time_coords = xr.cftime_range(start=start_date, periods=360, freq="D", calendar="360_day")
    else:
        calendar_type = "365_day"
        start_date = cftime.DatetimeNoLeap(2001, 1, 1)
        time_coords = xr.cftime_range(start=start_date, periods=365, freq="D", calendar="noleap")

     # Convert the date strings ("MM-DD") to datetime objects for the time coordinate
    #start_date = datetime(2001,1,1)
    #time_coords = [start_date + timedelta(days=i) for i in range(len(B_dict))]

    # Output name
    output_name = f"{model_name}_{clim_var}_{ssp}_regr_coeffs_{n_days}days.nc"
    output_path = os.path.join(output_directory,output_name)

    # Convert A_dict and B_dict to xarray.DataArray objects
    B_array = xr.DataArray(
        data=np.array([B_dict[day.strftime("%m-%d")] for day in time_coords]),  # Convert list of 2D arrays to a 3D array
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lat,
            "lon": lon
        },
        attrs={"source": model_name,
            "n_day_running_mean": n_days,
            "calendar": calendar_type,
            "description": "Regression coefficient B"}
    )

    D_array = xr.DataArray(
        data=np.array([D_dict[day.strftime("%m-%d")] for day in time_coords]),  # Convert list of 2D arrays to a 3D array
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lat,
            "lon": lon
        },
        attrs={"source": model_name,
            "n_day_running_mean": n_days,
             "calendar": calendar_type,
             "description": "Regression coefficient D"}
    )

    # Set coordinate metadata
    for array in [B_array, D_array]:
        array.coords["lat"].attrs["units"] = "degrees_north"
        array.coords["lat"].attrs["standard_name"] = "latitude"
        array.coords["lon"].attrs["units"] = "degrees_east"
        array.coords["lon"].attrs["standard_name"] = "longitude"

    # Create an xarray.Dataset to hold both B and D arrays
    ds_coeffs = xr.Dataset(
        {
            f"B_{model_name}": B_array,
            f"D_{model_name}": D_array,
        }
    )
    ds_coeffs.attrs["title"] = "Regression coefficients B and D"
    ds_coeffs.attrs["source"] = model_name
    ds_coeffs.attrs["n_day_running_mean"] = n_days
    ds_coeffs.attrs["calendar"] = calendar_type
    ds_coeffs.attrs["description"] = "Raw regression coefficients"

    # Save the dataset to a NetCDF file
    ds_coeffs.to_netcdf(output_path)

def save_fourier_coeffs_to_netcdf(output_directory, model_name, lats, lons, B_fourier, D_fourier, n_days):

    # Use a 360-day calendar for UKESM1 and HadGEM3 models:
    #if model_name in ["UKESM1.0-LL", "UKESM1-0-LL", "HadGEM3-GC31-LL", "KACE1.0-G", "KACE-1-0-G" ]:

        # 360-day calendar for UKESM1.0-LL, HadGEM3-CG31-LL, KACE-1-0-G
    #    start_date = cftime.Datetime360Day(2001, 1, 1)
    #    time_coords = xr.cftime_range(start=start_date, periods=360, freq="D", calendar="360_day")
    #    calendar_type = "360_day"
    #else:

        # A 365-day calendar for other models
    #    start_date = cftime.DatetimeNoLeap(2001, 1, 1)
    #    time_coords = xr.cftime_range(start=start_date, periods=365, freq="D", calendar="noleap")
    #    calendar_type = "365_day"
    
    # A 365-day calendar
    start_date = cftime.DatetimeNoLeap(2001, 1, 1)
    time_coords = xr.cftime_range(start=start_date, periods=365, freq="D", calendar="noleap")

    # Check if the ouput directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Output name
    output_name = f"{model_name}_fourier_fitted_coeffs_{n_days}days.nc"
    output_path = os.path.join(output_directory,output_name)

    # Regression coefficient B
    B_array = xr.DataArray(
        data=B_fourier,
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lats,
            "lon": lons
        },
        attrs={"source": model_name,
             "n_day_running_mean": n_days,
             "calendar": "365_day",
             "description": "Fourier-fitted regression coefficient B"
        }
    )

    # Regression coefficient D
    D_array = xr.DataArray(
        data=D_fourier,
        dims=["time", "lat", "lon"],
        coords={
            "time": time_coords,
            "lat": lats,
            "lon": lons
        },
        attrs={"source": model_name,
             "n_day_running_mean": n_days,
             "calendar": "365_day",
             "description": "Fourier-fitted regression coefficient D"
        }
    )

    # Set coordinate metadata
    for array in [B_array, D_array]:
        array.coords["lat"].attrs["units"] = "degrees_north"
        array.coords["lat"].attrs["standard_name"] = "latitude"
        array.coords["lon"].attrs["units"] = "degrees_east"
        array.coords["lon"].attrs["standard_name"] = "longitude"

    # Combine into a Dataset
    ds_fourier_coeffs = xr.Dataset(
        {
            f"B_{model_name}": B_array,
            f"D_{model_name}": D_array,
        })
    
    ds_fourier_coeffs.attrs["title"] = "Fourier-fitted regression coefficients B and D"
    ds_fourier_coeffs.attrs["source"] = model_name
    ds_fourier_coeffs.attrs["n_day_running_mean"] = n_days
    ds_fourier_coeffs.attrs["calendar"] = "365_day"
    ds_fourier_coeffs.attrs["description"] = "Fourier-fitted coefficients using 6 harmonics for seasonal variation of n-day running means"

    # Save the dataset to a NetCDF file
    ds_fourier_coeffs.to_netcdf(output_path)


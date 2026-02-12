# Import needed libraries and tools
import xarray as xr
import os
from scipy.stats import linregress
import argparse
import re
import regression_tools

# ---------------------- Argument parser ---------------------- #

parser = argparse.ArgumentParser(description="Compute and save regression coefficients B and D from climate model data")
parser.add_argument("--ts_file", type=str, required=True, help="Filename for the time series data")
parser.add_argument("--g11_file", type=str, required=True, help="Filename for the g11 data")
parser.add_argument("--n_days", type=int, default=1, help="Number of days for running mean (default=1).")
args = parser.parse_args()

# Extract arguments (time-series file, 11-year running mean of global temperature file and the number of days)
ts_file = args.ts_file
g11_file = args.g11_file
n_days = args.n_days

# ---------------------- Extract clim_var and ssp ---------------------- #

basename = os.path.basename(ts_file)
print(basename)
match = re.match(r".*_(tasmax|tasmin|tas)_(ssp\d{3}).*\.nc", basename)
if match:
    clim_var, ssp = match.groups()
    print(f"clim_var: {clim_var}, ssp: {ssp}")
else:
    raise ValueError(f"Could not extract clim_var and ssp from file {ts_file}")

# ---------------------- Define paths ---------------------- #

# Directories
path2tas_dir = f"/scratch/project_2014701/input_data_regression/time_series/{clim_var}/{ssp}"
path2g11_dir = f"/scratch/project_2014701/input_data_regression/g11/{ssp}"
path2raw_results = f"/scratch/project_2014701/raw_BD/{clim_var}/{ssp}/ndays_{n_days}"
path2fourier_results = f"/scratch/project_2014701/fourier_BD/{clim_var}/{ssp}/ndays_{n_days}"

# Ensure output directories exists
os.makedirs(path2raw_results, exist_ok=True)
os.makedirs(path2fourier_results, exist_ok=True)

# Paths to filenames
path2ts = os.path.join(path2tas_dir,ts_file)
path2g11 = os.path.join(path2g11_dir,g11_file)

# ---------------------- Open datasets ---------------------- #

ds_ts = xr.open_dataset(path2ts,engine="netcdf4")
ds_g11 = xr.open_dataset(path2g11,engine="netcdf4")

# Extract model name and coordinates
model_name = ds_ts.attrs["source_id"].split()[0]
lat = ds_ts["lat"].values
lon = ds_ts["lon"].values

# ---------------------- Logging ---------------------- #

print(f"Processing model: {model_name}")
print(f"Climate variable: {clim_var}, SSP: {ssp}")

calendar_360 = regression_tools.is_360_day_calendar(ds_ts)
year_length = 360 if calendar_360 else 365

print(f"Calendar type: {'360_day' if calendar_360 else '365_days'}, days in year: {year_length}")
print(f"Using {n_days}-day running mean")

# ---------------------- Compute regression coefficients ---------------------- #
if year_length == 365:

    try:
        B_fullyear, D_fullyear = regression_tools.get_regression_coefficients_year(ds_ts, ds_g11, clim_var, n_days)
        print("Regression coefficients computed!")
    except Exception as e:
        print(f"Error while computing regression coefficients: {e}")
else:
    try:
        B_fullyear, D_fullyear = regression_tools.get_regression_coefficients_360days(ds_ts, ds_g11, clim_var, n_days)
        print("Regression coefficients computed!")
    except Exception as e:
        print(f"Error while computing regression coefficients: {e}")

# ---------------------- Save raw B and D ---------------------- #
try:
    regression_tools.save_raw_coeffs_to_netcdf(ds_ts, path2raw_results, model_name, lat, lon, B_fullyear, D_fullyear, clim_var, ssp, n_days)
    print(f"Regression coefficients saved successfully to {path2raw_results}!")
except Exception as e:
    print(f"Error while saving raw B and D to a NetCDF-file: {e}")

# ----------- Fit a 6-component Fourier-Series to B and D ------------------ #
try:
    B_fourier, D_fourier = regression_tools.fit_fourier_series_to_all_coeffs(B_fullyear, D_fullyear, num_harmonics = 6)
    print("Fourier fit done!")
except Exception as e:
    print(f"Error while fitting a Fourier series to B and D: {e}")

# ----------------------- Extend 360 -> 365 days --------------- #
if calendar_360:
    try:
        B_fourier, D_fourier = regression_tools.extend_fourier_to_365_from_360(B_fourier, D_fourier, num_harmonics = 6)
        print("B and D extended!")
    except Exception as e:
        print(f"Error while extending a 360-day B and D to 365 days: {e}")

# -------------- Save Fourier-fitted B and D --------------------- #
try:
    regression_tools.save_fourier_coeffs_to_netcdf(path2fourier_results, model_name, lat, lon, B_fourier, D_fourier, n_days)
    print(f"Results saved to {path2fourier_results}")
except Exception as e:
    print(f"Error while saving Fourier-fitted B and D to a NetCDF-file: {e}")

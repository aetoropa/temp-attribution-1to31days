#!/bin/bash

# Climate variable (tas,tasmax,tasmin)
clim_var="tasmin"

# Define emission scenario
ssp="ssp585"

# Number of days
n_days="$1"

# Input directory
input_dir="/scratch/project_2014701/fourier_BD/${clim_var}/${ssp}/ndays_${n_days}"
echo $input_dir

# Output directory
output_dir="/scratch/project_2014701/regridded_BD/${clim_var}/${ssp}/ndays_${n_days}"

# Path to target grid
path2grid="/scratch/project_2014701/target_grid.txt"

# Ensure the output directory exists
mkdir -p "$output_dir"

# List of file paths to be operated
file_paths=("${input_dir}"/*coeffs_${n_days}days.nc)

# Loop through paths
for path in "${file_paths[@]}"; do
   
   # Extract filename
   filename=$(basename "$path")
   
   # Extract modelname from the filename
   model=${filename%%_fourier_fitted_coeffs_${n_days}days.nc}
  
   # Print script progress
   echo "Processing ${model}..." 

   # Temporary and final filenames
   temp_file="${output_dir}/${model}_temp.nc"
   final_file="${output_dir}/${model}_regridded_${n_days}days.nc"

   # Regrid fourier-fitted regression coefficients to 2.5 degree grid
   cdo remapbil,"$path2grid" "$path" "$temp_file"

   # Change the names of the regression coefficients so they're distinguishable
   cdo chname,"B_${model}",B,"D_${model}",D "$temp_file" "$final_file"
   #cdo chname,B,"B_${model}",D,"D_${model}" "$temp_file" "$final_file"

   # Track progress
   echo "Regression coefficients of model ${model} regridded."
done

# List of temp files
temp_files=("${output_dir}"/*temp.nc)

# List of post-processed files 
processed_files=("${output_dir}"/*_regridded_${n_days}days.nc)

# The number of post-processed files
num_files=${#processed_files[@]}

# Names of final files
multi_model_mean_file="${output_dir}/a_${clim_var}_multi-modelmean_${num_files}mod_${ssp}_${n_days}days.nc"
single_model_file="${output_dir}/a_${clim_var}_mean_var_${num_files}mod_${ssp}_${n_days}days.nc"

# Calculate model mean coefficients
cdo ensmean "${processed_files[@]}" "$multi_model_mean_file"

# Combine the single model coefficients into a single file
cdo merge "${temp_files[@]}" "$single_model_file"

# Remove temporary files
rm "${temp_files}"

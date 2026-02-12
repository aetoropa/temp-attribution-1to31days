#!/bin/bash

# ==========================
# Arguments
# ==========================
hist_file="$1"
proj_file="$2"
n_days="${3:-1}"   # default = 1 if not given

if [[ -z "$hist_file" || -z "$proj_file" ]]; then
    echo "Usage: $0 <historical_file> <ssp_file>"
    exit 1
fi

# ==========================
# Parse filename information
# ==========================
base_hist=$(basename "$hist_file")
base_proj=$(basename "$proj_file")

IFS='_' read -r var freq model scenario1 realization grid time1 <<< "$base_hist"
IFS='_' read -r _ _ _ scenario2 _ _ time2 <<< "$base_proj"

echo "Model: $model"
echo "Scenario: $scenario2"

# ==========================
# Directories
# ==========================
merge_dir="/scratch/project_2014701/CMIP6_data/${var}/${scenario2}"
ts_dir="/scratch/project_2014701/input_data_regression/time_series/${var}/${scenario2}"
g11_dir="/scratch/project_2014701/input_data_regression/g11/${scenario2}"

mkdir -p "$merge_dir"
mkdir -p "$ts_dir"
mkdir -p "$g11_dir"

# ==========================
# Step 1: Merge historical + SSP
# ==========================
merged_file="${merge_dir}/${var}_day_${model}_${scenario1}_${scenario2}_${realization}_${grid}_1850-2100.nc"

echo "Merging files..."
cdo mergetime "$hist_file" "$proj_file" "$merged_file"

# ==========================
# Step 2: Remove leap days / Select correct period
# ==========================
noleap_file="${ts_dir}/${model}_${var}_${scenario2}_noleap.nc"
final_ts_file="${ts_dir}/${model}_${var}_${scenario2}_noleap_1900_2095.nc"

if [[ "$model" == "HadGEM3-GC31-LL" || "$model" == "KACE-1-0-G" || "$model" == "UKESM1-0-LL" ]]; then
    echo "360-day model (no leap removal)"
    cdo seldate,1900-01-01,2095-12-30 "$merged_file" "$final_ts_file"

elif [[ "$model" == "CAMS-CSM1-0" ]]; then
    echo "CAMS-CSM1-0 (ends at 2099)"
    cdo del29feb "$merged_file" "$noleap_file"
    cdo seldate,1900-01-01,2094-12-31 "$noleap_file" \
        "${ts_dir}/${model}_${var}_${scenario2}_noleap_1900_2094.nc"
    rm -f "$noleap_file"

else
    echo "Standard 365-day model"
    cdo del29feb "$merged_file" "$noleap_file"
    cdo seldate,1900-01-01,2095-12-31 "$noleap_file" "$final_ts_file"
    rm -f "$noleap_file"
fi

# ==========================
# Step 3: Compute g11
# ==========================
g11_temp1="${g11_dir}/${model}_g1.nc"
g11_temp2="${g11_dir}/${model}_g11_temp.nc"
g11_final="${g11_dir}/${model}_${scenario2}_g11.nc"

echo "Computing yearly global mean..."
cdo yearmean -fldmean "$merged_file" "$g11_temp1"

echo "Computing 11-year running mean..."
cdo runmean,11 "$g11_temp1" "$g11_temp2"

if [[ "$model" == "HadGEM3-GC31-LL" || "$model" == "KACE-1-0-G" || "$model" == "UKESM1-0-LL" ]]; then
    cdo seldate,1900-07-01,2095-07-01 "$g11_temp2" "$g11_final"
else
    cdo seldate,1900-07-02,2095-07-02 "$g11_temp2" "$g11_final"
fi

# Clean temporary files
rm -f "$g11_temp1" "$g11_temp2"

echo "Pre-processing complete for $model"

# ==========================
# Step 4: Run regression script
# ==========================

echo "Running regression calculation..."

ts_filename=$(basename "$final_ts_file")
g11_filename=$(basename "$g11_final")

python calculate_regression_coefficients.py \
    --ts_file "$ts_filename" \
    --g11_file "$g11_filename" \
    --n_days 1

echo "Regression completed for $model"




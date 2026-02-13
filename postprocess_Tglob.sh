#!/bin/bash

clim_var="$1"
ssp="$2"

anomaly_dir="/scratch/project_2014701/yearly_anomalies/${ssp}"

cd "$anomaly_dir" || exit 1

model_files=(*_renamed.nc)
num_models=${#model_files[@]}

echo "Found ${num_models} models"

# -------------------------
# 1. Merge all models
# -------------------------
merged_file="tas_Tglob_${ssp}_${num_models}mod_1901-2099_minus_2000.nc"
cdo merge "${model_files[@]}" "$merged_file"

# -------------------------
# 2. Ensemble mean
# -------------------------
ensmean_file="tas_Tglob_${ssp}_multi-model_mean_${num_models}mod_1901-2099_minus_2000.nc"
cdo ensmean "${model_files[@]}" "$ensmean_file"

# -------------------------
# 3. Rename tas → dt
# -------------------------
ncrename -v tas,dt "$ensmean_file"

echo "Yearly anomaly ensemble processing complete."


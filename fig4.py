# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 23:16:32 2025

@author: Jesse
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress
import pymannkendall as mk
import xarray as xr
import geopandas as gpd
import rioxarray 
from matplotlib.ticker import MaxNLocator

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12, 
    'figure.dpi': 300,
    'axes.linewidth': 0.8
})


# ==============================================================================
# PART 1: DATA PROCESSING FOR ALL PANELS
# ==============================================================================

# --- Processing for Panel (a) and (b): Caspian Surface Area ---
print("Processing: Caspian Sea Surface Area...")
file_path_area = "C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/AMIR_CS_WQ/CaspianSea_Annual_SurfaceAreatest.csv"
df_area = pd.read_csv(file_path_area)
df_area.drop(columns=["system:index"], inplace=True, errors="ignore")
df_area["year"] = df_area["year"].astype(int)
df_area["water_area_km2"] = df_area["water_area_km2"].astype(float)
df_area = df_area[df_area["year"] <= 2020].sort_values('year').reset_index(drop=True)
df_area["decade"] = pd.cut(df_area["year"], bins=[1999, 2009, 2020], labels=["2000s", "2010s"])

# --- Processing for Panel (b) and (c): Caspian Sea Evaporation ---
print("Processing: Caspian Sea Evaporation...")
file_path_evap = r"C:\Users\Jesse\Downloads\newEra5_CaspianSeaEE.nc"
ds_clip = xr.open_dataset(file_path_evap)
evap_mm_daily = ds_clip["e"] * -1000
evap_mm_daily = evap_mm_daily.assign_coords(valid_time=ds_clip["valid_time"])
days_in_month = evap_mm_daily["valid_time"].dt.days_in_month
evap_mm_monthly = evap_mm_daily * days_in_month
evap_filtered = evap_mm_monthly.sel(valid_time=slice("1991-01-01", "2020-12-31"))
evap_mm_per_year = evap_filtered.groupby(evap_filtered["valid_time"].dt.year).sum()
weights = np.cos(np.deg2rad(ds_clip.latitude))
weights.name = "weights"
evap_yearly_mean = evap_mm_per_year.weighted(weights).mean(dim=["latitude", "longitude"])
years_evap = evap_yearly_mean.year.values
evap_values = evap_yearly_mean.values

# --- Processing for Panel (d): Volga Basin P-ET ---
print("Processing: Volga Basin P-ET...")
def get_basin_annual_series(nc_path, var_name):
    ds = xr.open_dataset(nc_path)
    ds["time"] = ds["time"].dt.floor("D")
    w = np.cos(np.deg2rad(ds["lat"])); w.name = "weights"
    series = ds[var_name].where(np.isfinite(ds[var_name])).weighted(w).mean(("lat","lon"))
    annual = series.resample(time="Y").sum().sel(time=slice("1991","2020"))
    df = annual.to_dataframe(name=var_name).reset_index()
    df["Year"] = df["time"].dt.year
    return df[["Year", var_name]]
df_vol_precip = get_basin_annual_series(r"C:\Users\Jesse\Downloads\GPCC_VolgaBasin1.nc", "precip")
era5_path = r"C:\Users\Jesse\Desktop\CASPIAN SEA\DATA\5e5006746d0d8b6034b570b900b21e0c\data_1.nc"
ds_et_volga = xr.open_dataset(era5_path).rename({'valid_time': 'time'})
ds_et_volga = ds_et_volga.sel(time=slice("1991-01-01", "2020-12-31"))
e_mm_volga = ds_et_volga["e"].astype(float) * -1000
e_mm_volga.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
e_mm_volga.rio.write_crs("EPSG:4326", inplace=True)
volga_shp = gpd.read_file(r"C:\Users\Jesse\Desktop\DATASETS\Shapefiles\Volga\mrb_basins.shp")
e_clip_volga = e_mm_volga.rio.clip(volga_shp.geometry, volga_shp.crs, drop=True)
weights_volga = np.cos(np.deg2rad(e_clip_volga.latitude)); weights_volga.name = "weights"
e_weighted_volga = e_clip_volga.weighted(weights_volga).mean(dim=["latitude", "longitude"])
days_in_month_volga = e_weighted_volga.time.dt.days_in_month
e_monthly_mm_volga = e_weighted_volga * days_in_month_volga
e_annual_volga = e_monthly_mm_volga.resample(time="Y").sum()
df_et_volga = e_annual_volga.to_dataframe(name="ET").reset_index()
df_et_volga["Year"] = df_et_volga["time"].dt.year
df_pet_volga = pd.merge(df_vol_precip, df_et_volga[["Year", "ET"]], on="Year")
df_pet_volga["P_minus_ET"] = df_pet_volga["precip"] - df_pet_volga["ET"]

print("All data processed successfully.")


# ==============================================================================
# PART 2: PANEL PLOT CREATION
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# --- Panel (a): Caspian Sea Surface Area ---
ax = axes[0, 0]
X_area = sm.add_constant(df_area["year"])
model_area = sm.OLS(df_area["water_area_km2"], X_area).fit()
pred_area = model_area.get_prediction(X_area).summary_frame(alpha=0.05)
sns.scatterplot(data=df_area, x="year", y="water_area_km2", hue="decade",
                palette="viridis", s=50, edgecolor='k', alpha=0.8, ax=ax)
ax.plot(df_area["year"], pred_area["mean"], color='red', linestyle='--', linewidth=2,
        label=f"OLS Trend: {model_area.params.iloc[1]:.0f} km²/year")
ax.fill_between(df_area["year"], pred_area["mean_ci_lower"], pred_area["mean_ci_upper"],
                color='gray', alpha=0.3, label="95% CI")
ax.set_ylabel("Surface Area (km²)")
ax.set_xlabel("Year")
ax.legend(loc='lower left')
ax.grid(True, linestyle='--', alpha=0)
ax.set_xticks(np.arange(2000, 2021, 5))
ax.text(-0.1, 1.05, '(a)', transform=ax.transAxes, size=12, weight='bold')

# --- Panel (b): Caspian Evaporation Rate & Volume ---
ax1 = axes[0, 1]
ax2 = ax1.twinx()
years_area = df_area["year"].values
area_values = df_area["water_area_km2"].values
common_years = np.intersect1d(years_evap, years_area)
evap_values_common = evap_values[np.isin(years_evap, common_years)]
area_values_common = area_values[np.isin(years_area, common_years)]
evap_volume = (evap_values_common * area_values_common) / 1e6
slope_rate, intercept_rate, _, _, _ = linregress(years_evap, evap_values)
slope_vol, intercept_vol, _, _, _ = linregress(common_years, evap_volume)
trend_rate = slope_rate * years_evap + intercept_rate
trend_vol = slope_vol * common_years + intercept_vol
line_rate = ax1.plot(years_evap, evap_values, color='tab:red', linewidth=2, label="Evaporation Rate (mm/year)")[0]
line_rate_trend = ax1.plot(years_evap, trend_rate, color='tab:red', linestyle='--', linewidth=1.2, label=f"Trend: {slope_rate:+.2f} mm/year")[0]
line_vol = ax2.plot(common_years, evap_volume, color='black', linewidth=2, linestyle='--', label="Evaporation Volume (km³/year)")[0]
line_vol_trend = ax2.plot(common_years, trend_vol, color='black', linestyle=':', linewidth=1.2, label=f"Trend: {slope_vol:+.2f} km³/year")[0]
ax1.set_xlabel("Year")
ax1.set_ylabel("Evaporation (mm/year)", color='tab:red')
ax2.set_ylabel("Evaporation Volume (km³/year)", color='black')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax2.tick_params(axis='y', labelcolor='black')
ax1.legend(handles=[line_rate, line_rate_trend, line_vol, line_vol_trend], loc='upper left')
ax1.grid(True, linestyle='--', alpha=0)
ax1.text(-0.1, 1.05, '(b)', transform=ax1.transAxes, size=12, weight='bold')
ax1.grid(False)
ax2.grid(False)


# --- Panel (c): Contribution of Anomalous Evaporation
ax1 = axes[1, 0]
ax2 = ax1.twinx()
et_series = pd.Series(evap_values, index=years_evap)
et_aligned = et_series.reindex(years_area).to_numpy()
area_dyn = df_area['water_area_km2'].to_numpy()

#2000-2009
k = 10
a_start, a_end = int(years_area[0]), int(years_area[k-1])
ET_base_A = np.nanmean(et_aligned[:k])
et_anom_A = et_aligned - ET_base_A
V_extra_A = area_dyn * et_anom_A * 1e-6

# 1991-2000
mask_pre = (years_evap >= 1991) & (years_evap <= 2000)
ET_base_B = np.nanmean(evap_values[mask_pre])
et_anom_B = et_aligned - ET_base_B
V_extra_B = area_dyn * et_anom_B * 1e-6

width = 0.4
bar_A = ax1.bar(years_area - width/2, V_extra_A, width=width, color='#1f77b4', label=f'{a_start}-{a_end} Baseline')
bar_B = ax1.bar(years_area + width/2, V_extra_B, width=width, color='#ff7f0e', label='1991-2000 Baseline')

line_cumul_A = ax2.plot(years_area, np.cumsum(V_extra_A), color='#1f77b4', lw=2, ls='--', label=f'Cumulative (A): {np.nansum(V_extra_A):.2f} km³')[0]
line_cumul_B = ax2.plot(years_area, np.cumsum(V_extra_B), color='#ff7f0e', lw=2, ls='--', label=f'Cumulative (B): {np.nansum(V_extra_B):.2f} km³')[0]


ax1.set_xlabel("Year")
ax1.set_ylabel("Annual Extra Evaporation Volume (km³)")
ax2.set_ylabel("Cumulative Extra Volume (km³)")
ax1.legend(handles=[bar_A, bar_B], loc='upper left')
ax2.legend(handles=[line_cumul_A, line_cumul_B], loc='lower right')
ax1.grid(True, linestyle='--', alpha=0)
ax1.text(-0.1, 1.05, '(c)', transform=ax1.transAxes, size=12, weight='bold')
ax1.grid(False)
ax2.grid(False)


# --- Panel (d): Volga Basin Water Balance (P-ET) with Trend Lines ---
ax = axes[1, 1]
years_pet = df_pet_volga["Year"].values
p_volga = df_pet_volga["precip"].values
e_volga = df_pet_volga["ET"].values
pet_volga = df_pet_volga["P_minus_ET"].values

# Calculate OLS trend lines
slope_p, intercept_p, _, _, _ = linregress(years_pet, p_volga)
slope_e, intercept_e, _, _, _ = linregress(years_pet, e_volga)
slope_pet, intercept_pet, _, _, _ = linregress(years_pet, pet_volga)
trend_p = slope_p * years_pet + intercept_p
trend_e = slope_e * years_pet + intercept_e
trend_pet = slope_pet * years_pet + intercept_pet
ax.plot(years_pet, p_volga, color="#0072B2", marker='o', markersize=4, lw=1.5, label="Precipitation (P)")
ax.plot(years_pet, e_volga, color="#D55E00", marker='s', markersize=4, lw=1.5, label="Evapotranspiration (ET)")
ax.plot(years_pet, pet_volga, color="#009E73", marker='^', markersize=4, lw=2, label="P - ET")
ax.axhline(0, color='black', linestyle=':', linewidth=1.5)
ax.plot(years_pet, trend_p, ls='--', lw=1.2, color="#0072B2")
ax.plot(years_pet, trend_e, ls='--', lw=1.2, color="#D55E00")
ax.plot(years_pet, trend_pet, ls='--', lw=1.2, color="#009E73")

ax.set_ylabel("Atmospheric Input (mm/year)")
ax.set_xlabel("Year")
ax.legend()
ax.grid(True, linestyle='--', alpha=0)
ax.text(-0.1, 1.05, '(d)', transform=ax.transAxes, size=12, weight='bold')


plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("Caspian_Water_Balance_4-Panel_Full_Detail.png", dpi=600, bbox_inches="tight")
plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 23:28:27 2025

@author: Jesse
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:08:21 2025

@author: Jesse
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
import geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import linregress
import pymannkendall as mk
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

cas_path = r"C:\Users\Jesse\Downloads\GPCC_CaspianBasin1.nc"
vol_path = r"C:\Users\Jesse\Downloads\GPCC_VolgaBasin1.nc"
shp_path = r"C:\Users\Jesse\Downloads\Caspian Shapefile Standford\XCA_adm0.shp"
var = "precip"
period1 = slice("1991", "2005")
period2 = slice("2006", "2020")

# === Load Caspian and Volga ===
def load_basin_data(nc_path):
    ds = xr.open_dataset(nc_path)
    ds["time"] = xr.decode_cf(ds).time
    P = ds[var].where(np.isfinite(ds[var]))
    P_annual = P.resample(time="Y").sum()
    P1 = P_annual.sel(time=period1)
    P2 = P_annual.sel(time=period2)
    m1 = P1.mean("time")
    m2 = P2.mean("time")
    diff = m2 - m1
    return m1, m2, diff, P

m1_cas, m2_cas, diff_cas, P_cas = load_basin_data(cas_path)
m1_vol, m2_vol, diff_vol, P_vol = load_basin_data(vol_path)

vmin_cas = float(xr.concat([m1_cas, m2_cas], "z").quantile(0.02))
vmax_cas = float(xr.concat([m1_cas, m2_cas], "z").quantile(0.98))
vmin_vol = float(xr.concat([m1_vol, m2_vol], "z").quantile(0.02))
vmax_vol = float(xr.concat([m1_vol, m2_vol], "z").quantile(0.98))
vmin = min(vmin_cas, vmin_vol)
vmax = max(vmax_cas, vmax_vol)
dmax = max(np.nanpercentile(np.abs(diff_cas), 98), np.nanpercentile(np.abs(diff_vol), 98))
norm_diff = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)


try:
    caspian_gdf = gpd.read_file(shp_path)
except:
    caspian_gdf = None

# === Plotting Function for Spatial Maps ===
def plot_spatial_panel(ax, data, P, title, cmap, norm=None, cb_label="", show_colorbar=True, caspian_gdf=None):
    im = ax.pcolormesh(P.lon, P.lat, data, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14, fontweight='bold')  
    ax.coastlines(resolution="10m", linewidth=0.5)  
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)  
    ax.set_extent([P.lon.min(), P.lon.max(), P.lat.min(), P.lat.max()], crs=ccrs.PlateCarree())
    
    if caspian_gdf is not None:
        feature = ShapelyFeature(caspian_gdf.geometry, crs=ccrs.PlateCarree())
        ax.add_feature(feature, facecolor="#EDC373", edgecolor="none", alpha=0.25, zorder=10)
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, shrink=0.85)
        cbar.ax.tick_params(labelsize=11.7)  
        if cb_label:
            cbar.set_label(cb_label, fontsize=14)  
    return im

# === Prepare Time Series Data ===
def prepare_timeseries_data():
    # Load datasets
    ds_casp = xr.open_dataset(cas_path)
    ds_vol = xr.open_dataset(vol_path)
    
    ds_casp["time"] = ds_casp["time"].dt.floor("D")
    ds_vol["time"] = ds_vol["time"].dt.floor("D")
    weights_casp = np.cos(np.deg2rad(ds_casp.lat))
    weights_casp.name = "weights"
    weights_vol = np.cos(np.deg2rad(ds_vol.lat))
    weights_vol.name = "weights"
    
    # Caspian
    caspian_mean = ds_casp[var].weighted(weights_casp).mean(dim=["lat", "lon"])
    caspian_annual = caspian_mean.resample(time="Y").sum().sel(time=slice("1991", "2020"))
    df_casp = caspian_annual.to_dataframe().reset_index()
    df_casp["Year"] = df_casp["time"].dt.year
    
    # Volga
    volga_mean = ds_vol[var].weighted(weights_vol).mean(dim=["lat", "lon"])
    volga_annual = volga_mean.resample(time="Y").sum().sel(time=slice("1991", "2020"))
    df_vol = volga_annual.to_dataframe().reset_index()
    df_vol["Year"] = df_vol["time"].dt.year
    
    return df_casp, df_vol

# === Boxplot Data ===
def prepare_boxplot_data():
    def basin_annual_series(nc_path, var):
        ds = xr.open_dataset(nc_path)
        ds["time"] = ds["time"].dt.floor("D")
        w = np.cos(np.deg2rad(ds["lat"]))
        w.name = "weights"
        series = ds[var].where(np.isfinite(ds[var])).weighted(w).mean(("lat","lon"))
        annual = series.resample(time="Y").sum().sel(time=slice("1991","2020"))
        df = annual.to_dataframe().reset_index()
        df["Year"] = df["time"].dt.year
        return df[["Year", var]]
    
    df_casp = basin_annual_series(cas_path, var)
    df_vol = basin_annual_series(vol_path, var)
    
    df_casp["Basin"] = "Caspian"
    df_vol["Basin"] = "Volga"
    df_all = pd.concat([df_casp, df_vol], ignore_index=True)
    df_all["Period"] = np.where(df_all["Year"] <= 2005, "1991-2005", "2006-2020")
    df_all.rename(columns={var: "Precip_mm_yr"}, inplace=True)
    
    return df_all

# === MAIN FIGURE ===
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(20, 14)) 
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.15)

# Row 1: Volga panels
ax_a1 = fig.add_subplot(gs[0, 0], projection=proj)
ax_a2 = fig.add_subplot(gs[0, 1], projection=proj)
ax_a3 = fig.add_subplot(gs[0, 2], projection=proj)

# Row 2: Caspian panels
ax_b1 = fig.add_subplot(gs[1, 0], projection=proj)
ax_b2 = fig.add_subplot(gs[1, 1], projection=proj)
ax_b3 = fig.add_subplot(gs[1, 2], projection=proj)

# Row 3: Time series and boxplots
ax_c = fig.add_subplot(gs[2, 0:2])  
ax_d = fig.add_subplot(gs[2, 2])

# === ROW 1: VOLGA SPATIAL PLOTS ===
plot_spatial_panel(ax_a1, m1_vol, P_vol, "Volga P [1991–2005]", cmap="Blues", 
                   norm=plt.Normalize(vmin, vmax), cb_label="mm/yr")
plot_spatial_panel(ax_a2, m2_vol, P_vol, "Volga P [2006–2020]", cmap="Blues", 
                   norm=plt.Normalize(vmin, vmax), cb_label="mm/yr")
plot_spatial_panel(ax_a3, diff_vol, P_vol, "ΔP Volga [P2 − P1]", cmap="RdBu", 
                   norm=norm_diff, cb_label="mm/yr")

# === ROW 2: CASPIAN SPATIAL PLOTS ===
plot_spatial_panel(ax_b1, m1_cas, P_cas, "Caspian P [1991–2005]", cmap="Blues", 
                   norm=plt.Normalize(vmin, vmax), cb_label="mm/yr", 
                   caspian_gdf=caspian_gdf)
plot_spatial_panel(ax_b2, m2_cas, P_cas, "Caspian P [2006–2020]", cmap="Blues", 
                   norm=plt.Normalize(vmin, vmax), cb_label="mm/yr", 
                   caspian_gdf=caspian_gdf)
plot_spatial_panel(ax_b3, diff_cas, P_cas, "ΔP Caspian [P2 − P1]", cmap="RdBu", 
                   norm=norm_diff, cb_label="mm/yr", 
                   caspian_gdf=caspian_gdf)

# === ROW 3, PANEL C: TIME SERIES ===
df_casp_ts, df_vol_ts = prepare_timeseries_data()
years = df_casp_ts["Year"].values
precip_casp = df_casp_ts["precip"].values
precip_vol = df_vol_ts["precip"].values

slope_casp, intercept_casp, r_casp, p_casp, _ = linregress(years, precip_casp)
slope_vol, intercept_vol, r_vol, p_vol, _ = linregress(years, precip_vol)

trend_casp = slope_casp * years + intercept_casp
trend_vol = slope_vol * years + intercept_vol


ax_c.plot(years, precip_casp, color="#FF8042", marker='s', label="Caspian Basin", 
          linestyle="-", linewidth=2.5, markersize=6)  
ax_c.plot(years, trend_casp, linestyle="dashed", color="#FF8042", lw=2, 
          label=f"Caspian Trend ({slope_casp:.2f} mm/yr)")

ax_c.plot(years, precip_vol, color="black", marker='D', label="Volga Basin", 
          linewidth=2.5, markerfacecolor="white", markeredgecolor="black", markersize=6)
ax_c.plot(years, trend_vol, color="black", lw=2, ls="--", 
          label=f"Volga Trend ({slope_vol:.2f} mm/yr)")

ax_c.set_xlabel("Year", fontsize=15)  
ax_c.set_ylabel("Precipitation (mm/year)", fontsize=15)
ax_c.grid(True, linestyle="--", alpha=0)
ax_c.set_xlim(years.min() - 0.5, years.max() + 0.5)
ax_c.set_ylim(bottom=200)
ax_c.legend(loc="lower center", ncol=2, frameon=False, fontsize=13)
ax_c.set_title("Annual Precipitation Trends (1991-2020)", fontsize=15, fontweight='bold')
ax_c.tick_params(axis='both', labelsize=12)  

# === ROW 3, PANEL D: BOXPLOTS ===
df_all = prepare_boxplot_data()
sns.set_style("white")
cb = sns.color_palette("colorblind")
palette = {"Caspian": cb[1], "Volga": cb[0]}

meanprops = dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                 markersize=6, markeredgewidth=0.8)  

sns.boxplot(data=df_all, x="Period", y="Precip_mm_yr", hue="Basin",
            order=["1991-2005", "2006-2020"], hue_order=["Caspian", "Volga"],
            palette=palette, width=0.65, linewidth=1,
            showfliers=False, showmeans=True, meanprops=meanprops, ax=ax_d)

sns.stripplot(data=df_all, x="Period", y="Precip_mm_yr", hue="Basin",
              order=["1991-2005", "2006-2020"], hue_order=["Caspian", "Volga"],
              dodge=True, palette=palette, size=4.5, alpha=0.6,  
              linewidth=0.5, edgecolor="gray", ax=ax_d)

handles, labels = ax_d.get_legend_handles_labels()
ax_d.legend(handles[:2], labels[:2], loc="upper left", frameon=False, fontsize=12)  

ax_d.set_xlabel("", fontsize=15)
ax_d.set_ylabel("Annual Precipitation (mm/year)", fontsize=15)
ax_d.grid(axis="y", linestyle="-", alpha=0.3)
ax_d.set_title("Period Comparison", fontsize=15, fontweight='bold')
ax_d.tick_params(axis='both', labelsize=12)  
ax_d.set_xticklabels(ax_d.get_xticklabels(), fontsize=12)

# === Global panel labels aligned to the left of each row ===
fig.text(0.10, 0.87, '(a)', fontsize=18, fontweight='bold', va='center')  # Volga row
fig.text(0.10, 0.59, '(b)', fontsize=18, fontweight='bold', va='center')  # Caspian row
fig.text(0.10, 0.31, '(c)', fontsize=18, fontweight='bold', va='center')  # Time series
fig.text(0.65, 0.31, '(d)', fontsize=18, fontweight='bold', va='center')  # Boxplot
plt.tight_layout()

# Save figure
#plt.savefig("Full_Panel_Plot_Caspian_Volgat.png", dpi=600, bbox_inches="tight")
#plt.savefig("Full_Panel_Plot_Caspian_Volga.pdf", bbox_inches="tight")
plt.show()



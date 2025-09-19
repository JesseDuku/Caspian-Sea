# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 23:21:27 2025

@author: Jesse
"""
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from rasterio.plot import show
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr
import seaborn as sns
import shapely.geometry as sgeom
from cartopy.io import shapereader
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patheffects as pe
from pyproj import Geod
from matplotlib.patches import Rectangle, FancyArrow  
import pandas as pd
from matplotlib.dates import AutoDateLocator, DateFormatter
from scipy.stats import linregress
import pymannkendall as mk
import os
from scipy.stats import mannwhitneyu
from matplotlib.colors import TwoSlopeNorm
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from rasterio.features import rasterize

file_basin = r"C:\Users\Jesse\Desktop\CASPIAN SEA\DATA\GRACE_Trend\Grace_new_trend\CaspianTSBasin.txt"
file_individual = r"C:\Users\Jesse\Desktop\CASPIAN SEA\DATA\GRACE_Trend\Grace_new_trend\CaspianTSIndividual.txt"
df_basin = pd.read_csv(file_basin, delim_whitespace=True, skiprows=1)
df_basin = df_basin.drop(columns=['uncertainty'])
df_basin = df_basin.rename(columns={'TWSA.1': 'uncertainty'})
df_individual = pd.read_csv(file_individual, delim_whitespace=True, skiprows=2)
df_individual = df_individual.reset_index().rename(columns={'index': 'year', 'HBAS_ID': 'month'})

volga_id = '2030068680'
df_individual = df_individual[['year', 'month', volga_id]]
df_individual = df_individual.rename(columns={volga_id: 'Volga_TWSA'})
df_basin['TWSA'] *= 10
df_basin['uncertainty'] *= 10
df_individual['Volga_TWSA'] *= 10

# === Create datetime columns ===
df_basin['date'] = pd.to_datetime(df_basin[['year', 'month']].assign(day=1))
df_individual['date'] = pd.to_datetime(df_individual[['year', 'month']].assign(day=1))
end_date = pd.Timestamp("2020-12-31")
df_basin_plot = df_basin[df_basin['date'] <= end_date].copy()
df_individual_plot = df_individual[df_individual['date'] <= end_date].copy()

# === Mann-Kendall Trend Test ===
result_volga = mk.original_test(df_individual_plot['Volga_TWSA'])
result_caspian = mk.original_test(df_basin_plot['TWSA'])

fig, ax = plt.subplots(figsize=(9, 4))

# === Caspian Basin TWSA with uncertainty ===
ax.plot(df_basin_plot['date'], df_basin_plot['TWSA'], color='black', linewidth=3, label="Caspian Basin TWSA (Excl. Sea)")
ax.fill_between(df_basin_plot['date'],
                 df_basin_plot['TWSA'] - df_basin_plot['uncertainty'],
                 df_basin_plot['TWSA'] + df_basin_plot['uncertainty'],
                 color='gray', alpha=0.3)

# === Volga Basin TWSA ===
ax.plot(df_individual_plot['date'], df_individual_plot['Volga_TWSA'],
        color='red', linestyle= "--", linewidth=2, label="Volga Basin TWSA")

# === Axis labels and legend ===
ax.set_ylabel("TWSA (mm)", fontsize=13)
ax.set_xlabel("Year", fontsize=13)
ax.legend()
ax.grid(True, alpha=0)
#ax.text(0.01, 0.02, "(a)", transform=ax.transAxes,
        #fontsize=12, fontweight='bold', va='bottom', ha='left')

# === Spine and Tick Styling ===
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color("black")
ax.tick_params(direction='in', length=6, width=1.2, top=True, right=True)

plt.tight_layout()
plt.savefig("Caspian_Volga_TWSA_Superimposed_UpTo2023.png", dpi=600, bbox_inches="tight")
plt.show()
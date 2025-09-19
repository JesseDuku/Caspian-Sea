# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 23:38:05 2025

@author: Jesse
"""


import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


volga = pd.read_excel('C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/Rivers_till_2020/VOLGA_V_km3.xlsx').iloc[10:].reset_index(drop=True)
ural = pd.read_excel('C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/Rivers_till_2020/URAL_V_km3.xlsx').iloc[10:].reset_index(drop=True)
kura = pd.read_excel('C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/Rivers_till_2020/KURA_V_km3.xlsx').iloc[10:].reset_index(drop=True)
sefidrud = pd.read_excel('C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/Rivers_till_2020/sefidrud_V_km3.xlsx').iloc[10:].reset_index(drop=True)
terek = pd.read_excel('C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/Rivers_till_2020/TEREK_V_km3.xlsx').iloc[10:].reset_index(drop=True)

columns = ['Year','January','February','March','April','May','June','July','August','September','October','November',
           'December','value 2','value 1','Total_Discharge_km3']

rivers = [volga, ural, kura, sefidrud, terek]
river_names = ['Volga', 'Ural', 'Kura', 'Sefidrud', 'Terek']

for r in rivers:
    r.columns = columns
    r.drop(columns=['value 1','value 2'], inplace=True)

# Filter data (1991-2020)
volga_filtered = volga[(volga['Year'] >= 1991) & (volga['Year'] <= 2020)]
ural_filtered = ural[(ural['Year'] >= 1991) & (ural['Year'] <= 2020)]
kura_filtered = kura[(kura['Year'] >= 1991) & (kura['Year'] <= 2020)]
sefidrud_filtered = sefidrud[(sefidrud['Year'] >= 1991) & (sefidrud['Year'] <= 2020)]
terek_filtered = terek[(terek['Year'] >= 1991) & (terek['Year'] <= 2020)]

df_box = pd.DataFrame({
    'Volga (Russia)': volga_filtered['Total_Discharge_km3'],
    'Ural (Russia/Kazakhstan)': ural_filtered['Total_Discharge_km3'],
    'Kura (Azerbaijan)': kura_filtered['Total_Discharge_km3'],
    'Sefidrud (Iran)': sefidrud_filtered['Total_Discharge_km3'],
    'Terek (Russia)': terek_filtered['Total_Discharge_km3']
})

# Merge on 'Year' 
df_total = volga_filtered[['Year', 'Total_Discharge_km3']].rename(columns={'Total_Discharge_km3': 'Volga'})
df_total = df_total.merge(ural_filtered[['Year', 'Total_Discharge_km3']].rename(columns={'Total_Discharge_km3': 'Ural'}), on='Year', how='left')
df_total = df_total.merge(kura_filtered[['Year', 'Total_Discharge_km3']].rename(columns={'Total_Discharge_km3': 'Kura'}), on='Year', how='left')
df_total = df_total.merge(sefidrud_filtered[['Year', 'Total_Discharge_km3']].rename(columns={'Total_Discharge_km3': 'Sefidrud'}), on='Year', how='left')
df_total = df_total.merge(terek_filtered[['Year', 'Total_Discharge_km3']].rename(columns={'Total_Discharge_km3': 'Terek'}), on='Year', how='left')

# Compute total inflow per year 
df_total['Total_Inflow_km3'] = df_total[['Volga', 'Ural', 'Kura', 'Sefidrud', 'Terek']].sum(axis=1, skipna=True)

df_casp = basin_annual_series(caspian_path, var)
df_vol  = basin_annual_series(volga_path,   var)

# Merge precipitation with discharge
df_casp.columns = ['Year', 'Caspian_Precip_mm']
df_vol.columns  = ['Year', 'Volga_Precip_mm']

# Merge all into df_total
df_total = df_total.merge(df_casp, on='Year', how='left')
df_total = df_total.merge(df_vol,  on='Year', how='left')

# === Split into two periods
first_half = df_total[df_total['Year'] <= 2005]
second_half = df_total[df_total['Year'] > 2005]

# === Total inflow & Caspian precipitation
mean_total_inflow_1 = first_half['Total_Inflow_km3'].mean()
mean_total_inflow_2 = second_half['Total_Inflow_km3'].mean()

mean_casp_precip_1 = first_half['Caspian_Precip_mm'].mean()
mean_casp_precip_2 = second_half['Caspian_Precip_mm'].mean()

# === Volga discharge & Volga precipitation
mean_volga_discharge_1 = first_half['Volga'].mean()
mean_volga_discharge_2 = second_half['Volga'].mean()

mean_volga_precip_1 = first_half['Volga_Precip_mm'].mean()
mean_volga_precip_2 = second_half['Volga_Precip_mm'].mean()

# === Print results
print("Total Inflow (1991–2005):", round(mean_total_inflow_1, 2), "km³")
print("Total Inflow (2006–2020):", round(mean_total_inflow_2, 2), "km³")
print("Caspian Precip (1991–2005):", round(mean_casp_precip_1, 2), "mm")
print("Caspian Precip (2006–2020):", round(mean_casp_precip_2, 2), "mm")
print()
print("Volga Discharge (1991–2005):", round(mean_volga_discharge_1, 2), "km³")
print("Volga Discharge (2006–2020):", round(mean_volga_discharge_2, 2), "km³")
print("Volga Precip (1991–2005):", round(mean_volga_precip_1, 2), "mm")
print("Volga Precip (2006–2020):", round(mean_volga_precip_2, 2), "mm")


# --- Long-term contributions (1991–2020 cumulative) ---
river_totals = df_total[['Volga','Ural','Kura','Sefidrud','Terek']].sum()
total_inflow_all = river_totals.sum()

df_contrib_longterm = (river_totals / total_inflow_all * 100).round(2)

print("\n=== Long-term % contributions (1991–2020) ===")
print(df_contrib_longterm)



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,7), sharex=True)


############################################
# PANEL 1: Caspian Basin Total Inflow vs. Precip
############################################
ax1 = axes[0]
ax2 = ax1.twinx()

ax1.plot(df_total["Year"],df_total["Caspian_Precip_mm"],
         color='#00C49F', linestyle='-', linewidth=2.5, label="Caspian basin Precipitation")
ax2.plot(df_total["Year"], df_total["Total_Inflow_km3"],
         color='#0088FE', linewidth=4, marker='o', markersize=4, label="Caspian Inflow")
ax1.axvline(x=2005.5, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.7)  # Vertical line at 2005
#ax2.axvline(x=2005, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.7)  # Vertical line for twin y-axis
ax1.text(0.02, 0.02, '(a)', transform=ax1.transAxes, fontsize=14, weight='bold', va='bottom', ha='left')

# Mean lines (first/second half) - precipitation
ax1.axhline(mean_casp_precip_1, color='#00C49F', linestyle='dashed', xmin=0, xmax=0.5)
ax1.axhline(mean_casp_precip_2, color='#00C49F', linestyle='dashed', xmin=0.5, xmax=1)

# Mean lines (first/second half) - inflow
ax2.axhline(mean_total_inflow_1, color='#0088FE', linestyle='dashed', xmin=0, xmax=0.5)
ax2.axhline(mean_total_inflow_2, color='#0088FE', linestyle='dashed', xmin=0.5, xmax=1)
#####################################################################################################
# Annotate first and second half mean for precipitation (Caspian)
#ax1.text(2001.5, mean_casp_precip_1, f"{mean_casp_precip_1:.2f} mm/yr", 
        # color='#00C49F', ha='right', va='bottom', fontsize=12)
#ax1.text(2011, mean_casp_precip_2-2, f"{mean_casp_precip_2:.2f} mm/yr", 
        # color='#00C49F', ha='left', va='top', fontsize=12)

# Annotate first and second half mean for inflow (Caspian)
#ax2.text(2004.5, mean_total_inflow_1, f"{mean_total_inflow_1:.2f} km³/yr", 
        # color='#0088FE', ha='right', va='bottom', fontsize=12)
#ax2.text(2016, mean_total_inflow_2-2, f"{mean_total_inflow_2:.2f} km³/yr", 
 #        color='#0088FE', ha='left', va='top', fontsize=12)

###################################################################################################
# Labels
ax1.set_ylabel("Precipitation (mm)", fontsize=14, color='black')
ax2.set_ylabel("Total Inflow (km³)", fontsize=14, color='black')
#ax1.set_title("Total Inflow vs. Caspian Basin Precipitation (1991–2020)",
              #fontsize=14, fontweight='bold')

# Custom legend for Caspian
legend_caspian = [
    mpatches.Patch(color='#0088FE', label="Caspian Inflow"),
    mpatches.Patch(color='#00C49F', label="Caspian basin Precipitation")
]
ax1.legend(handles=legend_caspian, loc="upper right", fontsize=10, frameon=False)

############################################
# PANEL 2: Volga Basin Discharge vs. Precip
############################################
ax3 = axes[1]
ax4 = ax3.twinx()

ax3.plot(df_total["Year"], df_total["Volga_Precip_mm"],
         color='black', linestyle='-', linewidth=2.5, label="Volga Precipitation")
ax4.plot(df_total["Year"], df_total["Volga"],
         color='#FF8042', linewidth=4, marker='o', markersize=4, label="Volga Inflow")
ax3.axvline(x=2005.5, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.7)  
ax4.axvline(x=2005.5, color='gray', linestyle='dashed', linewidth=1.5, alpha=0.7)  
# Mean lines (first/second half) - precipitation
ax3.axhline(mean_volga_precip_1, color='black', linestyle='dashed', xmin=0, xmax=0.5)
ax3.axhline(mean_volga_precip_2, color='black', linestyle='dashed', xmin=0.5, xmax=1)
ax3.text(0.02, 0.02, '(b)', transform=ax3.transAxes, fontsize=14, weight='bold',
         va='bottom', ha='left', zorder=10)

# Mean lines (first/second half) - discharge
ax4.axhline(mean_volga_discharge_1, color='#FF8042', linestyle='dashed', xmin=0, xmax=0.5)
ax4.axhline(mean_volga_discharge_2, color='#FF8042', linestyle='dashed', xmin=0.5, xmax=1)
###################################################################################################
# Annotate first and second half mean for precipitation (Volga)
#ax3.text(1992.8, mean_volga_precip_1, f"{mean_volga_precip_1:.2f} mm/yr", 
         #color='black', ha='right', va='bottom', fontsize=12)
#ax3.text(2015, mean_volga_precip_2 - 1, f"{mean_volga_precip_2:.2f} mm/yr", 
         #color='black', ha='left', va='bottom', fontsize=12)


# Annotate first and second half mean for discharge (Volga)
#ax4.text(1992.8, mean_volga_discharge_1-2, f"{mean_volga_discharge_1:.2f} km³/yr", 
         #color='#FF8042', ha='right', va='top', fontsize=12)
#ax4.text(2015.8, mean_volga_discharge_2-2, f"{mean_volga_discharge_2:.2f} km³/yr", 
         #color='#FF8042', ha='left', va='top', fontsize=12)
######################################################################################################

# Labels
ax3.set_ylabel("Precipitation (mm)", fontsize=14, color='black')
ax4.set_ylabel("Discharge (km³)", fontsize=14, color='black')
#ax3.set_title("Total Discharge vs. Volga Basin Precipitation (1991–2020)",
              #fontsize=14, fontweight='bold')

legend_volga = [
    mpatches.Patch(color='#FF8042', label="Volga Inflow"),
    mpatches.Patch(color='black', label="Volga Precipitation")
]
ax3.set_xlabel("Year", fontsize=14, color='black')


ax3.legend(handles=legend_volga, loc="upper right", fontsize=10, frameon=False)
plt.tight_layout()
#plt.savefig("panelplotdischargeprecip", dpi=300, bbox_inches="tight")
plt.show()
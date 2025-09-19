# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:27:49 2024

@author: Jesse
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# Load Data
df3 = pd.read_excel('C:/Users/Jesse/Desktop/CASPIAN SEA/DATA/AMIR_CS_WQ/MAY-November.xlsx')


years = df3["Year"]
chl_a = df3["May-November Average Chl-a Concentration"]
slope, intercept, r_value, p_value, std_err = linregress(years, chl_a)
trendline = slope * years + intercept
trend_label = trend_label = f"Trend: {slope:.3f} µg/L/yr"

plt.figure(figsize=(6,3.5), dpi=300)
sns.set_style("whitegrid")

plt.scatter(years, chl_a, color="#1f77b4", s=150, marker="o", edgecolor="black", label="May–November Avg Chl-a", alpha=0.8)
plt.plot(years, trendline, linestyle="--", color="red", linewidth=2, label=trend_label)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Chl-a Concentration (µg/L)", fontsize=12)
plt.xticks(np.arange(2002, 2022, 4), fontsize=12)
plt.yticks(fontsize=15)
ax = plt.gca()
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(1.5)
    ax.spines[spine].set_color("black")

# Legend
plt.legend(fontsize=13, loc="lower left")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0)
plt.ylim(0, 15)
plt.yticks(np.arange(0, 16, 5))  
plt.tight_layout()
#plt.savefig('May_November_ChlA_Styled.png', dpi=600)
plt.show()

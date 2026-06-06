#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# 1. CONFIGURATION AND BASE PATHS
# =============================================================================
BASE_PATH = Path(r"C:\Users\FlawiyaShirishMore\Downloads\Africa-Drought-Study\data")
OUTPUT_DIR = BASE_PATH / "Master_Analysis_Outputs"
MASTER_CACHE = BASE_PATH / "master_processed_ssi.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DISTRICTS = [
    'Chirundu', 'Choma', 'Gwembe', 'Kalomo', 'Kazungula', 
    'Mazabuka', 'Monze', 'Namwala', 'Pemba', 'Siavonga', 
    'Sinazongwe', 'Zimba'
]

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_crop_year(row):
    return row['date'].year + 1 if row['date'].month in [11, 12] else row['date'].year

def calculate_sheffield_baseline(df, col, window=31):
    daily_stats = df.groupby(['district', 'day_of_year'])[col].agg(['mean', 'std']).reset_index()
    results = []
    for district, group in daily_stats.groupby('district'):
        group = group.sort_values('day_of_year')
        triple = pd.concat([group] * 3).reset_index(drop=True)
        triple['mu_stable'] = triple['mean'].rolling(window=window, center=True).mean()
        triple['std_stable'] = triple['std'].rolling(window=window, center=True).mean()
        results.append(triple.iloc[len(group):2*len(group)].copy())
    return pd.concat(results, ignore_index=True)[['district', 'day_of_year', 'mu_stable', 'std_stable']]

def calculate_neg_integral(series, threshold=-0.8):
    stress = series[series < threshold]
    return (stress - threshold).sum() if not stress.empty else 0

# =============================================================================
# 3. DATA LOADING & SMART PROCESSING
# =============================================================================

if MASTER_CACHE.exists():
    print(f"✅ Loading cached processed data: {MASTER_CACHE}")
    master = pd.read_csv(MASTER_CACHE)
    master['date'] = pd.to_datetime(master['date'])
else:
    print("📂 Processing Raw Files...")
    df_ndvi = pd.read_csv(BASE_PATH / "master_southern_province_ndvi.csv")
    df_soil = pd.read_csv(BASE_PATH / "master_southern_province_soil-moisture-layer2.csv")
    df_lst = pd.read_csv(BASE_PATH / "master_southern_province_lst.csv")
    df_climate = pd.read_csv(BASE_PATH / "climate_merged.csv")
    df_spei3 = pd.read_csv(BASE_PATH / "master_southern_province_data.csv")

    dataframes = [df_ndvi, df_soil, df_lst, df_climate, df_spei3]
    master = dataframes[0]
    for df in dataframes[1:]:
        if 'district' in df.columns:
            keys = ['district', 'date', 'year', 'month', 'day']
            actual_keys = [k for k in keys if k in df.columns and k in master.columns]
            master = pd.merge(master, df, on=actual_keys, how='outer')
        else:
            cols_to_keep = [c for c in df.columns if c not in ['year', 'month', 'day', 'district']]
            master = pd.merge(master, df[cols_to_keep], on='date', how='left')

    master.replace(-999, np.nan, inplace=True)
    master['date'] = pd.to_datetime(master['date'])
    master['day_of_year'] = master['date'].dt.dayofyear
    
    if 'soil_moisture_layer2' in master.columns:
        master = master.rename(columns={'soil_moisture_layer2': 'soil_moisture_7_28'})

    # GAP FILLING
    sat_cols = ['ndvi', 'lst_celsius']
    master[sat_cols] = master.groupby('district')[sat_cols].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit=30)).reset_index(level=0, drop=True)
    lst_clim = master.groupby(['district', 'day_of_year'])['lst_celsius'].transform('mean')
    master['lst_celsius'] = master['lst_celsius'].fillna(lst_clim)

    # CALCULATIONS
    ssi_base = calculate_sheffield_baseline(master, 'soil_moisture_7_28')
    master = master.merge(ssi_base, on=['district', 'day_of_year'], how='left')
    master['SSI'] = (master['soil_moisture_7_28'] - master['mu_stable']) / (master['std_stable'] + 1e-6)

    master['D'] = master['precip_mm'].clip(lower=0) - master['pet_mm'].abs()
    master['D_90'] = master.groupby('district')['D'].transform(lambda x: x.rolling(90, min_periods=28).sum())
    spei_base = calculate_sheffield_baseline(master, 'D_90').rename(columns={'mu_stable':'mu_d90', 'std_stable':'std_d90'})
    master = master.merge(spei_base, on=['district', 'day_of_year'], how='left')
    master['SPEI3'] = (master['D_90'] - master['mu_d90']) / (master['std_d90'] + 1e-6)

    master['VCI_z'] = master.groupby(['district', 'day_of_year'])['ndvi'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    master['TCI_z'] = master.groupby(['district', 'day_of_year'])['lst_celsius'].transform(lambda x: (x.mean() - x) / (x.std() + 1e-6))
    master['VHI'] = (0.5 * master['VCI_z']) + (0.5 * master['TCI_z'])

    for col in ['SSI', 'SPEI3', 'VHI']:
        master[col] = master.groupby('district')[col].transform(lambda x: x.rolling(7, center=True, min_periods=1).mean())

    master.to_csv(MASTER_CACHE, index=False)

# =============================================================================
# 4. SEASONAL AGGREGATION
# =============================================================================
master['crop_year'] = master.apply(get_crop_year, axis=1)
master_crop = master[master['date'].dt.month.isin([11, 12, 1, 2, 3, 4, 5, 6, 7, 8])].copy()
master_crop.loc[master_crop['crop_year'] == 2024, 'NINO34'] = 2.03

df_seasonal = master_crop.groupby(['district', 'crop_year']).agg({
    'SSI': ['mean', calculate_neg_integral], 'VHI': 'mean', 'SPEI3': 'mean',
    'ndvi': 'mean', 'precip_mm': 'sum', 'NINO34': 'mean'
}).reset_index()
df_seasonal.columns = ['district', 'crop_year', 'SSI_seasonal', 'Drought_Energy', 'VHI_seasonal', 'SPEI3_seasonal', 'NDVI_seasonal', 'Total_Precip', 'NINO34']

# =============================================================================
# 5. GENERATING OUTPUTS (BOSS-READY COLORS)
# =============================================================================

# 1. DROUGHT CASCADE 2024 (ORIGINAL COLORS RESTORED)
print("📊 Rendering Spatiotemporal Drought Cascade (2024)...")
df_2024 = master_crop[master_crop['crop_year'] == 2024].copy()
df_prov_avg = df_2024.groupby('date').mean(numeric_only=True).reset_index()

fig_casc = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("1. Meteorological: Daily Rainfall vs. Evaporation (PET)", 
                                    "2. Hydrological: Daily SSI (Julian-Day Baseline)", 
                                    "3. Cumulative: SPEI-3 (90-Day Anomaly)", 
                                    "4. Biological: Vegetation Health Index (VHI)"))

# Row 1: Rain/PET
fig_casc.add_trace(go.Bar(x=df_prov_avg['date'], y=df_prov_avg['precip_mm'], name="Avg Rain", marker_color='dodgerblue'), row=1, col=1)
fig_casc.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg['pet_mm'].abs(), name="Avg PET", line=dict(color='crimson', width=2)), row=1, col=1)

# Row 2 & 3: Red/Blue Anomalies
for r, col in [(2, 'SSI'), (3, 'SPEI3')]:
    for d in TARGET_DISTRICTS:
        if d in df_2024['district'].values:
            df_d = df_2024[df_2024['district'] == d]
            fig_casc.add_trace(go.Scatter(x=df_d['date'], y=df_d[col], line=dict(color='rgba(150,150,150,0.12)', width=1), showlegend=False), row=r, col=1)
    
    fig_casc.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg[col].clip(lower=0), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)', line_color='blue', name=f"Prov {col} (Wet)"), row=r, col=1)
    fig_casc.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg[col].clip(upper=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)', line_color='red', name=f"Prov {col} (Dry)"), row=r, col=1)
    fig_casc.add_hline(y=-1.2, line_dash="dash", line_color="darkred", annotation_text="Extreme Drought", row=r, col=1)

# Row 4: Biological VHI
fig_casc.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg['VHI'], line=dict(color='darkgreen', width=3), name="Avg VHI"), row=4, col=1)
fig_casc.update_layout(height=1200, title_text="Southern Province: Spatiotemporal Drought Analysis (2024)", template="plotly_white")
fig_casc.write_html(OUTPUT_DIR / "Drought_Cascade_2024.html")

# 2. HISTORICAL SSI MATRIX
plt.figure(figsize=(15, 9))
pivot_ssi = df_seasonal.pivot(index='crop_year', columns='district', values='SSI_seasonal')
sns.heatmap(pivot_ssi[pivot_ssi.index >= 2001], cmap='RdBu', center=0, annot=True, fmt=".1f", linewidths=.5)
plt.title("Southern Province: Historical Drought Matrix (Nov-Aug)")
plt.savefig(OUTPUT_DIR / "Historical_SSI_Matrix.png")

# 3. INTERACTIVE VERIFICATION
df_seasonal['SEASON'] = (df_seasonal['crop_year'] - 1).astype(str) + "-" + df_seasonal['crop_year'].astype(str)
fig_verif = px.line(df_seasonal, x="SEASON", y="SSI_seasonal", color="district", title="District Drought Verification (SSI)")
for yr in df_seasonal[df_seasonal['NINO34'] >= 1.0]['SEASON'].unique():
    fig_verif.add_vrect(x0=yr, x1=yr, fillcolor="red", opacity=0.1, layer="below", line_width=0)
fig_verif.write_html(OUTPUT_DIR / "Interactive_SSI_Verification.html")

print(f"✨ DONE! Results saved in: {OUTPUT_DIR}")

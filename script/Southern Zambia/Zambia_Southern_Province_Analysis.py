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

# ==========================================
# 1. SETUP AND DATA LOADING
# ==========================================

# Note: Update this path to your local environment if necessary
project_root = Path(__file__).resolve().parent.parent
default_data_path = project_root / "outputs" / "Zambia" / "Files"
fallback_data_path = project_root / "C:\\Users\\FlawiyaShirishMore\\Downloads\\Africa-Drought-Study\\data"

if default_data_path.exists():
    base_path = default_data_path
elif fallback_data_path.exists():
    base_path = fallback_data_path
else:
    raise FileNotFoundError(
        f"Data directory not found. Checked:\n"
        f"  {default_data_path}\n"
        f"  {fallback_data_path}\n"
        "Please place the required input files in one of these locations."
    )

expected_files = [
    "master_southern_province_ndvi.csv",
    "master_southern_province_soil-moisture-layer2.csv",
    "master_southern_province_lst.csv",
    "climate_merged.csv",
    "master_southern_province_data.csv",
]
missing_files = [f for f in expected_files if not (base_path / f).exists()]
if missing_files:
    raise FileNotFoundError(
        f"Missing required input files in {base_path}:\n" +
        "\n".join(f"  - {f}" for f in missing_files) +
        "\nPlease make sure these files are present in the selected data directory."
    )

OUTPUT_DIR = project_root / "outputs" / "Week_4_Report_Zambia_South"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge_data(path):
    print("Loading dataframes...")
    df_ndvi = pd.read_csv(os.path.join(path, "master_southern_province_ndvi.csv"))
    df_soil = pd.read_csv(os.path.join(path, "master_southern_province_soil-moisture-layer2.csv"))
    df_lst = pd.read_csv(os.path.join(path, "master_southern_province_lst.csv"))
    df_climate = pd.read_csv(os.path.join(path, "climate_merged.csv"))
    df_spei3 = pd.read_csv(os.path.join(path, "master_southern_province_data.csv"))

    # Sequential "Safe Merge"
    dataframes = [df_ndvi, df_soil, df_lst, df_climate, df_spei3]
    master_final = dataframes[0]

    for df in dataframes[1:]:
        if 'district' in df.columns and 'district' in master_final.columns:
            keys = ['district', 'date', 'year', 'month', 'day']
            actual_keys = [k for k in keys if k in df.columns and k in master_final.columns]
            master_final = pd.merge(master_final, df, on=actual_keys, how='outer')
        else:
            cols_to_keep = [c for c in df.columns if c not in ['year', 'month', 'day', 'district']]
            master_final = pd.merge(master_final, df[cols_to_keep], on='date', how='left')

    # Data Clean-up
    master_final.replace(-999, np.nan, inplace=True)
    if 'soil_moisture_layer2' in master_final.columns:
        master_final = master_final.rename(columns={'soil_moisture_layer2': 'soil_moisture_7_28'})
    
    return master_final, df_climate

master_final, df_climate = load_and_merge_data(base_path)

# ==========================================
# 2. TEMPORAL RECONSTRUCTION (GAP FILLING)
# ==========================================

def clean_and_fill_data(df):
    print("Processing temporal gaps and LST reconstruction...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['district', 'date'])

    # Step 1: Climate Indices (Monthly to Daily)
    climate_cols = ['NAO', 'DMI', 'NINO34', 'TSA', 'EA']
    df[climate_cols] = df.groupby('district')[climate_cols].ffill()

    # Step 2: Vegetation and LST (Satellite Gaps)
    sat_cols = ['ndvi', 'ndvi_raw', 'evi', 'evi_raw', 'lst_celsius']
    df[sat_cols] = df.groupby('district')[sat_cols].apply(
        lambda x: x.interpolate(method='linear', limit_direction='both', limit=30)
    ).reset_index(level=0, drop=True)

    # Step 3: Soil Moisture & Climate (Reanalysis)
    daily_cols = ['soil_moisture_7_28', 'pet_mm', 'precip_mm']
    df[daily_cols] = df.groupby('district')[daily_cols].apply(
        lambda x: x.interpolate(method='linear', limit=7)
    ).reset_index(level=0, drop=True)

    # Step 4: LST Climatological Reanalysis
    df['day_of_year'] = df['date'].dt.dayofyear
    lst_climatology = df.groupby(['district', 'day_of_year'])['lst_celsius'].transform('mean')
    df['lst_celsius'] = df['lst_celsius'].fillna(lst_climatology)
    
    # Fallback to monthly
    lst_monthly_climatology = df.groupby(['district', 'month'])['lst_celsius'].transform('mean')
    df['lst_celsius'] = df['lst_celsius'].fillna(lst_monthly_climatology)

    # Final Filter
    df.dropna(subset=['ndvi', 'soil_moisture_7_28'], inplace=True)
    return df

master_final = clean_and_fill_data(master_final)

# ==========================================
# 3. DROUGHT INDEX CALCULATIONS
# ==========================================

def calculate_indices(df):
    print("Calculating SSI, VHI, and SPEI3...")
    # SSI
    df['SSI'] = df.groupby(['district', 'month'])['soil_moisture_7_28'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # VHI
    stats = df.groupby('district').agg({
        'ndvi': ['min', 'max'],
        'lst_celsius': ['min', 'max']
    })
    stats.columns = ['n_min', 'n_max', 'l_min', 'l_max']
    df = df.join(stats, on='district')
    
    df['VCI'] = (df['ndvi'] - df['n_min']) / (df['n_max'] - df['n_min'] + 1e-6)
    df['TCI'] = (df['l_max'] - df['lst_celsius']) / (df['l_max'] - df['l_min'] + 1e-6)
    df['VHI'] = 0.5 * df['VCI'] + 0.5 * df['TCI']
    df = df.drop(columns=['n_min', 'n_max', 'l_min', 'l_max'])

    # SPEI-3
    df['D'] = df['precip_mm'] + df['pet_mm']
    df['D3'] = df.groupby('district')['D'].transform(
        lambda x: x.rolling(window=90, min_periods=30).sum()
    )
    df['SPEI3'] = df.groupby(['district', 'month'])['D3'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    return df

master_final = calculate_indices(master_final)

# ==========================================
# 4. SEASONAL AGGREGATION (CROP YEAR)
# ==========================================

# Define Crop Year
master_final['CropYear'] = np.where(master_final['month'] >= 11, master_final['year'] + 1, master_final['year'])

# Filter for Maize Season (Nov-April)
maize_season_months = [11, 12, 1, 2, 3, 4]
df_maize_season = master_final[master_final['month'].isin(maize_season_months)].copy()

agg_columns = {
    'SSI': 'mean', 'VHI': 'mean', 'SPEI3': 'mean', 'ndvi': 'mean',
    'soil_moisture_7_28': 'mean', 'lst_celsius': 'mean',
    'precip_mm': 'sum', 'pet_mm': 'sum'
}

df_seasonal_agg = df_maize_season.groupby(['district', 'CropYear']).agg(agg_columns).reset_index()
df_seasonal_agg.columns = [
    'district', 'CropYear', 'SSI_seasonal', 'VHI_seasonal', 'SPEI3_seasonal', 
    'NDVI_seasonal', 'SoilMoist_7_28_seasonal', 'LST_seasonal', 'Total_Precip', 'Total_PET'
]
df_seasonal_agg['province'] = 'Southern'

# Prepare NINO34 data
df_climate['date'] = pd.to_datetime(df_climate['date'])
df_climate['month'] = df_climate['date'].dt.month
df_climate['year'] = df_climate['date'].dt.year
df_climate['CropYear'] = np.where(df_climate['month'] >= 11, df_climate['year'] + 1, df_climate['year'])
nino_seasonal = df_climate[df_climate['month'].isin(maize_season_months)].groupby('CropYear')['NINO34'].mean().reset_index()
nino_seasonal.loc[nino_seasonal['CropYear'] == 2024, 'NINO34'] = 2.03

# ==========================================
# 5. VISUALIZATIONS (PLOTLY & SEABORN)
# ==========================================

def plot_district_verification(df, nino_df, metric='SSI_seasonal'):
    print(f"Generating Plotly verification for {metric}...")
    data = df.copy()
    data['SEASON'] = (data['CropYear'] - 1).astype(str) + "-" + data['CropYear'].astype(str)
    data = pd.merge(data, nino_df[['CropYear', 'NINO34']], on='CropYear', how='left')
    
    fig = px.line(data, x="SEASON", y=metric, color="district",
                  title=f"Southern Province: District {metric} Verification")

    enso_events = data[['SEASON', 'NINO34']].drop_duplicates()
    for _, row in enso_events.iterrows():
        if pd.isna(row['NINO34']): continue
        if row['NINO34'] >= 0.5:
            fig.add_vrect(x0=row['SEASON'], x1=row['SEASON'], fillcolor="red", opacity=0.1, layer="below", line_width=0)
        elif row['NINO34'] <= -0.5:
            fig.add_vrect(x0=row['SEASON'], x1=row['SEASON'], fillcolor="blue", opacity=0.1, layer="below", line_width=0)
    
    fig.update_layout(template="plotly_white", hovermode="x unified")
    fig.write_html(OUTPUT_DIR / f"Southern_Province_{metric}_Verification.html")
    # fig.show()

# Generate plots
plot_district_verification(df_seasonal_agg, nino_seasonal, 'SSI_seasonal')
plot_district_verification(df_seasonal_agg, nino_seasonal, 'VHI_seasonal')

# ==========================================
# 6. CORRELATION ANALYSIS
# ==========================================

def generate_correlation_matrix(df, nino_df):
    print("Generating correlation heatmaps...")
    nino_temp = nino_df.rename(columns={'NINO34': 'nino34'})
    data = pd.merge(df, nino_temp[['CropYear', 'nino34']], on='CropYear', how='left')
    
    corr_cols = ['nino34', 'SSI_seasonal', 'NDVI_seasonal', 'SoilMoist_7_28_seasonal', 'VHI_seasonal', 'SPEI3_seasonal']
    corr_matrix = data[data['province'] == 'Southern'][corr_cols].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, fmt='.2f', linewidths=0.5)
    plt.title("Southern Province: Teleconnection & Index Correlation")
    plt.savefig(OUTPUT_DIR / "Southern_Province_NINO_Correlation_Matrix.png")
    # plt.show()

generate_correlation_matrix(df_seasonal_agg, nino_seasonal)

# ==========================================
# 7. FINAL STYLIZED REPORT (HTML)
# ==========================================

def save_styled_report(df, nino_df):
    print("Saving final validation report...")
    report_prov = df.groupby('CropYear').agg({
        'NDVI_seasonal': 'mean', 'Total_Precip': 'mean', 'SSI_seasonal': 'mean'
    }).reset_index()
    
    report_prov = pd.merge(report_prov, nino_df, on='CropYear', how='left')
    avg_ndvi = report_prov['NDVI_seasonal'].mean()
    avg_precip = report_prov['Total_Precip'].mean()
    
    report_prov['NDVI Anomaly (%)'] = ((report_prov['NDVI_seasonal'] - avg_ndvi) / avg_ndvi * 100)
    report_prov['Rain Anomaly (%)'] = ((report_prov['Total_Precip'] - avg_precip) / avg_precip * 100)
    report_prov['Season'] = (report_prov['CropYear'] - 1).astype(str) + "-" + report_prov['CropYear'].astype(str)
    
    report_df = report_prov[['Season', 'NINO34', 'NDVI_seasonal', 'NDVI Anomaly (%)', 'Total_Precip', 'Rain Anomaly (%)']].copy()
    report_df.columns = ['Season', 'NINO34 Index', 'Mean NDVI', 'NDVI Anomaly (%)', 'Total Rain (mm)', 'Rain Anomaly (%)']

    def get_impact_label(row):
        if row['NINO34 Index'] >= 1.0 and row['NDVI Anomaly (%)'] < -5: return "⚠️ Severe El Niño Drought"
        elif row['NINO34 Index'] >= 0.5: return "🔸 Mild El Niño Impact"
        elif row['NINO34 Index'] <= -0.5: return "🔹 La Niña (Wet)"
        else: return "✅ Neutral/Normal"

    report_df['Climate Status'] = report_df.apply(get_impact_label, axis=1)
    
    # Save HTML
    html_out = report_df.to_html()
    with open(OUTPUT_DIR / "Southern_Province_Validation_Report_Basic.html", "w", encoding="utf-8") as f:
        f.write(html_out)

save_styled_report(df_seasonal_agg, nino_seasonal)

print("--- PROCESS COMPLETE ---")

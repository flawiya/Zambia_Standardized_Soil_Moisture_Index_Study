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



#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# 1. CONFIGURATION AND BASE PATHS
# =============================================================================
# Using the specific path you provided
BASE_PATH = r"C:\Users\FlawiyaShirishMore\Downloads\Africa-Drought-Study\data"
OUTPUT_PATH = os.path.join(BASE_PATH, "Analysis_Outputs")
MASTER_FILE_PATH = os.path.join(BASE_PATH, "master_final_processed.csv")

os.makedirs(OUTPUT_PATH, exist_ok=True)

TARGET_DISTRICTS = [
    'Chirundu', 'Choma', 'Gwembe', 'Kalomo', 'Kazungula', 
    'Mazabuka', 'Monze', 'Namwala', 'Pemba', 'Siavonga', 
    'Sinazongwe', 'Zimba'
]

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_crop_year(row):
    """Aligns Nov-Dec data points to the following calendar harvest year."""
    return row['date'].year + 1 if row['date'].month in [11, 12] else row['date'].year

def calculate_sheffield_baseline(df, col, window=31):
    """Calculates a stable historical baseline across all years."""
    daily_stats = df.groupby(['district', 'day_of_year'])[col].agg(['mean', 'std']).reset_index()
    results = []
    for district, group in daily_stats.groupby('district'):
        group = group.sort_values('day_of_year')
        # Triple data to handle circular wrap-around
        triple = pd.concat([group] * 3).reset_index(drop=True)
        triple['mu_stable'] = triple['mean'].rolling(window=window, center=True).mean()
        triple['std_stable'] = triple['std'].rolling(window=window, center=True).mean()
        middle = triple.iloc[len(group):2*len(group)].copy()
        results.append(middle)
    stable_baseline = pd.concat(results, ignore_index=True)
    return stable_baseline[['district', 'day_of_year', 'mu_stable', 'std_stable']]

def calculate_neg_integral(series, threshold=-0.8):
    stress = series[series < threshold]
    if stress.empty: return 0
    return (stress - threshold).sum()

# =============================================================================
# 3. DATA LOADING (SEARCH FOR MASTER OR MERGE RAW)
# =============================================================================

if os.path.exists(MASTER_FILE_PATH):
    print(f"✅ Found existing Master File: {MASTER_FILE_PATH}")
    print("🚀 Loading pre-processed data...")
    master = pd.read_csv(MASTER_FILE_PATH)
    master['date'] = pd.to_datetime(master['date'])
else:
    print("📂 Master file not found. Starting raw data merge and index calculation...")
    
    # Define file names
    files = {
        "soil": "master_southern_province_soil-moisture-layer2.csv",
        "ndvi": "master_southern_province_ndvi.csv",
        "lst": "master_southern_province_lst.csv",
        "climate": "climate_merged.csv",
        "spei": "master_southern_province_data.csv"
    }

    # Verify all raw files exist before starting
    for f in files.values():
        if not os.path.exists(os.path.join(BASE_PATH, f)):
            raise FileNotFoundError(f"Missing required file: {f} in {BASE_PATH}")

    # Load Raw Data
    df_soil = pd.read_csv(os.path.join(BASE_PATH, files["soil"]))
    df_ndvi = pd.read_csv(os.path.join(BASE_PATH, files["ndvi"]))
    df_lst = pd.read_csv(os.path.join(BASE_PATH, files["lst"]))
    df_climate = pd.read_csv(os.path.join(BASE_PATH, files["climate"]))
    df_spei_data = pd.read_csv(os.path.join(BASE_PATH, files["spei"]))

    for df in [df_soil, df_ndvi, df_lst, df_climate, df_spei_data]:
        df['date'] = pd.to_datetime(df['date'])

    print("🔗 Merging datasets...")
    # Base merge
    master = pd.merge(df_soil[['date', 'district', 'soil_moisture_layer2']], 
                     df_spei_data[['date', 'district', 'precip_mm', 'pet_mm']], 
                     on=['date', 'district'], how='inner')

    # Add LST
    df_lst['lst_celsius'] = df_lst['lst_celsius'].replace(-999.0, np.nan)
    master = pd.merge(master, df_lst[['date', 'district', 'lst_celsius']], on=['date', 'district'], how='left')
    
    # Add NDVI and Climate
    master = pd.merge(master, df_ndvi[['date', 'district', 'ndvi']], on=['date', 'district'], how='left')
    master = pd.merge(master, df_climate[['date', 'NINO34']], on='date', how='left')

    master['day_of_year'] = master['date'].dt.dayofyear
    master['crop_year'] = master.apply(get_crop_year, axis=1)

    # --- Calculating Indices ---
    print("📉 Calculating Standardized Indices (SSI, SPEI3, VHI)...")
    
    # 1. SSI (Soil Moisture)
    baseline_stats_ssi = calculate_sheffield_baseline(master, 'soil_moisture_layer2')
    master = master.merge(baseline_stats_ssi, on=['district', 'day_of_year'], how='left')
    master['SSI'] = (master['soil_moisture_layer2'] - master['mu_stable']) / (master['std_stable'] + 1e-6)

    # 2. SPEI-3 (90-Day Water Balance)
    master['precip_mm'] = master['precip_mm'].clip(lower=0)
    master['D'] = master['precip_mm'] - master['pet_mm'].abs()
    master['D_90'] = master.groupby('district')['D'].transform(lambda x: x.rolling(window=90, min_periods=28).sum())
    
    baseline_stats_spei = calculate_sheffield_baseline(master, 'D_90')
    baseline_stats_spei = baseline_stats_spei.rename(columns={'mu_stable': 'mu_d90', 'std_stable': 'std_d90'})
    master = master.merge(baseline_stats_spei, on=['district', 'day_of_year'], how='left')
    master['SPEI3'] = (master['D_90'] - master['mu_d90']) / (master['std_d90'] + 1e-6)

    # 3. VHI (Vegetation Health)
    master['ndvi'] = master.groupby('district')['ndvi'].transform(lambda x: x.interpolate().ffill().bfill())
    master['lst_celsius'] = master.groupby('district')['lst_celsius'].transform(lambda x: x.interpolate().ffill().bfill())
    master['VCI_z'] = master.groupby(['district', 'day_of_year'])['ndvi'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    master['TCI_z'] = master.groupby(['district', 'day_of_year'])['lst_celsius'].transform(lambda x: (x.mean() - x) / (x.std() + 1e-6))
    master['VHI'] = (0.5 * master['VCI_z']) + (0.5 * master['TCI_z'])

    # Final Smoothing (7-day window)
    for col in ['SSI', 'SPEI3', 'VHI']:
        master[col] = master.groupby('district')[col].transform(lambda x: x.rolling(window=7, center=True, min_periods=1).mean())

    # Save for future use
    print(f"💾 Saving processed data to {MASTER_FILE_PATH}...")
    master.to_csv(MASTER_FILE_PATH, index=False)

# =============================================================================
# 4. AGGREGATION & VISUALIZATION
# =============================================================================

# Filter for Growing Season (Nov-Apr)
master_crop = master[master['date'].dt.month.isin([11, 12, 1, 2, 3, 4])].copy()

print("📋 Generating Seasonal Aggregation Matrix for 2024...")
seasonal_matrix = master_crop[master_crop['crop_year'] == 2024].groupby('district').agg(
    SSI_Mean=('SSI', 'mean'),
    SSI_Min=('SSI', 'min'),
    SSI_Neg_Integral=('SSI', calculate_neg_integral),
    VHI_Mean=('VHI', 'mean'),
    NDVI_Max=('ndvi', 'max'),
    SPEI3_Min=('SPEI3', 'min'),
    NINO34_Avg=('NINO34', 'mean')
).reset_index().sort_values(by='SSI_Neg_Integral')

seasonal_matrix.to_csv(os.path.join(OUTPUT_PATH, "Seasonal_Drought_Matrix_2024.csv"), index=False)

print("📊 Rendering Spatiotemporal Drought Cascade (2024)...")
df_2024 = master_crop[master_crop['crop_year'] == 2024].copy()
df_prov_avg = df_2024.groupby('date').mean(numeric_only=True).reset_index()

fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("1. Meteorological: Daily Rainfall vs. Evaporation (PET)", 
                                    "2. Hydrological: Daily SSI (Julian-Day Baseline)", 
                                    "3. Cumulative: SPEI-3 (90-Day Anomaly)", 
                                    "4. Biological: Vegetation Health Index (VHI)"))

fig.add_trace(go.Bar(x=df_prov_avg['date'], y=df_prov_avg['precip_mm'], name="Avg Rain", marker_color='dodgerblue'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg['pet_mm'].abs(), name="Avg PET", line=dict(color='crimson', width=2)), row=1, col=1)

for r, col in [(2, 'SSI'), (3, 'SPEI3')]:
    for d in TARGET_DISTRICTS:
        if d in df_2024['district'].values:
            df_d = df_2024[df_2024['district'] == d]
            fig.add_trace(go.Scatter(x=df_d['date'], y=df_d[col], line=dict(color='rgba(150,150,150,0.12)', width=1), showlegend=False), row=r, col=1)
    
    fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg[col].clip(lower=0), fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)', line_color='blue', name=f"Prov {col} (Wet)"), row=r, col=1)
    fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg[col].clip(upper=0), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)', line_color='red', name=f"Prov {col} (Dry)"), row=r, col=1)

fig.add_trace(go.Scatter(x=df_prov_avg['date'], y=df_prov_avg['VHI'], line=dict(color='darkgreen', width=3), name="Avg VHI"), row=4, col=1)

for r in [2, 3]:
    fig.add_hline(y=-1.2, line_dash="dash", line_color="darkred", annotation_text="Extreme Drought", row=r, col=1)

fig.update_layout(height=1200, title_text="Southern Province: Spatiotemporal Drought Analysis (2024)", template="plotly_white", hovermode="x unified")
fig.write_html(os.path.join(OUTPUT_PATH, "Drought_Cascade_2024.html"))

print("🌡️ Generating Historical SSI Matrix (2001-2025)...")
heatmap_df = master_crop[master_crop['district'].isin(TARGET_DISTRICTS)].copy()
matrix_data = heatmap_df.groupby(['crop_year', 'district'])['SSI'].mean().reset_index()
heatmap_pivot = matrix_data.pivot(index='crop_year', columns='district', values='SSI')
heatmap_pivot = heatmap_pivot[heatmap_pivot.index >= 2001]

plt.figure(figsize=(16, 10), dpi=300)
sns.heatmap(heatmap_pivot, cmap='RdBu', center=0, vmin=-3, vmax=3, linewidths=0.5, linecolor='white', cbar_kws={'label': 'Seasonal Mean SSI'})
plt.title("Southern Province: Historical Drought Matrix (Nov-Apr)", fontsize=16, fontweight='bold')
plt.savefig(os.path.join(OUTPUT_PATH, "Historical_SSI_Matrix.png"), bbox_inches='tight')

print(f"\n✨ DONE! Results saved in: {OUTPUT_PATH}")

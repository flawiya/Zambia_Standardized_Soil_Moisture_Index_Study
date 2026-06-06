#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import geopandas as gpd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
BASE_DATA_PATH = r"./data"
SHP_PATH = os.path.join(BASE_DATA_PATH, "africa_agricultural_domain_2019", "africa_agricultural_domain_2019.shp")
GEOGLAM_PATH = os.path.join(BASE_DATA_PATH, "GEOGLAM_CM4EW_Calendars_V1.4", "GEOGLAM_CM4EW_Calendars_V1.4.shp")
ERA5_CSV_PATH = os.path.join(BASE_DATA_PATH, "Africa_Agri_districts_ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv")

# TARGET OUTPUT DIRECTORY
OUTPUT_DIR = r"C:\Users\FlawiyaShirishMore\Downloads\Africa-Drought-Study\outputs\Zambia_Gamma_SSI_Optimization_Study"

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✅ Created Output Directory: {OUTPUT_DIR}")

# =============================================================================
# 2. DATA LOADING & ZAMBIA FILTERING
# =============================================================================
print("Step 1: Loading Shapefiles and Filtering for Zambia...")
gdf_all_districts = gpd.read_file(SHP_PATH)
gdf_geoglam = gpd.read_file(GEOGLAM_PATH)

zambia_districts = gdf_all_districts[gdf_all_districts['ISO3'] == 'ZMB'].copy()
zambia_districts['ADM_NAME'] = zambia_districts['ADM_NAME'].astype(str).str.strip().str.upper()

zambia_crops = gdf_geoglam[gdf_geoglam['country'] == 'Zambia'].copy()

# Identify crops sharing the Maize 1 window
try:
    maize_ref = zambia_crops[zambia_crops['crop'] == 'Maize 1'].iloc[0]
    p_start, h_end = maize_ref['planting'], maize_ref['endofseaso']
    mask = (zambia_crops['planting'].between(p_start - 30, p_start + 30)) | \
           (zambia_crops['endofseaso'].between(h_end - 30, h_end + 30))
    same_calendar_crops = zambia_crops[mask]['crop'].unique()
    print(f"   Maize 1 Window: Planting {p_start}, Harvest {h_end}")
    print(f"   Similar Crops: {list(same_calendar_crops)}")
except IndexError:
    print("❌ Error: 'Maize 1' not found in Zambia calendar.")

# =============================================================================
# 3. ERA5 PROCESSING & SSI CALCULATION
# =============================================================================
print("\nStep 2: Processing ERA5 Soil Moisture for Zambia Districts...")
df_era5 = pd.read_csv(ERA5_CSV_PATH)
df_era5['feature_id'] = df_era5['feature_id'].astype(str).str.strip().str.upper()

zmb_names = zambia_districts['ADM_NAME'].unique()
df_zmb = df_era5[df_era5['feature_id'].isin(zmb_names)].copy()

# Daily Climatology (Z-score SSI)
climatology = df_zmb.groupby(['feature_id', 'doy'])['volumetric_soil_water_layer_2'].agg(['mean', 'std']).reset_index()
df_zmb = df_zmb.merge(climatology, on=['feature_id', 'doy'], how='left')
df_zmb['SSI'] = (df_zmb['volumetric_soil_water_layer_2'] - df_zmb['mean']) / (df_zmb['std'] + 1e-6)

# Actuarial alignment (Crop Year logic)
df_zmb['crop_year'] = np.where(df_zmb['month'] >= 11, df_zmb['year'] + 1, df_zmb['year'])
df_zmb_risk = df_zmb[df_zmb['month'].between(1, 4)].copy()

# =============================================================================
# 4. ANNUAL AGGREGATION & SENSITIVITY PREP
# =============================================================================
print("Step 3: Aggregating into Annual Metrics...")
SSI_THRESHOLD = -1.0
df_annual = df_zmb_risk.groupby(['feature_id', 'crop_year']).apply(
    lambda x: (x['SSI'] <= SSI_THRESHOLD).sum()
).reset_index(name='Extreme_Days_Count')

metadata = zambia_districts[['ADM_NAME', 'crop_pct']].rename(columns={'ADM_NAME': 'feature_id'})
df_final = df_annual.merge(metadata, on='feature_id', how='inner')
df_pivot = df_final.pivot_table(index='crop_year', columns='feature_id', values='Extreme_Days_Count')

# =============================================================================
# 5. ANALYSIS: GROUPING SYNCHRONICITY (SENSITIVITY)
# =============================================================================
def get_group_synchronicity(pivot_table, threshold_districts):
    if len(threshold_districts) < 2: return 0
    corr_matrix = pivot_table[threshold_districts].corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return upper_tri.stack().mean()

print("Step 4: Running Synchronicity Analysis...")
sync_results = []
thresholds = [0, 10, 20, 30]

for t in thresholds:
    t_districts = df_final[df_final['crop_pct'] >= t]['feature_id'].unique()
    avg_corr = get_group_synchronicity(df_pivot, t_districts)
    sync_results.append({'Threshold': f"{t}%", 'Avg_Correlation': avg_corr, 'Count': len(t_districts)})

df_sync = pd.DataFrame(sync_results)
df_sync.to_csv(os.path.join(OUTPUT_DIR, "Zambia_CropArea_Sensitivity_Report.csv"), index=False)

# Visualizing Synchronicity
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(data=df_sync, x='Threshold', y='Count', palette='viridis')
plt.title('District Pool Sensitivity')
plt.subplot(1, 2, 2)
sns.lineplot(data=df_sync, x='Threshold', y='Avg_Correlation', marker='o', color='red')
plt.title('Temporal Grouping Outcome (r)')
plt.savefig(os.path.join(OUTPUT_DIR, "Zambia_Grouping_Outcome_Comparison.png"), dpi=300)
plt.close()

# =============================================================================
# 6. RISK CLUSTERING & MAPPING
# =============================================================================
def get_risk_clusters(pivot_table, threshold_districts):
    data = pivot_table[threshold_districts].T.fillna(pivot_table.mean())
    scaled = StandardScaler().fit_transform(data)
    clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    return pd.DataFrame({'feature_id': threshold_districts, 'Cluster': clusters})

print("Step 5: Generating Spatial Risk Clusters...")
clusters_0 = get_risk_clusters(df_pivot, df_final[df_final['crop_pct'] >= 0]['feature_id'].unique())
clusters_10 = get_risk_clusters(df_pivot, df_final[df_final['crop_pct'] >= 10]['feature_id'].unique())

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
zambia_districts.merge(clusters_0, left_on='ADM_NAME', right_on='feature_id', how='left').plot(
    column='Cluster', ax=axes[0], cmap='Set3', legend=True, edgecolor='black', missing_kwds={'color': 'lightgrey'})
axes[0].set_title("ALL Districts (r=0.50)")
zambia_districts.merge(clusters_10, left_on='ADM_NAME', right_on='feature_id', how='left').plot(
    column='Cluster', ax=axes[1], cmap='Set3', legend=True, edgecolor='black', missing_kwds={'color': 'lightgrey'})
axes[1].set_title("10% Crop Area Threshold (r=0.67)")
plt.savefig(os.path.join(OUTPUT_DIR, "Zambia_Spatial_Cluster_Comparison.png"), dpi=300)
plt.close()

# =============================================================================
# 7. HISTORICAL DROUGHT FREQUENCY (TIME SERIES)
# =============================================================================
print("Step 6: Generating Historical Time Series...")
time_series_data = []
for t in thresholds:
    valid_districts = df_final[df_final['crop_pct'] >= t]['feature_id'].unique()
    annual_avg = df_final[df_final['feature_id'].isin(valid_districts)].groupby('crop_year')['Extreme_Days_Count'].mean().reset_index()
    annual_avg['Threshold'] = f"{t}% Crop Area"
    time_series_data.append(annual_avg)

df_ts_compare = pd.concat(time_series_data)
plt.figure(figsize=(15, 7))
sns.lineplot(data=df_ts_compare, x='crop_year', y='Extreme_Days_Count', hue='Threshold', marker='o')
plt.axhline(y=40, color='black', linestyle=':')
plt.title("Zambia Historical Drought Frequency: Threshold Sensitivity")
plt.savefig(os.path.join(OUTPUT_DIR, "Zambia_Historical_Drought_Frequency.png"), dpi=300)
plt.close()

# =============================================================================
# 8. DECISION MATRIX & FINAL OUTPUTS
# =============================================================================
print("Step 7: Finalizing Decision Matrix...")
df_decision = df_sync.copy()
df_decision['Coverage_Loss_%'] = ((70 - df_decision['Count']) / 70 * 100).round(1)
df_decision['Statistical_Reliability'] = df_decision['Avg_Correlation'].apply(
    lambda x: 'High' if x > 0.65 else ('Moderate' if x > 0.55 else 'Low')
)
df_decision.to_csv(os.path.join(OUTPUT_DIR, "Zambia_Threshold_Decision_Matrix.csv"), index=False)

# Ground Truth 2024
plt.figure(figsize=(10, 6))
sns.regplot(data=df_final[df_final['crop_year'] == 2024], x='crop_pct', y='Extreme_Days_Count', color='blue')
plt.title("Sensitivity Analysis: Crop Density vs Drought (2024)")
plt.savefig(os.path.join(OUTPUT_DIR, "Zambia_2024_Ground_Truth_Regression.png"), dpi=300)
plt.close()

print(f"\n🚀 ALL ANALYSIS COMPLETE. Check results in: {OUTPUT_DIR}")

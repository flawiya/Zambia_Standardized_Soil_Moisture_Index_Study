import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json
import os
from dash import Dash, dcc, html, Input, Output
from matplotlib.colors import Normalize

# =========================================================
# 1. CONFIGURATION & PATHS
# =========================================================
data_path = r"data\ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"
shape_path = r"data\Zambia_agri_districts.shp"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

SSI_THRESHOLD = -1.5  # Standardized to "Severe Drought"

print("📂 Loading ERA5-Land Data...")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

# =========================================================
# 2. CALCULATE STANDARDIZED SOIL MOISTURE INDEX (SSI)
# =========================================================
print("📉 Calculating Daily SSI (Julian Day Climatology)...")
climatology = (df.groupby(['feature_id', 'doy'])['volumetric_soil_water_layer_2']
               .agg(Historical_Mean='mean', Historical_Std='std').reset_index())

df = df.merge(climatology, on=['feature_id', 'doy'], how='left')
df['SSI'] = (df['volumetric_soil_water_layer_2'] - df['Historical_Mean']) / (df['Historical_Std'] + 1e-6)

# =========================================================
# 3. SEASONAL AGGREGATION (JAN-APR RISK PERIOD)
# =========================================================
print("🗓️ Aggregating Risk Period: January to April...")
df_risk_period = df[(df['month'] >= 1) & (df['month'] <= 4)].copy()

# Metric 1: Sum of SSI (Total Intensity)
ssi_sum = df_risk_period.groupby(['feature_id', 'year'])['SSI'].sum().reset_index(name='Sum_SSI')

# Metric 2: Count of Extreme Drought Days
df_risk_period['Is_Extreme_Drought'] = (df_risk_period['SSI'] <= SSI_THRESHOLD).astype(int)
drought_days_count = df_risk_period.groupby(['feature_id', 'year'])['Is_Extreme_Drought'].sum().reset_index(name='Count_Extreme_Drought_Days')

# Combine and Export CSV
df_yearly_summary = pd.merge(ssi_sum, drought_days_count, on=['feature_id', 'year'])
df_yearly_summary.to_csv(os.path.join(output_dir, 'Zambia_National_SSI_Summary.csv'), index=False)

# =========================================================
# 4. NATIONAL VS DISTRICT TREND PLOTTING
# =========================================================
print("📊 Generating Time Series Trends...")
df_nat_avg = df_yearly_summary.groupby('year', as_index=False)[['Sum_SSI', 'Count_Extreme_Drought_Days']].mean()
df_nat_avg['feature_id'] = 'NATIONAL AVERAGE'
df_plot = pd.concat([df_yearly_summary, df_nat_avg], ignore_index=True)

fig1 = px.line(df_plot, x='year', y='Sum_SSI', color='feature_id',
               title='Zambia: National Moisture Intensity (Jan-Apr) 2000-2025',
               labels={'Sum_SSI': 'SSI Intensity (Lower = Drier)'})

# Professional Styling: Grey out districts, highlight National Average
for trace in fig1.data:
    if trace.name == 'NATIONAL AVERAGE':
        trace.line.update(color='red', width=5)
        trace.update(opacity=1.0)
    else:
        trace.line.update(color='lightgray', width=1)
        trace.update(opacity=0.4)

fig1.add_vline(x=2024, line_dash="dash", line_color="black", annotation_text="2024 Systemic Drought")
fig1.update_layout(template="plotly_white", showlegend=False)
fig1.write_html(os.path.join(output_dir, "Zambia_SSI_Intensity_Trends.html"))

# =========================================================
# 5. GEOSPATIAL VISUALIZATION (STATIC COMPARISON)
# =========================================================
print("🗺️ Generating Comparative Maps (2023 vs 2024)...")
gdf = gpd.read_file(shape_path)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)

gdf['ADM_NAME'] = gdf['ADM_NAME'].str.strip().str.upper()
df_yearly_summary['feature_id'] = df_yearly_summary['feature_id'].str.strip().str.upper()

df_23_24 = df_yearly_summary[df_yearly_summary['year'].isin([2023, 2024])]
gdf_map = gdf.merge(df_23_24, left_on='ADM_NAME', right_on='feature_id', how='left')

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
cmap = plt.cm.RdYlBu  # Moisture scale
norm = Normalize(vmin=df_yearly_summary['Sum_SSI'].min(), vmax=df_yearly_summary['Sum_SSI'].max())

for ax, yr in zip(axes, [2023, 2024]):
    gdf_yr = gdf_map[gdf_map['year'] == yr]
    gdf_yr.plot(column='Sum_SSI', cmap=cmap, ax=ax, norm=norm, edgecolor='lightgrey', linewidth=0.1)
    ax.set_title(f"Zambia SSI Intensity: {yr}", fontsize=15, fontweight='bold')
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Zambia_2023_2024_National_Comparison.png"), dpi=300)

# =========================================================
# 6. DASHBOARD (INTERACTIVE DETAILS-ON-DEMAND)
# =========================================================
app = Dash(__name__)

map_2024_data = gdf.merge(df_yearly_summary[df_yearly_summary['year'] == 2024], 
                          left_on='ADM_NAME', right_on='feature_id')
zambia_geojson = json.loads(gdf.to_json())

app.layout = html.Div(style={'font-family': 'Segoe UI', 'padding': '20px'}, children=[
    html.H1("Zambia Agricultural Drought: National Risk Dashboard"),
    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        html.Div(style={'flex': '1'}, children=[
            dcc.Graph(id='zambia-map', figure=px.choropleth(
                map_2024_data, geojson=zambia_geojson, locations='ADM_NAME',
                featureidkey="properties.ADM_NAME",
                color='Count_Extreme_Drought_Days', color_continuous_scale='Reds',
                title="2024: Count of Severe Drought Days (SSI ≤ -1.5)"
            ).update_geos(fitbounds="locations", visible=False)
             .update_traces(marker_line_width=0.1, marker_line_color='lightgrey'))
        ]),
        html.Div(style={'flex': '1'}, children=[
            dcc.Graph(id='timeseries-chart')
        ])
    ])
])

@app.callback(Output('timeseries-chart', 'figure'), Input('zambia-map', 'clickData'))
def update_timeseries(clickData):
    district_name = clickData['points'][0]['location'] if clickData else 'GWEMBE'
    district_data = df_yearly_summary[df_yearly_summary['feature_id'] == district_name].sort_values('year')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=district_data['year'], y=district_data['Count_Extreme_Drought_Days'], 
                             mode='lines+markers', name=district_name, line=dict(color='firebrick', width=3)))
    fig.update_layout(title=f"25-Year History: {district_name}", template='plotly_white')
    return fig

# =========================================================
# 7. ANIMATED SPATIOTEMPORAL MAPPING
# =========================================================
print("🎬 Generating Animated Spatiotemporal Pulse...")
map_df_sorted = df_yearly_summary.sort_values(by=['year'])

fig_anim = px.choropleth(
    map_df_sorted, geojson=zambia_geojson, locations='feature_id',
    featureidkey="properties.ADM_NAME",
    color='Count_Extreme_Drought_Days', animation_frame='year',
    color_continuous_scale='YlOrRd', range_color=[0, 100],
    title='Zambia: Annual Evolution of Severe Drought Days (2000-2026)',
)

fig_anim.update_traces(marker_line_width=0.1, marker_line_color='lightgrey')
fig_anim.update_geos(fitbounds="locations", visible=False)
fig_anim.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig_anim.write_html(os.path.join(output_dir, "Zambia_National_Drought_Animation.html"))

print(f"✅ ALL TASKS COMPLETE. Results saved in: {output_dir}")

if __name__ == '__main__':
    app.run(debug=False, port=8050)   # Correct command
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

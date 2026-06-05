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
import matplotlib as mpl

# =========================================================
# 1. LOAD DATA & FORMATTING
# =========================================================
# Paths based on your setup
data_path = r"data\ERA5_LAND_DAILY_AGGR_2000_2026_timeseries.csv"
shape_path = r"data\Zambia_agri_districts.shp"

print("📂 Loading ERA5-Land Data...")
df = pd.read_csv(data_path)

# Ensure date is datetime
df['date'] = pd.to_datetime(df['date'])

# =========================================================
# 2. CALCULATE STANDARDIZED SOIL MOISTURE INDEX (SSI)
# =========================================================
# Establishing a daily climatological baseline (Day of Year) to capture flash droughts
print("📉 Calculating Daily SSI (Julian Day Baseline)...")
climatology = (df.groupby(['feature_id', 'doy'])['volumetric_soil_water_layer_2']
               .agg(Historical_Mean='mean', Historical_Std='std').reset_index())

df = df.merge(climatology, on=['feature_id', 'doy'], how='left')

# SSI Formula: (Current - Mean) / StdDev
df['SSI'] = (df['volumetric_soil_water_layer_2'] - df['Historical_Mean']) / (df['Historical_Std'] + 1e-6)

# =========================================================
# 3. SEASONAL AGGREGATION (JAN-APR RISK PERIOD)
# =========================================================
print("🗓️ Aggregating Risk Period: January to April...")
df_risk_period = df[(df['month'] >= 1) & (df['month'] <= 4)].copy()

# Metric 1: Sum of SSI (Total Intensity)
ssi_sum = df_risk_period.groupby(['feature_id', 'year'])['SSI'].sum().reset_index(name='Sum_SSI')

# Metric 2: Count of Extreme Drought Days (SSI <= -1.0)
df_risk_period['Is_Extreme_Drought'] = (df_risk_period['SSI'] <= -1.0).astype(int)
drought_days_count = df_risk_period.groupby(['feature_id', 'year'])['Is_Extreme_Drought'].sum().reset_index(name='Count_Extreme_Drought_Days')

# Combine Metrics
df_yearly_summary = pd.merge(ssi_sum, drought_days_count, on=['feature_id', 'year'])
df_yearly_summary.to_csv('Zambia_District_SSI_Jan-Apr_2000-2024.csv', index=False)

# =========================================================
# 4. INTERACTIVE PLOTTING (NATIONAL VS DISTRICT)
# =========================================================
print("📊 Generating Interactive Time Series Plots...")
df_nat_avg = df_yearly_summary.groupby('year', as_index=False)[['Sum_SSI', 'Count_Extreme_Drought_Days']].mean()
df_nat_avg['feature_id'] = 'National Average'
df_plot = pd.concat([df_yearly_summary, df_nat_avg], ignore_index=True)

# Plot: Sum SSI
fig1 = px.line(df_plot, x='year', y='Sum_SSI', color='feature_id',
               title='Zambia: Sum of Daily SSI (Jan-Apr) 2000-2025',
               labels={'Sum_SSI': 'SSI Intensity (Negative = Drier)'})

for trace in fig1.data:
    if trace.name == 'National Average':
        trace.line.update(color='red', width=4)
    else:
        # We update the trace's overall opacity and the line's color/width separately
        trace.update(opacity=0.5) 
        trace.line.update(color='lightgray', width=1.5)

fig1.add_vline(x=2024, line_dash="dash", line_color="black", annotation_text="2024 Drought")
fig1.write_html("Zambia_SSI_Intensity_History.html")

# =========================================================
# 5. GEOSPATIAL VISUALIZATION (MAPS)
# =========================================================
print("🗺️ Processing Shapefiles and Static Maps...")
gdf = gpd.read_file(shape_path)
gdf['ADM_NAME'] = gdf['ADM_NAME'].str.strip().str.upper()
df_yearly_summary['feature_id'] = df_yearly_summary['feature_id'].str.strip().str.upper()

# Comparative Map: 2023 vs 2024
df_23_24 = df_yearly_summary[df_yearly_summary['year'].isin([2023, 2024])]
gdf_map = gdf.merge(df_23_24, left_on='ADM_NAME', right_on='feature_id', how='left')

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
cmap = plt.cm.RdYlBu_r
norm = Normalize(vmin=df_yearly_summary['Sum_SSI'].min(), vmax=df_yearly_summary['Sum_SSI'].max())

for ax, yr in zip(axes, [2023, 2024]):
    gdf_yr = gdf_map[gdf_map['year'] == yr]
    gdf_yr.plot(column='Sum_SSI', cmap=cmap, ax=ax, norm=norm, edgecolor='black', linewidth=0.3)
    ax.set_title(f"Zambia SSI Intensity: {yr}", fontweight='bold')
    ax.set_axis_off()

plt.tight_layout()
plt.savefig("Zambia_2023_2024_Comparison.png", dpi=300)

# =========================================================
# 6. DASHBOARD (INTERACTIVE DETAILS-ON-DEMAND)
# =========================================================
print("🖥️ Starting Interactive Dashboard (Local Server)...")
app = Dash(__name__)

map_2024 = gdf.merge(df_yearly_summary[df_yearly_summary['year'] == 2024], 
                     left_on='ADM_NAME', right_on='feature_id')

app.layout = html.Div([
    html.H2("Zambia Agricultural Drought Dashboard: Extreme Dry Days"),
    html.Div([
        dcc.Graph(id='zambia-map', figure=px.choropleth(
            map_2024, geojson=map_2024.geometry, locations=map_2024.index,
            color='Count_Extreme_Drought_Days', color_continuous_scale='YlOrRd'
        ).update_geos(fitbounds="locations", visible=False).update_layout(clickmode='event+select'),
        style={'width': '48%', 'display': 'inline-block'}),
        
        dcc.Graph(id='timeseries-chart', style={'width': '48%', 'display': 'inline-block'})
    ])
])

@app.callback(Output('timeseries-chart', 'figure'), Input('zambia-map', 'clickData'))
def update_timeseries(clickData):
    district_name = clickData['points'][0]['location'] if clickData else 'CHIBOMBO'
    district_data = df_yearly_summary[df_yearly_summary['feature_id'] == district_name].sort_values('year')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=district_data['year'], y=district_data['Count_Extreme_Drought_Days'], 
                             mode='lines+markers', name=district_name, line=dict(color='firebrick')))
    fig.update_layout(title=f"25-Year Trend: {district_name}", template='plotly_white')
    return fig

# Note: In a standard .py file, the dashboard starts here.
# app.run(port=8050)

# =========================================================
# 7. ANIMATED SPATIOTEMPORAL MAPPING (FIXED)
# =========================================================
print("🎬 Generating Animated Drought Pulse...")

# Simplify the map so the file isn't too heavy
gdf_simple = gdf.copy()
gdf_simple['geometry'] = gdf_simple['geometry'].simplify(tolerance=0.001)

# Convert to GeoJSON format
zambia_geojson = json.loads(gdf_simple.to_json())

# Sort data by year so the animation plays in order
map_df_sorted = df_yearly_summary.sort_values(by=['year'])

fig_anim = px.choropleth(
    map_df_sorted, 
    geojson=zambia_geojson, 
    locations='feature_id',        # The column in your CSV
    featureidkey="properties.ADM_NAME",  # The matching property in your Shapefile
    color='Count_Extreme_Drought_Days', 
    animation_frame='year',
    color_continuous_scale='YlOrRd', 
    range_color=[0, df_yearly_summary['Count_Extreme_Drought_Days'].max()],
    title='Zambia: Annual Pulse of Extreme Drought Days (Jan-Apr)',
    labels={'Count_Extreme_Drought_Days': 'Dry Days'}
)

# This centers the map on Zambia automatically
fig_anim.update_geos(fitbounds="locations", visible=False)

fig_anim.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig_anim.write_html("Zambia_Drought_Frequency_Animation.html")

print("✅ Map should now be visible in Zambia_Drought_Frequency_Animation.html")

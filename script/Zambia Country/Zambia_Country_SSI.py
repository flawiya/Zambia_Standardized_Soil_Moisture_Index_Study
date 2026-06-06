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

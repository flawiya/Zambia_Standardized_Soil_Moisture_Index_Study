import ee
import time

try:
    ee.Initialize()
except Exception as e:
    print("Initialization failed. Run 'earthengine authenticate'.")
    raise

# ============================================
# CONFIGURATION
# ============================================
ASSET_PATH = 'projects/************/assets/Zambia_Administrative_Boundaries_Districts'
EXPORT_FOLDER = 'zambia-all-district-soil'
# Updated Band Name
BAND_NAME = 'volumetric_soil_water_layer_2'

districts = ee.FeatureCollection(ASSET_PATH)

# ============================================
# EXTRACTION FUNCTION
# ============================================
def extract_daily_features(image):
    """Extract district averages for ERA5-Land Daily."""
    # 1. Select the band
    selected_image = image.select(BAND_NAME)
    
    date_str = image.date().format('YYYY-MM-dd')
    year = image.date().get('year')
    month = image.date().get('month')
    day = image.date().get('day')
    
    def map_district(district):
        # ERA5-Land scale is ~9km (9000m)
        stats = selected_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=district.geometry(),
            scale=9000,          
            maxPixels=1e9
        )
        
        # ERA5-Land values are already physical units (m^3/m^3), no need to multiply
        val = ee.Number(stats.get(BAND_NAME))
        
        return ee.Feature(None, {
            'district': district.get('DISTRICT'), 
            'year': year,
            'month': month,
            'day': day,
            'date': date_str,
            'soil_moisture_layer2': ee.Algorithms.If(val, val, -999)
        })
    
    return districts.map(map_district)

# ============================================
# EXECUTION
# ============================================
years = range(2000, 2026)

for year in years:
    print(f"Processing year {year}...")
    
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    # ERA5-Land Daily Aggregated collection
    collection = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                  .filterDate(start, end))
    
    count = collection.size().getInfo()
    if count > 0:
        # Extract features
        yearly_features = collection.map(extract_daily_features).flatten()
        
        # Export
        task = ee.batch.Export.table.toDrive(
            collection=yearly_features,
            description=f'SoilMoisture-zambia-all-districts-{year}',
            folder=EXPORT_FOLDER,
            fileNamePrefix=f'soil_moisture_{year}',
            fileFormat='CSV'
        )
        task.start()
        print(f"  → Task started: {task.id}")
    else:
        print(f"  → No data found for {year}")
    
    time.sleep(2)

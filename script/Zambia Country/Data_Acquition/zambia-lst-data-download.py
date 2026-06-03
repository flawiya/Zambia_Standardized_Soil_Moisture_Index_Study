
#projects/**********/Zambia_Administrative_Boundaries_Districts

#!/usr/bin/env python3
"""
Export daily ERA5-Land data for Zambia's districts as one CSV per year (2000–2025).
Run: python zambia_export_yearly.py
"""

import ee
import time
from icecream import ic
ic.configureOutput(prefix=f'Debug | ', includeContext=True)

try:
    ee.Initialize()
except Exception as e:
    print("Initialization failed. Run 'earthengine authenticate' and 'earthengine set_project <your-project-id>' first.")
    raise

# ============================================
# CONFIGURATION – UPDATE THESE
# ============================================
ASSET_PATH = 'projects/**********/assets/Zambia_Administrative_Boundaries_Districts'   
EXPORT_FOLDER = 'zambia-all-district-lst'                      # Google Drive folder
BANDS = 'LST_Day_1km'

#!/usr/bin/env python3
"""Export MODIS MOD13A2 NDVI/EVI for Southern Province districts (16-day composites)."""

# Load districts
districts = ee.FeatureCollection(ASSET_PATH)
print(f"Loaded {districts.size().getInfo()} districts")

# ============================================
# QUALITY MASK FUNCTION (DEFINED FIRST)
# ============================================
def mask_quality(image):
    """
    Mask poor quality and cloudy pixels using QC_Day band.
    Keeps only pixels where:
        - LST quality bits (0-1) = 00 (good)
        - Cloud contamination bits (2-3) = 00 (cloud free)
    """
    qa = image.select('QC_Day')
    # Keep only good (0) or marginal (1) quality pixels
    quality_mask = qa.bitwiseAnd(0b00000011).eq(0)
    return image.updateMask(quality_mask)

# ============================================
# EXTRACTION FUNCTION FOR ONE 16-DAY IMAGE
# ============================================
def extract_daily_features(image):
    """Extract district averages for a single MODIS composite."""
    # 1. Apply quality mask
    masked_image = mask_quality(image)
    # 2. Select the bands we actually need
    selected_image = masked_image.select(BANDS)
    
    date_str = image.date().format('YYYY-MM-dd')
    year = image.date().get('year')
    month = image.date().get('month')
    day = image.date().get('day')
    
    def map_district(district):
        stats = selected_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=district.geometry(),
            scale=1000,          # MODIS 1km resolution
            maxPixels=1e9
        )
        
        lst_raw = stats.get(BANDS)
        lst_celsius = ee.Number(ee.Algorithms.If(
            lst_raw,
            ee.Number(lst_raw).multiply(0.02).subtract(273.15),
            -999
        ))
        
        return ee.Feature(None, {
            'district': district.get('DISTRICT'),   # Update if your property name differs
            'year': year,
            'month': month,
            'day': day,
            'date': date_str,
            'LST_raw': lst_raw,
            'LST_Celsius': lst_celsius
        })
    
    return districts.map(map_district)

# ============================================
# LOOP OVER YEARS AND SUBMIT EXPORT TASKS
# ============================================
years = range(2000, 2026)   # 2001 to 2025 (2000 may have partial data)

for year in years:
    print(f"Processing year {year}...")
    
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    # Load the full collection (without selecting bands yet)
    yearly_collection = (ee.ImageCollection('MODIS/061/MOD11A1')
                         .filterDate(start, end))
    
    # Check number of images
    count = yearly_collection.size().getInfo()
    print(f"  Found {count} daily images for {year}")
    
    if count > 0:
        # Map extraction over all images and flatten
        yearly_features = yearly_collection.map(extract_daily_features).flatten()
        
        # Export to Drive
        task = ee.batch.Export.table.toDrive(
            collection=yearly_features,
            description=f'LST_SouthernProvince_{year}_Daily',
            folder=EXPORT_FOLDER,
            fileNamePrefix=f'lst_southern_{year}',
            fileFormat='CSV'
        )
        task.start()
        print(f"  → Task started: {task.id}")
    else:
        print(f"  → No MODIS data found for {year}")
    
    time.sleep(2)   # Small delay to avoid overwhelming the queue

print("\nAll tasks submitted")

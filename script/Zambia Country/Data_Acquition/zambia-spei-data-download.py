#!/usr/bin/env python3
"""
Export daily precipitation (CHIRPS) and PET (ERA5-Land) for Southern Province districts.
One CSV per year (2000-2025) with columns: district, year, month, day, date, precip_mm, pet_mm.
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
ASSET_PATH = 'projects/******************/assets/Zambia_Administrative_Boundaries_Districts'   # CHANGE THIS to your actual asset path
EXPORT_FOLDER = 'zambia-all-district-daily-pet-precip'                      # Google Drive folder
PRECIP_BAND = 'precipitation'
PET_BAND = 'potential_evaporation_sum'

#!/usr/bin/env python3
"""Export MODIS MOD13A2 NDVI/EVI for Southern Province districts (16-day composites)."""

# Load districts
districts = ee.FeatureCollection(ASSET_PATH)
print(f"Loaded {districts.size().getInfo()} districts")

def combine_collections(year):
    """Return an ImageCollection where each image has both 'precip' and 'pet_mm' bands."""
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
              .filterDate(start, end)
              .select(PRECIP_BAND))
    era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
            .filterDate(start, end)
            .select(PET_BAND))

    def combine_images(chirps_img):
        date = chirps_img.date()
        era5_img = era5.filterDate(date, date.advance(1, 'day')).first()
        precip = chirps_img.select(PRECIP_BAND).rename('precip')
        pet = era5_img.select(PET_BAND).rename('pet')
        pet_mm = pet.multiply(1000).rename('pet_mm')
        combined = precip.addBands(pet_mm)
        return combined.set('system:time_start', chirps_img.get('system:time_start'))

    combined_collection = chirps.map(combine_images)
    return combined_collection

def extract_daily_features(image):
    """Extract district averages for a single MODIS composite."""
    # 1. Apply quality mask
    #masked_image = mask_quality(image)
    # 2. Select the bands we actually need
    #selected_image = masked_image.select(BANDS)
    
    date_str = image.date().format('YYYY-MM-dd')
    year = image.date().get('year')
    month = image.date().get('month')
    day = image.date().get('day')

    precip_img = image.select('precip')
    pet_img = image.select('pet_mm')

    def map_district(district):
        precip_stats = precip_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district.geometry(),
                scale=11132,
                maxPixels=1e9
            )
        precip_raw = precip_stats.get('precip')
        precip_mm = ee.Number(ee.Algorithms.If(precip_raw, precip_raw, -999))

        pet_stats = pet_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district.geometry(),
                scale=11132,
                maxPixels=1e9
            )

        pet_raw = pet_stats.get('pet_mm')
        pet_mm = ee.Number(ee.Algorithms.If(pet_raw, pet_raw, -999))

        return ee.Feature(None, {
            'district': district.get('DISTRICT'),   # Update if your property name differs
            'year': year,
            'month': month,
            'day': day,
            'date': date_str,
            'precip_mm': precip_mm,
            'pet_mm': pet_mm
        })
    
    return districts.map(map_district)

# ============================================
# LOOP OVER YEARS AND SUBMIT EXPORT TASKS
# ============================================
years = range(2000, 2026)   # 2001 to 2025 (2000 may have partial data)

for year in years:
    print(f"Processing year {year}...")
    
    # Load the full collection (without selecting bands yet)
    combined_collection = combine_collections(year)
    count = combined_collection.size().getInfo()
    print(f" Found {count} daily images for {year}")
    
    if count > 0:
        # Map extraction over all images and flatten
        yearly_features = combined_collection.map(extract_daily_features).flatten()

        # Export to Drive
        task = ee.batch.Export.table.toDrive(
            collection=yearly_features,
            description=f'DailyPrecipPET_zambia_{year}',
            folder=EXPORT_FOLDER,
            fileNamePrefix=f'daily_precip_pet_zambia_{year}',
            fileFormat='CSV'
        )
        task.start()
        print(f"  → Task started: {task.id}")
    else:
        print(f"  → No data found for {year}")
    
    time.sleep(2)   # Small delay to avoid overwhelming the queue

print("\nAll tasks submitted")

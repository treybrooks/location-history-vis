from datetime import datetime, timezone
import polars as pl
import polars.selectors as cs
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import ijson
import mmap
import time

DATA_PATH = "Records.json"
GEOJSON_PATH = "geojson/ne_50m_admin_0_countries.geojson"

def get_countries(df, granularity=2):
    print("Labeling country data...")
    start_time = time.time()
    # round lat/long and convert to Points for geometry comparison
    df = df.with_columns([
        pl.col('lat').round(granularity).alias('lat_round'),
        pl.col('lon').round(granularity).alias('lon_round')
    ])

    # Collect unique rounded points and make them Geo savvy
    unique_df = df.select(['lat_round', 'lon_round']).unique().to_pandas()
    unique_df['geometry'] = [Point(*p) for p in tuple(zip(unique_df['lon_round'], unique_df['lat_round']))]
    unique_gdf = gpd.GeoDataFrame(unique_df, crs=4326)

    # Load Country GeoData and join to unique points
    countries = gpd.read_file(GEOJSON_PATH)
    labeled_gdf = gpd.sjoin(unique_gdf, countries, how='inner')
    labeled_gdf['country'] = labeled_gdf['name']

    # Convert to Polars and join labeled back to original data
    country_df = pl.from_pandas(labeled_gdf[['lat_round', 'lon_round', 'country']])
    df = df.join(country_df, on=['lat_round', 'lon_round'], how='left')

    end_time = time.time()
    print(f"Labeling completed in {end_time - start_time:.2f} seconds.")
    # filter useless rows and drop now unnessesary colummns
    return df.filter(pl.col('country') != 'other').drop(['lat_round', 'lon_round'])

def clean_location_data():
    print("Starting to process the data...")
    start_time = time.time()

    # Get total number of records
    with open(DATA_PATH, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        locations = ijson.items(mm, 'locations.item')
        df = pl.DataFrame(locations)
    
    # little formatting and select only the important columns
    df = df.select([
        (pl.col('latitudeE7').cast(pl.Float64) / 1e7).alias('lat'),
        (pl.col('longitudeE7').cast(pl.Float64) / 1e7).alias('lon'),
        (pl.col('timestamp').str.to_datetime()).alias('time')
    ]).drop_nulls(subset=cs.float())

    end_time = time.time()
    print(f"Ingestion completed in {end_time - start_time:.2f} seconds.")
    return df

if __name__ == "__main__":
    df = clean_location_data()
    
    df = get_countries(df)
    
    df = df.sample(fraction=0.05, seed=0)

    # Filter by date if you like
    # df = df.filter(pl.col('time') >= datetime(2020, 1, 1).replace(tzinfo=timezone.utc))

    # Convert time column to what the graph expects
    df = df.with_columns([(pl.col('time').dt.strftime("%Y-%m-%d %H:%M:%S"))])
    df.write_json('location_sample.json')

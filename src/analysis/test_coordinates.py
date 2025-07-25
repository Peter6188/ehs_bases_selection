#!/usr/bin/env python3
"""
Quick test to verify coordinates and distance calculations
"""

import pandas as pd
import numpy as np
from pyproj import Transformer

# Load and check data
pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')

print("Sample of raw coordinates:")
print(pop_df[['GEO_NAME', 'latitude', 'longitude']].head())
print(f"\nLatitude range: {pop_df['latitude'].min()} to {pop_df['latitude'].max()}")
print(f"Longitude range: {pop_df['longitude'].min()} to {pop_df['longitude'].max()}")

# Transform coordinates
transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
utm_easting = pop_df['longitude']
utm_northing = pop_df['latitude']

lon_deg, lat_deg = transformer.transform(utm_easting.values, utm_northing.values)

print(f"\nAfter transformation:")
print(f"Latitude range: {lat_deg.min():.2f} to {lat_deg.max():.2f}")
print(f"Longitude range: {lon_deg.min():.2f} to {lon_deg.max():.2f}")

# Test Haversine distance between two Nova Scotia points
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Test with Halifax (approx 44.65, -63.57) and Sydney (approx 46.14, -60.18)
halifax_lat, halifax_lon = 44.65, -63.57
sydney_lat, sydney_lon = 46.14, -60.18

distance = haversine_distance(halifax_lat, halifax_lon, sydney_lat, sydney_lon)
print(f"\nTest distance Halifax to Sydney: {distance:.2f} km (should be ~300km)")

# Test with first two transformed coordinates
if len(lat_deg) >= 2:
    test_dist = haversine_distance(lat_deg[0], lon_deg[0], lat_deg[1], lon_deg[1])
    print(f"Distance between first two communities: {test_dist:.2f} km")
    print(f"Community 1: ({lat_deg[0]:.2f}, {lon_deg[0]:.2f})")
    print(f"Community 2: ({lat_deg[1]:.2f}, {lon_deg[1]:.2f})")

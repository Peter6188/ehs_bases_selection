#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and test data
pop_df = pd.read_csv('0 polulation_location_polygon.csv')
hospitals_gdf = gpd.read_file('1 Hospitals.geojson')

print(f"Population data: {pop_df.shape}")
print(f"Hospital data: {hospitals_gdf.shape}")

# Quick clustering test
clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]

X = clean_df[['longitude', 'latitude']].values[:20]  # Test with first 20
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
weights = clean_df['C1_COUNT_TOTAL'].values[:20]

# Simple k=4 clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled, sample_weight=weights)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster centers (lon, lat):")
for i, center in enumerate(centers):
    print(f"  EHS-{i+1}: {center[1]:.6f}, {center[0]:.6f}")

print("Test successful!")

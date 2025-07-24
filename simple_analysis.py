#!/usr/bin/env python3
"""
Simple EMS Analysis Runner - Simplified version for testing
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("=== Starting EMS Base Location Analysis ===")

try:
    # Load population data
    print("Loading population data...")
    pop_df = pd.read_csv('0 polulation_location_polygon.csv')
    print(f"Population data loaded: {pop_df.shape}")
    
    # Load hospital data
    print("Loading hospital data...")
    hospitals_gdf = gpd.read_file('1 Hospitals.geojson')
    print(f"Hospital data loaded: {hospitals_gdf.shape}")
    
    # Clean population data
    print("Cleaning data...")
    clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
    clean_df = clean_df[
        (clean_df['latitude'] >= 43.0) & (clean_df['latitude'] <= 47.0) &
        (clean_df['longitude'] >= -67.0) & (clean_df['longitude'] <= -59.0)
    ]
    print(f"Clean data: {len(clean_df)} communities, {clean_df['C1_COUNT_TOTAL'].sum():,} total population")
    
    # Prepare data for clustering
    X = clean_df[['longitude', 'latitude']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weights = clean_df['C1_COUNT_TOTAL'].values
    
    # Find optimal k using silhouette score
    print("Finding optimal number of clusters...")
    best_k = 2
    best_score = 0
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        print(f"Testing k={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled, sample_weight=sample_weights)
        
        labels = kmeans.labels_
        score = silhouette_score(X_scaled, labels, sample_weight=sample_weights)
        silhouette_scores.append(score)
        print(f"Silhouette: {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal k: {best_k} (silhouette score: {best_score:.3f})")
    
    # Perform final clustering
    print(f"Performing final clustering with k={best_k}...")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_kmeans.fit(X_scaled, sample_weight=sample_weights)
    
    # Get cluster centers
    cluster_centers_scaled = final_kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Add cluster assignments
    clean_df['cluster'] = final_kmeans.labels_
    
    # Calculate distances
    print("Calculating coverage metrics...")
    distances = []
    for idx, row in clean_df.iterrows():
        cluster_id = row['cluster']
        center_lat, center_lon = cluster_centers[cluster_id, 1], cluster_centers[cluster_id, 0]
        
        # Haversine distance approximation
        lat1, lon1 = np.radians([row['latitude'], row['longitude']])
        lat2, lon2 = np.radians([center_lat, center_lon])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = c * 6371  # Earth radius in km
        distances.append(distance)
    
    clean_df['distance_to_ems'] = distances
    
    # Create output
    print("Creating results...")
    
    # EHS locations
    ems_locations = []
    for i, center in enumerate(cluster_centers):
        ems_locations.append({
            'EHS_Base_ID': f'EHS-{i+1}',
            'Longitude': center[0],
            'Latitude': center[1]
        })
    
    ems_df = pd.DataFrame(ems_locations)
    ems_df.to_csv('proposed_ems_locations.csv', index=False)
    
    # Community assignments
    clean_df[['GEO_NAME', 'latitude', 'longitude', 'C1_COUNT_TOTAL', 'cluster', 'distance_to_ems']].to_csv('community_assignments.csv', index=False)
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Optimal number of EHS bases: {best_k}")
    print(f"Total population served: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    print(f"Average distance to EHS base: {np.mean(distances):.2f} km")
    print(f"Maximum distance to EHS base: {np.max(distances):.2f} km")
    
    total_pop = clean_df['C1_COUNT_TOTAL'].sum()
    weighted_avg = (clean_df['distance_to_ems'] * clean_df['C1_COUNT_TOTAL']).sum() / total_pop
    print(f"Population-weighted average distance: {weighted_avg:.2f} km")
    
    print("\nProposed EHS Base Locations:")
    for location in ems_locations:
        print(f"  {location['EHS_Base_ID']}: {location['Latitude']:.6f}, {location['Longitude']:.6f}")
    
    # Coverage analysis
    print("\nCoverage Analysis:")
    for threshold in [10, 15, 20, 30]:
        coverage = clean_df[clean_df['distance_to_ems'] <= threshold]['C1_COUNT_TOTAL'].sum()
        percentage = (coverage / total_pop) * 100
        print(f"  Within {threshold} km: {coverage:,} people ({percentage:.1f}%)")
    
    print("\nFiles generated:")
    print("  - proposed_ems_locations.csv")
    print("  - community_assignments.csv")
    
    print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

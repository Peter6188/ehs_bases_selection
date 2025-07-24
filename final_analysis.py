#!/usr/bin/env python3
"""
EMS Base Location Analysis - Final Version
Finds optimal Emergency Health Services base locations in Nova Scotia
using population-weighted K-Means clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def main():
    print("=== EMS Base Location Analysis ===")
    
    # Load population data
    print("Loading population data...")
    pop_df = pd.read_csv('0 polulation_location_polygon.csv')
    print(f"Population data shape: {pop_df.shape}")
    
    # Clean data
    print("Cleaning data...")
    clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
    
    # Filter to Nova Scotia bounds
    clean_df = clean_df[
        (clean_df['latitude'] >= 43.0) & (clean_df['latitude'] <= 47.0) &
        (clean_df['longitude'] >= -67.0) & (clean_df['longitude'] <= -59.0)
    ]
    
    print(f"Clean communities: {len(clean_df)}")
    print(f"Total population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    # Prepare data for clustering
    X = clean_df[['longitude', 'latitude']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weights = clean_df['C1_COUNT_TOTAL'].values
    
    # Find optimal number of clusters using silhouette score
    print("Finding optimal number of clusters...")
    best_k = 2
    best_score = -1
    scores = []
    
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled, sample_weight=sample_weights)
        score = silhouette_score(X_scaled, labels, sample_weight=sample_weights)
        scores.append(score)
        print(f"k={k}: silhouette_score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal k: {best_k} (score: {best_score:.3f})")
    
    # Perform final clustering
    print("Performing final clustering...")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_kmeans.fit(X_scaled, sample_weight=sample_weights)
    
    # Get cluster centers in original coordinates
    cluster_centers_scaled = final_kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Add cluster assignments to data
    clean_df = clean_df.copy()
    clean_df['cluster'] = final_kmeans.labels_
    
    # Calculate distances to nearest EHS base
    distances = []
    for idx, row in clean_df.iterrows():
        cluster_id = row['cluster']
        center_lon, center_lat = cluster_centers[cluster_id]
        
        # Simple distance calculation (Haversine approximation)
        lat1, lon1 = np.radians([row['latitude'], row['longitude']])
        lat2, lon2 = np.radians([center_lat, center_lon])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_km = c * 6371  # Earth radius in km
        distances.append(distance_km)
    
    clean_df['distance_to_ems'] = distances
    
    # Create EHS locations dataframe
    ems_locations = []
    for i, center in enumerate(cluster_centers):
        ems_locations.append({
            'EHS_Base_ID': f'EHS-{i+1}',
            'Longitude': center[0],
            'Latitude': center[1],
            'Communities_Served': len(clean_df[clean_df['cluster'] == i]),
            'Population_Served': clean_df[clean_df['cluster'] == i]['C1_COUNT_TOTAL'].sum()
        })
    
    ems_df = pd.DataFrame(ems_locations)
    
    # Save results
    print("Saving results...")
    ems_df.to_csv('proposed_ems_locations.csv', index=False)
    
    output_df = clean_df[['GEO_NAME', 'latitude', 'longitude', 'C1_COUNT_TOTAL', 
                         'cluster', 'distance_to_ems']].copy()
    output_df['assigned_ems'] = output_df['cluster'].apply(lambda x: f'EHS-{x+1}')
    output_df.to_csv('community_assignments.csv', index=False)
    
    # Print results
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Optimal number of EHS bases: {best_k}")
    print(f"Total communities analyzed: {len(clean_df)}")
    print(f"Total population served: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    print(f"\nDistance Statistics:")
    print(f"Average distance to EHS base: {np.mean(distances):.2f} km")
    print(f"Median distance to EHS base: {np.median(distances):.2f} km")
    print(f"Maximum distance to EHS base: {np.max(distances):.2f} km")
    
    # Population-weighted average distance
    total_pop = clean_df['C1_COUNT_TOTAL'].sum()
    weighted_avg_distance = (clean_df['distance_to_ems'] * clean_df['C1_COUNT_TOTAL']).sum() / total_pop
    print(f"Population-weighted average distance: {weighted_avg_distance:.2f} km")
    
    print(f"\nProposed EHS Base Locations:")
    for _, location in ems_df.iterrows():
        print(f"  {location['EHS_Base_ID']}: "
              f"Lat {location['Latitude']:.6f}, Lon {location['Longitude']:.6f}")
        print(f"    Serves {location['Communities_Served']} communities, "
              f"{location['Population_Served']:,} people")
    
    # Coverage analysis
    print(f"\nCoverage Analysis:")
    for threshold in [10, 15, 20, 30]:
        within_threshold = clean_df[clean_df['distance_to_ems'] <= threshold]
        coverage_pop = within_threshold['C1_COUNT_TOTAL'].sum()
        coverage_pct = (coverage_pop / total_pop) * 100
        coverage_communities = len(within_threshold)
        community_pct = (coverage_communities / len(clean_df)) * 100
        
        print(f"  Within {threshold} km:")
        print(f"    Population: {coverage_pop:,} ({coverage_pct:.1f}%)")
        print(f"    Communities: {coverage_communities} ({community_pct:.1f}%)")
    
    print(f"\nOutput files generated:")
    print(f"  - proposed_ems_locations.csv: EHS base coordinates and details")
    print(f"  - community_assignments.csv: Community assignments and distances")
    
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()

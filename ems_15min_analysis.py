#!/usr/bin/env python3
"""
EMS 15-Minute Coverage Analysis
Determines minimum number of EHS bases needed for 15-minute coverage of all 95 communities
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in kilometers"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371  # Earth radius in km

def test_coverage(clean_df, k, target_distance=15):
    """Test if k clusters provides 100% coverage within target distance"""
    X = clean_df[['longitude', 'latitude']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    weights = clean_df['C1_COUNT_TOTAL'].values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled, sample_weight=weights)
    
    # Get cluster centers in original coordinates
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Calculate distances
    max_distance = 0
    over_threshold = 0
    distances = []
    
    for idx, row in clean_df.iterrows():
        cluster_id = kmeans.labels_[idx]
        center_lon, center_lat = centers[cluster_id]
        
        distance = haversine_distance(row['latitude'], row['longitude'], 
                                    center_lat, center_lon)
        distances.append(distance)
        
        if distance > target_distance:
            over_threshold += 1
        
        max_distance = max(max_distance, distance)
    
    coverage_pct = ((len(clean_df) - over_threshold) / len(clean_df)) * 100
    
    return coverage_pct, max_distance, over_threshold, centers, kmeans.labels_, distances

def main():
    print("=== EMS 15-Minute Coverage Analysis ===")
    print("Finding minimum EHS bases for 15-minute coverage of all communities\n")
    
    # Load and clean data
    pop_df = pd.read_csv('0 polulation_location_polygon.csv')
    clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
    clean_df = clean_df[
        (clean_df['latitude'] >= 43.0) & (clean_df['latitude'] <= 47.0) &
        (clean_df['longitude'] >= -67.0) & (clean_df['longitude'] <= -59.0)
    ]
    
    print(f"Communities analyzed: {len(clean_df)}")
    print(f"Total population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    # Test different numbers of clusters
    results = []
    target_distance = 15  # 15 km = approximately 15 minutes for emergency vehicles
    
    print(f"\nTesting cluster sizes for {target_distance}km coverage:")
    print("k\tCoverage%\tMax Distance\tCommunities >15km")
    print("-" * 50)
    
    for k in range(2, 25):  # Test up to 24 clusters
        coverage_pct, max_dist, over_threshold, centers, labels, distances = test_coverage(
            clean_df, k, target_distance
        )
        
        results.append({
            'k': k,
            'coverage_pct': coverage_pct,
            'max_distance': max_dist,
            'over_threshold': over_threshold,
            'centers': centers,
            'labels': labels,
            'distances': distances
        })
        
        print(f"{k}\t{coverage_pct:.1f}%\t\t{max_dist:.2f} km\t\t{over_threshold}")
        
        # Stop when we achieve 100% coverage
        if coverage_pct >= 100.0:
            optimal_k = k
            print(f"\n✅ OPTIMAL SOLUTION FOUND: k = {k}")
            print(f"All {len(clean_df)} communities within {target_distance}km!")
            break
    else:
        # If no perfect solution found, choose best available
        best_result = max(results, key=lambda x: x['coverage_pct'])
        optimal_k = best_result['k']
        print(f"\n⚠️  No perfect solution found. Best: k = {optimal_k}")
        print(f"Coverage: {best_result['coverage_pct']:.1f}%")
    
    # Get optimal solution details
    optimal_result = next(r for r in results if r['k'] == optimal_k)
    
    # Create results
    print(f"\n=== OPTIMAL EHS BASE CONFIGURATION ===")
    print(f"Number of EHS bases needed: {optimal_k}")
    print(f"Coverage achieved: {optimal_result['coverage_pct']:.1f}%")
    print(f"Maximum distance: {optimal_result['max_distance']:.2f} km")
    
    print(f"\nProposed EHS Base Locations:")
    ems_locations = []
    for i, center in enumerate(optimal_result['centers']):
        lon, lat = center
        ems_locations.append({
            'EHS_Base_ID': f'EHS-{i+1:02d}',
            'Latitude': lat,
            'Longitude': lon
        })
        print(f"  EHS-{i+1:02d}: {lat:.6f}, {lon:.6f}")
    
    # Add cluster assignments to dataframe
    clean_df = clean_df.copy()
    clean_df['cluster'] = optimal_result['labels']
    clean_df['distance_to_ems'] = optimal_result['distances']
    
    # Save results
    ems_df = pd.DataFrame(ems_locations)
    ems_df.to_csv('optimal_ems_locations_15min.csv', index=False)
    
    assignments_df = clean_df[['GEO_NAME', 'latitude', 'longitude', 'C1_COUNT_TOTAL', 
                              'cluster', 'distance_to_ems']].copy()
    assignments_df['assigned_ems'] = assignments_df['cluster'].apply(lambda x: f'EHS-{x+1:02d}')
    assignments_df.to_csv('community_assignments_15min.csv', index=False)
    
    # Coverage analysis
    print(f"\n=== COVERAGE ANALYSIS ===")
    total_pop = clean_df['C1_COUNT_TOTAL'].sum()
    
    thresholds = [5, 10, 15, 20, 30]
    for threshold in thresholds:
        within = clean_df[clean_df['distance_to_ems'] <= threshold]
        comm_count = len(within)
        pop_count = within['C1_COUNT_TOTAL'].sum()
        comm_pct = (comm_count / len(clean_df)) * 100
        pop_pct = (pop_count / total_pop) * 100
        
        status = "✅" if threshold >= 15 and comm_pct >= 100 else "📊"
        print(f"{status} Within {threshold:2d}km: {comm_count:2d}/{len(clean_df)} communities ({comm_pct:5.1f}%), "
              f"{pop_count:6,}/{total_pop:6,} people ({pop_pct:5.1f}%)")
    
    # Population-weighted metrics
    weighted_avg = (clean_df['distance_to_ems'] * clean_df['C1_COUNT_TOTAL']).sum() / total_pop
    print(f"\nPopulation-weighted average distance: {weighted_avg:.2f} km")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot coverage vs number of clusters
    k_values = [r['k'] for r in results]
    coverage_values = [r['coverage_pct'] for r in results]
    max_distances = [r['max_distance'] for r in results]
    
    plt.subplot(2, 1, 1)
    plt.plot(k_values, coverage_values, 'bo-', label='Coverage %')
    plt.axhline(y=100, color='red', linestyle='--', label='100% target')
    plt.axvline(x=optimal_k, color='green', linestyle=':', label=f'Optimal k={optimal_k}')
    plt.xlabel('Number of EHS Bases (k)')
    plt.ylabel('Coverage Percentage (%)')
    plt.title('Coverage vs Number of EHS Bases')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(k_values, max_distances, 'ro-', label='Max Distance')
    plt.axhline(y=15, color='red', linestyle='--', label='15km target')
    plt.axvline(x=optimal_k, color='green', linestyle=':', label=f'Optimal k={optimal_k}')
    plt.xlabel('Number of EHS Bases (k)')
    plt.ylabel('Maximum Distance (km)')
    plt.title('Maximum Distance vs Number of EHS Bases')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ems_15min_coverage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: ems_15min_coverage_analysis.png")
    
    print(f"\n=== FILES GENERATED ===")
    print(f"- optimal_ems_locations_15min.csv: {optimal_k} EHS base coordinates")
    print(f"- community_assignments_15min.csv: Community assignments and distances")
    print(f"- ems_15min_coverage_analysis.png: Coverage analysis visualization")
    
    print(f"\n🎯 RECOMMENDATION: Deploy {optimal_k} EHS bases for 15-minute coverage")

if __name__ == "__main__":
    main()

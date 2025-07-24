#!/usr/bin/env python3
"""
EMS Base Location Analysis for Nova Scotia
Using K-means clustering with population weighting to find optimal EHS base locations
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sk    print("\n=== Analysis Complete ===")
    print("Check the generated files:")
    print("- cluster_optimization.png: Optimization plots including 15-minute coverage")
    print("- ems_locations_map.png: Main results map")
    print("- cluster_statistics.png: Statistical plots")
    print("- proposed_ems_locations.csv: EHS base coordinates")
    print("- community_cluster_assignments.csv: Detailed assignments")
    print("- cluster_statistics.csv: Cluster summary statistics")
    print(f"\nFinal result: {best_k} EHS bases to ensure 15-minute coverage for all communities")trics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the population and hospital data"""
    print("Loading population data...")
    
    # Load population data
    pop_df = pd.read_csv('0 polulation_location_polygon.csv')
    
    # Display basic info about the data
    print(f"Population data shape: {pop_df.shape}")
    print(f"Columns: {pop_df.columns.tolist()}")
    
    # Check for missing values in key columns
    print("\nMissing values check:")
    key_cols = ['GEO_NAME', 'C1_COUNT_TOTAL', 'latitude', 'longitude']
    for col in key_cols:
        if col in pop_df.columns:
            print(f"{col}: {pop_df[col].isnull().sum()}")
    
    # Load hospital data
    print("\nLoading hospital data...")
    hospitals_gdf = gpd.read_file('1 Hospitals.geojson')
    
    print(f"Hospital data shape: {hospitals_gdf.shape}")
    print(f"Hospital types: {hospitals_gdf['type'].value_counts()}")
    
    return pop_df, hospitals_gdf

def clean_population_data(pop_df):
    """Clean and prepare population data for analysis"""
    print("\nCleaning population data...")
    
    # Remove rows with missing coordinates or population
    clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
    
    # Filter out rows with zero population
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
    
    # Remove any extreme outliers in coordinates (basic sanity check for NS)
    clean_df = clean_df[
        (clean_df['latitude'] >= 43.0) & (clean_df['latitude'] <= 47.0) &
        (clean_df['longitude'] >= -67.0) & (clean_df['longitude'] <= -59.0)
    ]
    
    print(f"After cleaning: {len(clean_df)} communities")
    print(f"Total population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    return clean_df

def find_optimal_clusters(X, sample_weights, clean_df, max_clusters=20, max_distance_km=15):
    """Find optimal number of clusters ensuring 15-minute coverage (15km threshold)"""
    print(f"\nFinding optimal number of clusters for {max_distance_km}km coverage...")
    
    inertias = []
    silhouette_scores = []
    max_distances = []
    coverage_percentages = []
    K_range = range(2, max_clusters + 1)
    
    # Standardize coordinates for distance calculation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for k in K_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled, sample_weight=sample_weights)
        
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        labels = kmeans.labels_
        sil_score = silhouette_score(X_scaled, labels, sample_weight=sample_weights)
        silhouette_scores.append(sil_score)
        
        # Calculate actual distances to cluster centers
        cluster_centers_scaled = kmeans.cluster_centers_
        cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
        
        distances = []
        for idx, row in clean_df.iterrows():
            cluster_id = labels[idx]
            center_lon, center_lat = cluster_centers[cluster_id]
            
            # Haversine distance calculation
            lat1, lon1 = np.radians([row['latitude'], row['longitude']])
            lat2, lon2 = np.radians([center_lat, center_lon])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_km = c * 6371  # Earth radius in km
            distances.append(distance_km)
        
        max_dist = np.max(distances)
        max_distances.append(max_dist)
        
        # Calculate coverage within 15km
        within_threshold = np.sum(np.array(distances) <= max_distance_km)
        coverage_pct = (within_threshold / len(distances)) * 100
        coverage_percentages.append(coverage_pct)
        
        print(f"  Silhouette: {sil_score:.3f}, Max distance: {max_dist:.2f}km, Coverage: {coverage_pct:.1f}%")
    
    # Plot optimization metrics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Elbow curve
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)
    
    # Max distances
    ax3.plot(K_range, max_distances, 'go-')
    ax3.axhline(y=max_distance_km, color='red', linestyle='--', label=f'{max_distance_km}km threshold')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Maximum Distance (km)')
    ax3.set_title('Maximum Distance to EHS Base')
    ax3.legend()
    ax3.grid(True)
    
    # Coverage percentages
    ax4.plot(K_range, coverage_percentages, 'mo-')
    ax4.axhline(y=100, color='red', linestyle='--', label='100% coverage')
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Coverage Percentage (%)')
    ax4.set_title(f'Percentage within {max_distance_km}km')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_optimization.png', dpi=300, bbox_inches='tight')
    print("Cluster optimization plot saved as 'cluster_optimization.png'")
    plt.close()
    
    # Find minimum k that achieves 100% coverage within 15km
    full_coverage_k = None
    for i, coverage in enumerate(coverage_percentages):
        if coverage >= 100.0:
            full_coverage_k = K_range[i]
            break
    
    if full_coverage_k:
        print(f"\nMinimum k for 100% coverage within {max_distance_km}km: {full_coverage_k}")
        best_k = full_coverage_k
    else:
        # If no k achieves 100% coverage, find k that minimizes max distance
        min_max_dist_idx = np.argmin(max_distances)
        best_k = K_range[min_max_dist_idx]
        print(f"\nNo k achieves 100% coverage. Best k (min max distance): {best_k}")
        print(f"Best coverage achieved: {coverage_percentages[min_max_dist_idx]:.1f}%")
    
    return best_k, inertias, silhouette_scores, max_distances, coverage_percentages

def perform_clustering(clean_df, k):
    """Perform K-means clustering with population weighting"""
    print(f"\nPerforming K-means clustering with k={k}...")
    
    # Prepare features (coordinates)
    X = clean_df[['longitude', 'latitude']].values
    
    # Standardize the coordinates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use population as sample weights
    sample_weights = clean_df['C1_COUNT_TOTAL'].values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled, sample_weight=sample_weights)
    
    # Get cluster centers in original scale
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    # Add cluster labels to dataframe
    clean_df = clean_df.copy()
    clean_df['cluster'] = kmeans.labels_
    
    # Calculate cluster statistics
    cluster_stats = []
    for i in range(k):
        cluster_data = clean_df[clean_df['cluster'] == i]
        stats = {
            'cluster_id': i,
            'center_longitude': cluster_centers[i, 0],
            'center_latitude': cluster_centers[i, 1],
            'num_communities': len(cluster_data),
            'total_population': cluster_data['C1_COUNT_TOTAL'].sum(),
            'avg_population': cluster_data['C1_COUNT_TOTAL'].mean(),
            'population_density': cluster_data['C1_COUNT_TOTAL'].sum() / len(cluster_data)
        }
        cluster_stats.append(stats)
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    return clean_df, cluster_centers, cluster_stats_df, kmeans

def visualize_results(clean_df, cluster_centers, hospitals_gdf, cluster_stats_df):
    """Create visualizations of the clustering results"""
    print("\nCreating visualizations...")
    
    # Create main map
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Plot communities colored by cluster, sized by population
    scatter = ax.scatter(clean_df['longitude'], clean_df['latitude'], 
                        c=clean_df['cluster'], 
                        s=clean_df['C1_COUNT_TOTAL']/50,  # Scale down for visibility
                        alpha=0.6, 
                        cmap='tab10',
                        edgecolors='black',
                        linewidth=0.5)
    
    # Plot existing hospitals
    hospitals_gdf.plot(ax=ax, color='red', marker='h', markersize=100, 
                      label='Existing Hospitals', alpha=0.8, edgecolor='black')
    
    # Plot proposed EHS base locations (cluster centers)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
              c='yellow', s=300, marker='*', 
              label='Proposed EHS Bases', 
              edgecolors='black', linewidth=2)
    
    # Add cluster center labels
    for i, (lon, lat) in enumerate(cluster_centers):
        ax.annotate(f'EHS-{i+1}', (lon, lat), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Optimal EHS Base Locations in Nova Scotia\n(Community size proportional to population)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for clusters
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID')
    
    plt.tight_layout()
    plt.savefig('ems_locations_map.png', dpi=300, bbox_inches='tight')
    print("EMS locations map saved as 'ems_locations_map.png'")
    plt.close()  # Close the figure instead of showing
    
    # Create cluster statistics visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Population by cluster
    ax1.bar(cluster_stats_df['cluster_id'], cluster_stats_df['total_population'])
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Total Population')
    ax1.set_title('Total Population by Cluster')
    
    # Number of communities by cluster
    ax2.bar(cluster_stats_df['cluster_id'], cluster_stats_df['num_communities'])
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Communities')
    ax2.set_title('Number of Communities by Cluster')
    
    # Average population by cluster
    ax3.bar(cluster_stats_df['cluster_id'], cluster_stats_df['avg_population'])
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Average Population')
    ax3.set_title('Average Population by Cluster')
    
    # Population density by cluster
    ax4.bar(cluster_stats_df['cluster_id'], cluster_stats_df['population_density'])
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Population Density')
    ax4.set_title('Population Density by Cluster')
    
    plt.tight_layout()
    plt.savefig('cluster_statistics.png', dpi=300, bbox_inches='tight')
    print("Cluster statistics plot saved as 'cluster_statistics.png'")
    plt.close()  # Close the figure instead of showing

def calculate_coverage_metrics(clean_df, cluster_centers, hospitals_gdf):
    """Calculate coverage metrics for the proposed EHS bases using accurate Haversine distance"""
    print("\nCalculating coverage metrics using Haversine distance formula...")
    
    # Calculate distances from each community to its assigned EHS base using Haversine formula
    distances_to_assigned = []
    for idx, row in clean_df.iterrows():
        cluster_id = row['cluster']
        center_lon, center_lat = cluster_centers[cluster_id]
        
        # Haversine distance calculation
        lat1, lon1 = np.radians([row['latitude'], row['longitude']])
        lat2, lon2 = np.radians([center_lat, center_lon])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_km = c * 6371  # Earth radius in km
        distances_to_assigned.append(distance_km)
    
    clean_df['distance_to_ems'] = distances_to_assigned
    
    # Calculate coverage statistics
    print("\nCoverage Statistics:")
    print(f"Average distance to EHS base: {np.mean(distances_to_assigned):.2f} km")
    print(f"Median distance to EHS base: {np.median(distances_to_assigned):.2f} km")
    print(f"Maximum distance to EHS base: {np.max(distances_to_assigned):.2f} km")
    print(f"Minimum distance to EHS base: {np.min(distances_to_assigned):.2f} km")
    
    # Population-weighted average distance
    total_pop = clean_df['C1_COUNT_TOTAL'].sum()
    weighted_avg_distance = (clean_df['distance_to_ems'] * clean_df['C1_COUNT_TOTAL']).sum() / total_pop
    print(f"Population-weighted average distance: {weighted_avg_distance:.2f} km")
    
    # Coverage within different distance thresholds (15 minutes = ~15km for emergency vehicles)
    thresholds = [5, 10, 15, 20, 30]  # km (15km = ~15 minutes for emergency vehicles)
    print("\nPopulation coverage within distance thresholds:")
    for threshold in thresholds:
        within_threshold = clean_df[clean_df['distance_to_ems'] <= threshold]
        coverage_pop = within_threshold['C1_COUNT_TOTAL'].sum()
        coverage_communities = len(within_threshold)
        pop_percentage = (coverage_pop / total_pop) * 100
        comm_percentage = (coverage_communities / len(clean_df)) * 100
        
        print(f"Within {threshold} km (~{threshold} min):")
        print(f"  Population: {coverage_pop:,} people ({pop_percentage:.1f}%)")
        print(f"  Communities: {coverage_communities} ({comm_percentage:.1f}%)")
    
    # Check 15-minute coverage specifically
    within_15_min = clean_df[clean_df['distance_to_ems'] <= 15]
    coverage_15min = len(within_15_min)
    coverage_15min_pct = (coverage_15min / len(clean_df)) * 100
    
    print(f"\n*** 15-MINUTE COVERAGE TARGET ***")
    print(f"Communities within 15km (15min): {coverage_15min}/{len(clean_df)} ({coverage_15min_pct:.1f}%)")
    
    if coverage_15min_pct >= 100.0:
        print("✅ TARGET ACHIEVED: All communities within 15-minute response time!")
    else:
        print("❌ TARGET NOT MET: Some communities exceed 15-minute response time")
        over_15min = clean_df[clean_df['distance_to_ems'] > 15]
        print(f"Communities over 15min: {len(over_15min)}")
        print("Furthest communities:")
        furthest = over_15min.nlargest(5, 'distance_to_ems')[['GEO_NAME', 'distance_to_ems']]
        for _, row in furthest.iterrows():
            print(f"  {row['GEO_NAME']}: {row['distance_to_ems']:.2f} km")
    
    return clean_df

def export_results(cluster_centers, cluster_stats_df, clean_df):
    """Export results to CSV files"""
    print("\nExporting results...")
    
    # Export EHS base locations
    ems_locations = pd.DataFrame({
        'EHS_Base_ID': [f'EHS-{i+1}' for i in range(len(cluster_centers))],
        'Longitude': cluster_centers[:, 0],
        'Latitude': cluster_centers[:, 1]
    })
    
    # Merge with cluster statistics
    ems_locations = pd.concat([ems_locations, cluster_stats_df[['total_population', 'num_communities', 'population_density']]], axis=1)
    
    ems_locations.to_csv('proposed_ems_locations.csv', index=False)
    print("Proposed EMS locations saved to: proposed_ems_locations.csv")
    
    # Export detailed cluster assignments
    clean_df[['GEO_NAME', 'latitude', 'longitude', 'C1_COUNT_TOTAL', 'cluster', 'distance_to_ems']].to_csv('community_cluster_assignments.csv', index=False)
    print("Community cluster assignments saved to: community_cluster_assignments.csv")
    
    # Export cluster statistics
    cluster_stats_df.to_csv('cluster_statistics.csv', index=False)
    print("Cluster statistics saved to: cluster_statistics.csv")

def main():
    """Main analysis function"""
    print("=== EMS Base Location Analysis for Nova Scotia ===")
    print("Using K-means clustering with population weighting\n")
    
    # Load data
    pop_df, hospitals_gdf = load_and_prepare_data()
    
    # Clean data
    clean_df = clean_population_data(pop_df)
    
    # Prepare features for clustering
    X = clean_df[['longitude', 'latitude']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sample_weights = clean_df['C1_COUNT_TOTAL'].values
    
    # Find optimal number of clusters with 15-minute coverage constraint
    best_k, inertias, silhouette_scores, max_distances, coverage_percentages = find_optimal_clusters(
        X_scaled, sample_weights, clean_df, max_clusters=20, max_distance_km=15
    )
    
    # Perform clustering with optimal k
    clean_df, cluster_centers, cluster_stats_df, kmeans = perform_clustering(clean_df, best_k)
    
    # Display cluster statistics
    print("\nCluster Statistics:")
    print(cluster_stats_df.round(2))
    
    # Visualize results
    visualize_results(clean_df, cluster_centers, hospitals_gdf, cluster_stats_df)
    
    # Calculate coverage metrics
    clean_df = calculate_coverage_metrics(clean_df, cluster_centers, hospitals_gdf)
    
    # Export results
    export_results(cluster_centers, cluster_stats_df, clean_df)
    
    print("\n=== Analysis Complete ===")
    print("Check the generated files:")
    print("- cluster_optimization.png: Optimization plots")
    print("- ems_locations_map.png: Main results map")
    print("- cluster_statistics.png: Statistical plots")
    print("- proposed_ems_locations.csv: EMS base coordinates")
    print("- community_cluster_assignments.csv: Detailed assignments")
    print("- cluster_statistics.csv: Cluster summary statistics")

if __name__ == "__main__":
    main()

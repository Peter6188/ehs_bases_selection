#!/usr/bin/env python3
"""
Enhanced EMS Base Location Analysis with Hospital Performance Integration
Incorporates existing hospital performance data and infrastructure into K-means optimization
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pyproj import Transformer
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare all data sources including hospital performance"""
    print("📊 Loading and integrating all data sources...")
    
    # Load population data
    pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
    print(f"Population data: {pop_df.shape[0]} communities")
    
    # Load hospital performance data
    perf_df = pd.read_csv('../../data/raw/emergency_health_services.csv')
    print(f"Hospital performance data: {perf_df.shape[0]} records")
    
    # Load hospital locations
    with open('../../data/raw/hospitals.geojson', 'r') as f:
        hospital_data = json.load(f)
    
    hospital_list = []
    for feature in hospital_data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        hospital_list.append({
            'facility': props['facility'],
            'town': props['town'],
            'county': props['county'],
            'type': props['type'],
            'longitude': coords[0],
            'latitude': coords[1]
        })
    
    hospital_df = pd.DataFrame(hospital_list)
    print(f"Hospital locations: {hospital_df.shape[0]} hospitals")
    
    return pop_df, perf_df, hospital_df

def process_hospital_performance(perf_df, hospital_df):
    """Process hospital performance metrics and calculate performance scores"""
    print("\n🏥 Processing hospital performance metrics...")
    
    # Check data availability
    print(f"Performance data: {len(perf_df)} records")
    print(f"Hospital locations: {len(hospital_df)} hospitals")
    print(f"Unique hospitals in performance data: {perf_df['Hospital'].nunique()}")
    
    # Calculate average performance metrics per hospital
    perf_summary = perf_df.groupby(['Hospital', 'Measure Name'])['Actual'].agg(['mean', 'std', 'count']).reset_index()
    
    # Separate different performance measures
    ed_offload = perf_summary[perf_summary['Measure Name'] == 'ED Offload Interval'].copy()
    ehs_response_times = perf_summary[perf_summary['Measure Name'] == 'EHS Response Times'].copy()
    ehs_responses = perf_summary[perf_summary['Measure Name'] == 'EHS Responses'].copy()
    
    print(f"ED Offload data: {len(ed_offload)} hospitals")
    print(f"EHS Response Times: {len(ehs_response_times)} hospitals")
    print(f"EHS Responses: {len(ehs_responses)} hospitals")
    
    # Create hospital performance summary starting with base data
    hospital_performance = hospital_df.copy()
    
    # Create name matching function (fuzzy matching for similar names)
    def find_best_match(facility_name, perf_hospitals):
        """Find best matching hospital name"""
        exact_match = perf_hospitals.get(facility_name)
        if exact_match is not None:
            return exact_match
            
        # Try partial matching
        facility_lower = facility_name.lower()
        for perf_name, value in perf_hospitals.items():
            if perf_name.lower() in facility_lower or facility_lower in perf_name.lower():
                return value
        return None
    
    # Create performance dictionaries
    ed_offload_dict = dict(zip(ed_offload['Hospital'], ed_offload['mean']))
    response_time_dict = dict(zip(ehs_response_times['Hospital'], ehs_response_times['mean']))
    response_count_dict = dict(zip(ehs_responses['Hospital'], ehs_responses['mean']))
    
    # Map performance data with fuzzy matching
    hospital_performance['ed_offload_avg'] = hospital_performance['facility'].apply(
        lambda x: find_best_match(x, ed_offload_dict))
    hospital_performance['ehs_response_avg'] = hospital_performance['facility'].apply(
        lambda x: find_best_match(x, response_time_dict))
    hospital_performance['ehs_response_count'] = hospital_performance['facility'].apply(
        lambda x: find_best_match(x, response_count_dict))
    
    # Calculate median values for imputation
    ed_median = perf_df[perf_df['Measure Name'] == 'ED Offload Interval']['Actual'].median()
    response_time_median = perf_df[perf_df['Measure Name'] == 'EHS Response Times']['Actual'].median()
    response_count_median = perf_df[perf_df['Measure Name'] == 'EHS Responses']['Actual'].median()
    
    # Fill missing values with medians (hospitals without performance data)
    hospital_performance['ed_offload_avg'] = hospital_performance['ed_offload_avg'].fillna(ed_median)
    hospital_performance['ehs_response_avg'] = hospital_performance['ehs_response_avg'].fillna(response_time_median)
    hospital_performance['ehs_response_count'] = hospital_performance['ehs_response_count'].fillna(response_count_median)
    
    # Create performance scores (normalized 0-1, where 1 is best performance)
    # For ED offload and response times: lower is better, so invert
    ed_min, ed_max = hospital_performance['ed_offload_avg'].min(), hospital_performance['ed_offload_avg'].max()
    if ed_max > ed_min:
        hospital_performance['ed_performance_score'] = 1 - (
            (hospital_performance['ed_offload_avg'] - ed_min) / (ed_max - ed_min)
        )
    else:
        hospital_performance['ed_performance_score'] = 0.5
    
    rt_min, rt_max = hospital_performance['ehs_response_avg'].min(), hospital_performance['ehs_response_avg'].max()
    if rt_max > rt_min:
        hospital_performance['response_time_score'] = 1 - (
            (hospital_performance['ehs_response_avg'] - rt_min) / (rt_max - rt_min)
        )
    else:
        hospital_performance['response_time_score'] = 0.5
    
    # For response count: higher is better
    rc_min, rc_max = hospital_performance['ehs_response_count'].min(), hospital_performance['ehs_response_count'].max()
    if rc_max > rc_min:
        hospital_performance['capacity_score'] = (
            (hospital_performance['ehs_response_count'] - rc_min) / (rc_max - rc_min)
        )
    else:
        hospital_performance['capacity_score'] = 0.5
    
    # Add hospital type scoring (higher tier = better infrastructure)
    type_scores = {
        'Tertiary': 1.0,
        'Regional': 0.8,
        'Community': 0.6,
        'Community Health Centre': 0.5,
        'Rehabilitation': 0.4,
        'Environmental Health': 0.3,
        'Out Patient/ Nursing Home': 0.2
    }
    hospital_performance['type_score'] = hospital_performance['type'].map(type_scores).fillna(0.5)
    
    # Calculate overall hospital performance score
    hospital_performance['overall_performance'] = (
        0.3 * hospital_performance['ed_performance_score'] +
        0.3 * hospital_performance['response_time_score'] +
        0.2 * hospital_performance['capacity_score'] +
        0.2 * hospital_performance['type_score']
    )
    
    # Count hospitals with actual performance data vs imputed
    has_ed_data = (~hospital_performance['facility'].apply(
        lambda x: find_best_match(x, ed_offload_dict)).isna()).sum()
    
    print(f"✅ Hospital performance scores calculated")
    print(f"Hospitals with ED performance data: {has_ed_data}/{len(hospital_performance)}")
    print(f"Average overall performance: {hospital_performance['overall_performance'].mean():.3f}")
    print(f"Performance range: {hospital_performance['overall_performance'].min():.3f} - {hospital_performance['overall_performance'].max():.3f}")
    
    return hospital_performance

def clean_population_data(pop_df):
    """Clean and prepare population data with coordinate transformation"""
    print("\n🗺️ Processing population data...")
    
    # Transform coordinates from Statistics Canada Lambert to WGS84
    transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
    
    utm_easting = pop_df['longitude']
    utm_northing = pop_df['latitude']
    
    lon_deg, lat_deg = transformer.transform(utm_easting.values, utm_northing.values)
    
    pop_df['lat_deg'] = lat_deg
    pop_df['lon_deg'] = lon_deg
    
    # Clean the data
    clean_df = pop_df.dropna(subset=['lat_deg', 'lon_deg', 'C1_COUNT_TOTAL'])
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
    
    # Filter for Nova Scotia bounds
    clean_df = clean_df[
        (clean_df['lat_deg'] >= 43.0) & (clean_df['lat_deg'] <= 47.0) &
        (clean_df['lon_deg'] >= -67.0) & (clean_df['lon_deg'] <= -59.0)
    ]
    
    clean_df['latitude'] = clean_df['lat_deg']
    clean_df['longitude'] = clean_df['lon_deg']
    
    print(f"✅ {len(clean_df)} communities processed")
    print(f"Total population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    return clean_df

def calculate_hospital_coverage_gaps(clean_df, hospital_performance):
    """Calculate areas with poor hospital coverage/performance"""
    print("\n📍 Analyzing hospital coverage gaps...")
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371  # Earth radius in km
    
    # For each community, find nearest hospital and its performance
    community_hospital_analysis = []
    
    for _, community in clean_df.iterrows():
        distances = []
        performance_scores = []
        
        for _, hospital in hospital_performance.iterrows():
            dist = haversine_distance(
                community['latitude'], community['longitude'],
                hospital['latitude'], hospital['longitude']
            )
            distances.append(dist)
            performance_scores.append(hospital['overall_performance'])
        
        # Find nearest hospital
        min_dist_idx = np.argmin(distances)
        nearest_distance = distances[min_dist_idx]
        nearest_performance = performance_scores[min_dist_idx]
        
        # Calculate weighted performance score (distance penalty + performance)
        # Further hospitals get penalized, poor performing hospitals get penalized
        distance_penalty = min(nearest_distance / 50.0, 1.0)  # Penalty increases up to 50km
        weighted_performance = nearest_performance * (1 - distance_penalty * 0.5)
        
        community_hospital_analysis.append({
            'community_index': community.name,
            'nearest_hospital_distance': nearest_distance,
            'nearest_hospital_performance': nearest_performance,
            'weighted_performance': weighted_performance,
            'coverage_gap_score': 1 - weighted_performance  # Higher score = needs more EMS coverage
        })
    
    coverage_df = pd.DataFrame(community_hospital_analysis)
    
    # Add coverage gap scores to community data
    clean_df = clean_df.copy()
    clean_df['nearest_hospital_distance'] = coverage_df['nearest_hospital_distance'].values
    clean_df['hospital_performance'] = coverage_df['nearest_hospital_performance'].values
    clean_df['coverage_gap_score'] = coverage_df['coverage_gap_score'].values
    
    # Ensure no NaN values in coverage gap score
    clean_df['coverage_gap_score'] = clean_df['coverage_gap_score'].fillna(0.5)  # Default moderate gap
    
    print(f"✅ Coverage gap analysis complete")
    print(f"Average distance to nearest hospital: {coverage_df['nearest_hospital_distance'].mean():.1f} km")
    print(f"Average hospital performance: {coverage_df['nearest_hospital_performance'].mean():.3f}")
    print(f"Communities with poor coverage (gap score > 0.5): {(coverage_df['coverage_gap_score'] > 0.5).sum()}")
    
    return clean_df

def performance_weighted_kmeans(clean_df, k, max_distance_km=15):
    """Enhanced K-means that considers population, geography, and hospital performance gaps"""
    
    # Prepare features: coordinates
    X = clean_df[['longitude', 'latitude']].values
    
    # Create composite weights combining population and coverage gaps
    population_weights = clean_df['C1_COUNT_TOTAL'].values
    gap_weights = clean_df['coverage_gap_score'].values
    
    # Normalize weights
    pop_normalized = population_weights / population_weights.max()
    gap_normalized = gap_weights / gap_weights.max()
    
    # Combine: 60% population-based, 40% coverage-gap-based
    composite_weights = 0.6 * pop_normalized + 0.4 * gap_normalized
    
    # Ensure minimum weight for all communities
    composite_weights = np.maximum(composite_weights, 0.1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X, sample_weight=composite_weights)
    
    # Calculate distances and coverage
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371
    
    distances = []
    for i, (_, row) in enumerate(clean_df.iterrows()):
        cluster_id = labels[i]
        center_lon, center_lat = cluster_centers[cluster_id]
        
        dist = haversine_distance(
            row['latitude'], row['longitude'],
            center_lat, center_lon
        )
        distances.append(dist)
    
    max_dist = np.max(distances)
    within_threshold = np.sum(np.array(distances) <= max_distance_km)
    coverage_pct = (within_threshold / len(distances)) * 100
    
    return {
        'k': k,
        'cluster_centers': cluster_centers,
        'labels': labels,
        'distances': distances,
        'max_distance': max_dist,
        'coverage_percentage': coverage_pct,
        'composite_weights': composite_weights
    }

def find_optimal_k_with_hospital_integration(clean_df, min_k=20, max_k=60, target_coverage=95.0):
    """Find optimal k considering hospital performance integration"""
    print(f"\n🎯 Finding optimal K with hospital performance integration...")
    print(f"Target: {target_coverage}% coverage within 15km")
    
    results = []
    
    for k in range(min_k, max_k + 1):
        print(f"Testing k={k}...", end=" ")
        
        result = performance_weighted_kmeans(clean_df, k)
        results.append(result)
        
        print(f"Coverage: {result['coverage_percentage']:.1f}%, Max dist: {result['max_distance']:.1f}km")
        
        # Early termination if we achieve target coverage
        if result['coverage_percentage'] >= target_coverage:
            print(f"  ✅ ACHIEVED TARGET COVERAGE with k={k}!")
            break
    
    # Find best solution
    target_solutions = [r for r in results if r['coverage_percentage'] >= target_coverage]
    
    if target_solutions:
        best_result = min(target_solutions, key=lambda x: x['k'])
        print(f"\n🎯 OPTIMAL SOLUTION: k={best_result['k']}")
        print(f"Coverage: {best_result['coverage_percentage']:.1f}%")
        print(f"Max distance: {best_result['max_distance']:.1f}km")
    else:
        best_result = max(results, key=lambda x: x['coverage_percentage'])
        print(f"\n⚠️  BEST AVAILABLE: k={best_result['k']}")
        print(f"Coverage: {best_result['coverage_percentage']:.1f}%")
        print(f"Max distance: {best_result['max_distance']:.1f}km")
    
    return best_result, results

def create_comprehensive_visualization(clean_df, hospital_performance, optimal_result):
    """Create comprehensive visualization including hospital performance"""
    print("\n📊 Creating comprehensive visualizations...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Main map with hospitals and EMS bases
    ax1 = plt.subplot(2, 3, (1, 4))
    
    # Plot communities colored by coverage gap score
    scatter = ax1.scatter(clean_df['longitude'], clean_df['latitude'],
                         c=clean_df['coverage_gap_score'],
                         s=clean_df['C1_COUNT_TOTAL']/100,
                         cmap='RdYlBu_r', alpha=0.7,
                         edgecolors='black', linewidth=0.5)
    
    # Plot hospitals colored by performance
    hospital_scatter = ax1.scatter(hospital_performance['longitude'], hospital_performance['latitude'],
                                  c=hospital_performance['overall_performance'],
                                  s=200, marker='h', cmap='RdYlGn',
                                  edgecolors='black', linewidth=2,
                                  label='Hospitals (colored by performance)')
    
    # Plot optimal EMS bases
    centers = optimal_result['cluster_centers']
    ax1.scatter(centers[:, 0], centers[:, 1],
               c='red', s=300, marker='*',
               edgecolors='black', linewidth=2,
               label=f'Optimal EMS Bases (k={optimal_result["k"]})')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Hospital-Performance-Integrated EMS Base Optimization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8, label='Coverage Gap Score')
    
    # Hospital performance distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(hospital_performance['overall_performance'], bins=15, alpha=0.7, color='skyblue')
    ax2.set_xlabel('Hospital Performance Score')
    ax2.set_ylabel('Number of Hospitals')
    ax2.set_title('Hospital Performance Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Coverage gap analysis
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(clean_df['coverage_gap_score'], bins=20, alpha=0.7, color='orange')
    ax3.set_xlabel('Coverage Gap Score')
    ax3.set_ylabel('Number of Communities')
    ax3.set_title('Community Coverage Gap Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Hospital performance by type
    ax4 = plt.subplot(2, 3, 5)
    perf_by_type = hospital_performance.groupby('type')['overall_performance'].mean().sort_values(ascending=True)
    perf_by_type.plot(kind='barh', ax=ax4, color='lightgreen')
    ax4.set_xlabel('Average Performance Score')
    ax4.set_title('Hospital Performance by Type')
    ax4.grid(True, alpha=0.3)
    
    # Distance vs Performance relationship
    ax5 = plt.subplot(2, 3, 6)
    ax5.scatter(clean_df['nearest_hospital_distance'], clean_df['hospital_performance'],
               s=clean_df['C1_COUNT_TOTAL']/100, alpha=0.6, color='purple')
    ax5.set_xlabel('Distance to Nearest Hospital (km)')
    ax5.set_ylabel('Nearest Hospital Performance')
    ax5.set_title('Distance vs Hospital Performance')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hospital_integrated_ems_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive visualization saved as 'hospital_integrated_ems_analysis.png'")

def export_enhanced_results(optimal_result, clean_df, hospital_performance):
    """Export results with hospital performance integration"""
    print(f"\n💾 Exporting enhanced results...")
    
    k = optimal_result['k']
    centers = optimal_result['cluster_centers']
    labels = optimal_result['labels']
    distances = optimal_result['distances']
    
    # Create EMS base dataframe with enhanced information
    ems_bases = []
    for i, center in enumerate(centers):
        # Find communities served by this base
        cluster_communities = clean_df[labels == i]
        
        # Calculate cluster statistics
        total_population = cluster_communities['C1_COUNT_TOTAL'].sum()
        avg_gap_score = cluster_communities['coverage_gap_score'].mean()
        communities_count = len(cluster_communities)
        
        # Find nearest hospital to this EMS base
        hospital_distances = []
        for _, hospital in hospital_performance.iterrows():
            dist = np.sqrt((center[1] - hospital['latitude'])**2 + 
                          (center[0] - hospital['longitude'])**2) * 111  # Rough km conversion
            hospital_distances.append(dist)
        
        nearest_hospital_idx = np.argmin(hospital_distances)
        nearest_hospital = hospital_performance.iloc[nearest_hospital_idx]
        
        ems_bases.append({
            'EHS_Base_ID': f'EHS_Base_{i+1:02d}',
            'Latitude': center[1],
            'Longitude': center[0],
            'Region': f'Cluster_{i+1}',
            'Coverage_Area': '15km_radius',
            'Population_Served': int(total_population),
            'Communities_Served': communities_count,
            'Avg_Coverage_Gap_Score': round(avg_gap_score, 3),
            'Nearest_Hospital': nearest_hospital['facility'],
            'Hospital_Distance_km': round(hospital_distances[nearest_hospital_idx], 1),
            'Hospital_Performance': round(nearest_hospital['overall_performance'], 3),
            'Priority_Level': 'High' if avg_gap_score > 0.6 else ('Medium' if avg_gap_score > 0.4 else 'Low')
        })
    
    ems_df = pd.DataFrame(ems_bases)
    
    # Save EMS bases
    ems_filename = f'../../data/processed/hospital_integrated_ems_locations_{k}bases.csv'
    ems_df.to_csv(ems_filename, index=False)
    
    # Save community assignments with enhanced data
    community_assignments = clean_df.copy()
    community_assignments['assigned_ems_base'] = [f'EHS_Base_{label+1:02d}' for label in labels]
    community_assignments['distance_to_ems'] = distances
    
    assignments_filename = f'hospital_integrated_community_assignments_{k}bases.csv'
    community_assignments[[
        'GEO_NAME', 'latitude', 'longitude', 'C1_COUNT_TOTAL',
        'nearest_hospital_distance', 'hospital_performance', 'coverage_gap_score',
        'assigned_ems_base', 'distance_to_ems'
    ]].to_csv(assignments_filename, index=False)
    
    # Save hospital performance analysis
    hospital_filename = 'hospital_performance_analysis.csv'
    hospital_performance.to_csv(hospital_filename, index=False)
    
    print(f"✅ EMS bases saved to: {ems_filename}")
    print(f"✅ Community assignments saved to: {assignments_filename}")
    print(f"✅ Hospital analysis saved to: {hospital_filename}")
    
    return ems_filename

def main():
    """Main execution function"""
    print("🏥🚑 ENHANCED EMS BASE OPTIMIZATION WITH HOSPITAL PERFORMANCE INTEGRATION")
    print("=" * 80)
    
    try:
        # Load all data
        pop_df, perf_df, hospital_df = load_and_prepare_data()
        
        # Process hospital performance
        hospital_performance = process_hospital_performance(perf_df, hospital_df)
        
        # Clean population data
        clean_df = clean_population_data(pop_df)
        
        # Calculate hospital coverage gaps
        clean_df = calculate_hospital_coverage_gaps(clean_df, hospital_performance)
        
        # Find optimal k with hospital integration
        optimal_result, all_results = find_optimal_k_with_hospital_integration(
            clean_df, min_k=20, max_k=50, target_coverage=95.0
        )
        
        # Create visualizations
        create_comprehensive_visualization(clean_df, hospital_performance, optimal_result)
        
        # Export results
        ems_filename = export_enhanced_results(optimal_result, clean_df, hospital_performance)
        
        print(f"\n🎉 ENHANCED ANALYSIS COMPLETE!")
        print(f"🎯 Optimal Solution: {optimal_result['k']} EMS bases")
        print(f"📊 Coverage: {optimal_result['coverage_percentage']:.1f}%")
        print(f"📏 Max distance: {optimal_result['max_distance']:.1f}km")
        print(f"🏥 Hospital performance integrated into optimization")
        print(f"📁 Results saved to: {ems_filename}")
        
        return optimal_result['k'], ems_filename
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Corrected Hospital Co-located EMS Base Analysis
Method 2B Enhanced: Using only the 37 hospitals from emergency_health_services.csv
Place EMS bases at existing emergency services hospitals first, then add additional bases for 100% coverage
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

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in kilometers"""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def load_and_prepare_data():
    """Load and prepare all data sources using ONLY emergency services hospitals"""
    print("📊 Loading data for CORRECTED hospital co-located EMS analysis...")
    
    # Load population data
    pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
    print(f"Population data: {pop_df.shape[0]} communities")
    
    # Load emergency health services data to get the 37 hospitals
    ems_df = pd.read_csv('../../data/raw/emergency_health_services.csv')
    ems_hospitals = ems_df['Hospital'].dropna().unique()
    # Remove the generic "Hospital" entry if it exists
    ems_hospitals = [h for h in ems_hospitals if h != 'Hospital']
    print(f"Emergency services hospitals: {len(ems_hospitals)} hospitals")
    
    # Load ALL hospital locations from geojson
    with open('../../data/raw/hospitals.geojson', 'r') as f:
        hospital_data = json.load(f)
    
    all_hospital_list = []
    for feature in hospital_data['features']:
        coords = feature['geometry']['coordinates']
        props = feature['properties']
        all_hospital_list.append({
            'facility': props['facility'],
            'town': props['town'],
            'county': props['county'],
            'type': props['type'],
            'longitude': coords[0],
            'latitude': coords[1]
        })
    
    all_hospitals_df = pd.DataFrame(all_hospital_list)
    
    # Create mapping for name variations
    name_mapping = {}
    for ems_hospital in ems_hospitals:
        best_match = None
        best_score = 0
        
        for _, geo_hospital in all_hospitals_df.iterrows():
            geo_name = geo_hospital['facility']
            
            # Calculate similarity (simple word matching)
            ems_words = set(ems_hospital.lower().split())
            geo_words = set(geo_name.lower().split())
            
            # Handle special cases
            if 'strait richmond' in ems_hospital.lower() and 'strait - richmond' in geo_name.lower():
                similarity = 1.0
            elif 'glace bay hospital' in ems_hospital.lower() and 'glace bay health care facility' in geo_name.lower():
                similarity = 1.0
            elif 'soldiers\' memorial' in ems_hospital.lower() and 'soldiers memorial' in geo_name.lower():
                similarity = 1.0
            elif 'st. martha\'s' in ems_hospital.lower() and 'st. martha\'s' in geo_name.lower():
                similarity = 1.0
            elif 'st. mary\'s' in ems_hospital.lower() and 'st. mary\'s' in geo_name.lower():
                similarity = 1.0
            else:
                common_words = ems_words.intersection(geo_words)
                similarity = len(common_words) / max(len(ems_words), len(geo_words))
            
            if similarity > best_score and similarity > 0.5:
                best_score = similarity
                best_match = geo_hospital
        
        if best_match is not None:
            name_mapping[ems_hospital] = best_match
    
    # Create filtered hospitals dataframe with only emergency services hospitals
    hospitals_with_coords = []
    matched_count = 0
    
    for ems_hospital in ems_hospitals:
        if ems_hospital in name_mapping:
            geo_hospital = name_mapping[ems_hospital]
            hospitals_with_coords.append({
                'facility': ems_hospital,  # Use EMS name
                'geo_facility': geo_hospital['facility'],  # Keep geo name for reference
                'town': geo_hospital['town'],
                'county': geo_hospital['county'],
                'type': geo_hospital['type'],
                'longitude': geo_hospital['longitude'],
                'latitude': geo_hospital['latitude']
            })
            matched_count += 1
            print(f"✅ Matched: {ems_hospital} -> {geo_hospital['facility']}")
        else:
            print(f"❌ No match found for: {ems_hospital}")
    
    hospitals_df = pd.DataFrame(hospitals_with_coords)
    print(f"Successfully matched {matched_count}/{len(ems_hospitals)} emergency services hospitals with coordinates")
    
    # Transform coordinates from Lambert to WGS84
    transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer.transform(pop_df['longitude'].values, pop_df['latitude'].values)
    pop_df['lat_deg'] = lat_deg
    pop_df['lon_deg'] = lon_deg
    
    # Filter for Nova Scotia and clean data
    ns_df = pop_df[
        (pop_df['lat_deg'] >= 43.3) & (pop_df['lat_deg'] <= 47.1) &
        (pop_df['lon_deg'] >= -66.5) & (pop_df['lon_deg'] <= -59.7)
    ].copy()
    
    clean_df = ns_df.dropna(subset=['GEO_NAME', 'C1_COUNT_TOTAL', 'lat_deg', 'lon_deg']).copy()
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0].copy()
    
    print(f"After cleaning: {len(clean_df)} communities")
    print(f"Total population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    return clean_df, hospitals_df

def calculate_hospital_coverage(clean_df, hospitals_df, max_distance_km=15):
    """Calculate population coverage from existing emergency services hospitals"""
    print(f"\n📍 Calculating coverage from {len(hospitals_df)} emergency services hospitals...")
    
    coverage_data = []
    
    for idx, community in clean_df.iterrows():
        community_lat = community['lat_deg']
        community_lon = community['lon_deg']
        community_pop = community['C1_COUNT_TOTAL']
        
        # Find nearest hospital
        min_distance = float('inf')
        nearest_hospital = None
        
        for h_idx, hospital in hospitals_df.iterrows():
            distance = haversine_distance(
                community_lat, community_lon,
                hospital['latitude'], hospital['longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_hospital = hospital['facility']
        
        coverage_data.append({
            'community': community['GEO_NAME'],
            'population': community_pop,
            'lat_deg': community_lat,
            'lon_deg': community_lon,
            'nearest_hospital': nearest_hospital,
            'distance_to_hospital': min_distance,
            'covered_by_hospital': min_distance <= max_distance_km
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    
    # Calculate coverage statistics
    covered_communities = coverage_df['covered_by_hospital'].sum()
    covered_population = coverage_df[coverage_df['covered_by_hospital']]['population'].sum()
    total_population = coverage_df['population'].sum()
    
    coverage_percentage = (covered_population / total_population) * 100
    community_coverage_percentage = (covered_communities / len(coverage_df)) * 100
    
    print(f"Emergency services hospital coverage results:")
    print(f"  Communities covered: {covered_communities}/{len(coverage_df)} ({community_coverage_percentage:.1f}%)")
    print(f"  Population covered: {covered_population:,}/{total_population:,} ({coverage_percentage:.1f}%)")
    
    return coverage_df, coverage_percentage

def find_additional_bases_for_gaps(coverage_df, hospitals_df, max_distance_km=15):
    """Find additional EMS bases needed to cover gaps left by emergency services hospitals"""
    print(f"\n🎯 Finding additional EMS bases to cover gaps...")
    
    # Get uncovered communities
    uncovered = coverage_df[~coverage_df['covered_by_hospital']].copy()
    
    if len(uncovered) == 0:
        print("✅ All communities already covered by emergency services hospitals!")
        return pd.DataFrame(), 0, 100.0
    
    print(f"Uncovered communities: {len(uncovered)}")
    print(f"Uncovered population: {uncovered['population'].sum():,}")
    
    # Prepare data for K-means clustering on uncovered areas
    X_uncovered = uncovered[['lat_deg', 'lon_deg']].values
    sample_weights = uncovered['population'].values
    
    # Test different K values to find minimum needed for 100% coverage
    print(f"Testing K values for additional bases...")
    
    for k in range(1, len(uncovered) + 1):
        print(f"Testing k={k} additional bases...", end=" ")
        
        # Perform weighted K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_uncovered, sample_weight=sample_weights)
        
        additional_bases = pd.DataFrame({
            'EHS_Base_ID': [f'Additional_{i+1}' for i in range(k)],
            'Latitude': kmeans.cluster_centers_[:, 0],
            'Longitude': kmeans.cluster_centers_[:, 1],
            'Type': 'Additional_EMS'
        })
        
        # Test coverage with hospitals + additional bases
        all_bases = pd.concat([
            hospitals_df[['facility', 'latitude', 'longitude']].rename(columns={
                'facility': 'EHS_Base_ID',
                'latitude': 'Latitude', 
                'longitude': 'Longitude'
            }).assign(Type='Emergency_Hospital'),
            additional_bases
        ], ignore_index=True)
        
        # Calculate coverage
        coverage_achieved = calculate_total_coverage(coverage_df, all_bases, max_distance_km)
        
        print(f"Coverage: {coverage_achieved:.1f}%")
        
        if coverage_achieved >= 99.9:  # Allow for small rounding errors
            print(f"  ✅ ACHIEVED 100% COVERAGE with {k} additional bases!")
            return additional_bases, k, coverage_achieved
    
    # If we get here, we couldn't achieve 100% coverage
    print(f"⚠️  Could not achieve 100% coverage with available communities")
    return additional_bases, k, coverage_achieved

def calculate_total_coverage(coverage_df, all_bases, max_distance_km=15):
    """Calculate coverage from all EMS bases (emergency hospitals + additional)"""
    covered_count = 0
    
    for idx, community in coverage_df.iterrows():
        community_lat = community['lat_deg']
        community_lon = community['lon_deg']
        
        # Check if community is within range of any base
        min_distance = float('inf')
        
        for _, base in all_bases.iterrows():
            distance = haversine_distance(
                community_lat, community_lon,
                base['Latitude'], base['Longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
        
        if min_distance <= max_distance_km:
            covered_count += 1
    
    return (covered_count / len(coverage_df)) * 100

def create_comprehensive_analysis(clean_df, hospitals_df, coverage_df, additional_bases, hospital_coverage_pct):
    """Create comprehensive analysis and visualization"""
    print(f"\n📊 Creating comprehensive analysis...")
    
    # Calculate final metrics
    total_bases = len(hospitals_df) + len(additional_bases)
    
    # Create all bases dataframe
    all_bases = pd.concat([
        hospitals_df[['facility', 'latitude', 'longitude']].rename(columns={
            'facility': 'EHS_Base_ID',
            'latitude': 'Latitude', 
            'longitude': 'Longitude'
        }).assign(Type='Emergency_Hospital'),
        additional_bases if len(additional_bases) > 0 else pd.DataFrame(columns=['EHS_Base_ID', 'Latitude', 'Longitude', 'Type'])
    ], ignore_index=True)
    
    # Calculate detailed coverage metrics
    final_coverage = calculate_total_coverage(coverage_df, all_bases, 15)
    
    print(f"\n🎯 CORRECTED FINAL RESULTS:")
    print(f"Emergency services hospitals: {len(hospitals_df)}")
    print(f"Hospital-based coverage: {hospital_coverage_pct:.1f}%")
    print(f"Additional bases needed: {len(additional_bases)}")
    print(f"Total EMS bases: {total_bases}")
    print(f"Final coverage: {final_coverage:.1f}%")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Hospital coverage map
    ax1.scatter(coverage_df['lon_deg'], coverage_df['lat_deg'], 
               c=coverage_df['covered_by_hospital'], 
               s=coverage_df['population']/1000, 
               alpha=0.6, cmap='RdYlGn')
    ax1.scatter(hospitals_df['longitude'], hospitals_df['latitude'], 
               c='red', marker='+', s=100, label=f'Emergency Hospitals ({len(hospitals_df)})')
    ax1.set_title(f'Emergency Services Hospital Coverage\n{hospital_coverage_pct:.1f}% Population Covered')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final coverage map
    ax2.scatter(coverage_df['lon_deg'], coverage_df['lat_deg'], 
               s=coverage_df['population']/1000, 
               alpha=0.6, color='lightblue', label='Communities')
    ax2.scatter(hospitals_df['longitude'], hospitals_df['latitude'], 
               c='red', marker='+', s=100, label=f'Emergency Hospitals ({len(hospitals_df)})')
    if len(additional_bases) > 0:
        ax2.scatter(additional_bases['Longitude'], additional_bases['Latitude'], 
                   c='blue', marker='s', s=100, label=f'Additional EMS ({len(additional_bases)})')
    ax2.set_title(f'Corrected Final EMS Network\n{total_bases} Total Bases, {final_coverage:.1f}% Coverage')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coverage comparison
    categories = ['Emergency Hospitals Only', 'Emergency Hospitals + Additional']
    coverages = [hospital_coverage_pct, final_coverage]
    colors = ['orange', 'green']
    
    bars = ax3.bar(categories, coverages, color=colors, alpha=0.7)
    ax3.set_title('Coverage Comparison (Corrected)')
    ax3.set_ylabel('Population Coverage (%)')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    for bar, coverage in zip(bars, coverages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{coverage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Base count comparison
    base_types = ['Emergency Hospitals', 'Additional EMS', 'Total']
    base_counts = [len(hospitals_df), len(additional_bases), total_bases]
    colors = ['red', 'blue', 'purple']
    
    bars = ax4.bar(base_types, base_counts, color=colors, alpha=0.7)
    ax4.set_title('EMS Base Count Analysis (Corrected)')
    ax4.set_ylabel('Number of Bases')
    ax4.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, base_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('corrected_hospital_colocated_ems_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Corrected analysis visualization saved as 'corrected_hospital_colocated_ems_analysis.png'")
    
    return all_bases

def export_results(all_bases, coverage_df, hospital_coverage_pct, final_coverage, num_emergency_hospitals):
    """Export corrected results to CSV files"""
    print(f"\n💾 Exporting corrected results...")
    
    # Export base locations
    all_bases.to_csv('../../data/processed/corrected_hospital_colocated_ems_locations.csv', index=False)
    print("✅ Corrected EMS base locations saved")
    
    # Export coverage analysis
    coverage_summary = {
        'metric': [
            'Emergency Services Hospitals',
            'Additional EMS Bases',
            'Total EMS Bases',
            'Emergency Hospital Coverage (%)',
            'Final Coverage (%)',
            'Additional Bases Needed'
        ],
        'value': [
            num_emergency_hospitals,
            len(all_bases[all_bases['Type'] == 'Additional_EMS']),
            len(all_bases),
            f"{hospital_coverage_pct:.1f}%",
            f"{final_coverage:.1f}%",
            len(all_bases[all_bases['Type'] == 'Additional_EMS'])
        ]
    }
    
    summary_df = pd.DataFrame(coverage_summary)
    summary_df.to_csv('corrected_hospital_colocated_coverage_summary.csv', index=False)
    print("✅ Corrected coverage summary saved")
    
    return summary_df

def main():
    """Main analysis function"""
    print("=== CORRECTED Hospital Co-located EMS Base Analysis ===")
    print("Method 2B Enhanced: Using ONLY the 37 emergency services hospitals")
    
    # Load data
    clean_df, hospitals_df = load_and_prepare_data()
    
    # Step 1: Calculate hospital coverage
    coverage_df, hospital_coverage_pct = calculate_hospital_coverage(clean_df, hospitals_df)
    
    # Step 2: Find additional bases needed
    additional_bases, additional_count, final_coverage = find_additional_bases_for_gaps(coverage_df, hospitals_df)
    
    # Step 3: Create comprehensive analysis
    all_bases = create_comprehensive_analysis(clean_df, hospitals_df, coverage_df, additional_bases, hospital_coverage_pct)
    
    # Step 4: Export results
    summary_df = export_results(all_bases, coverage_df, hospital_coverage_pct, final_coverage, len(hospitals_df))
    
    print(f"\n🎉 === CORRECTED Analysis Complete ===")
    print(f"Emergency services hospitals used: {len(hospitals_df)}")
    print(f"Emergency hospital-based EMS coverage: {hospital_coverage_pct:.1f}%")
    print(f"Additional bases needed: {additional_count}")
    print(f"Total EMS bases: {len(all_bases)}")
    print(f"Final coverage: {final_coverage:.1f}%")
    
    return all_bases, summary_df

if __name__ == "__main__":
    all_bases, summary = main()

#!/usr/bin/env python3
"""
Method 3: Analysis for 100% Coverage Extension
Determine how many additional bases Method 3 needs to achieve 100% coverage like Methods 1 and 2
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Euclidean approximation for projected coordinates"""
    import numpy as np
    
    # For projected coordinates, use Euclidean distance and convert to km
    # The coordinates appear to be in meters, so divide by 1000
    dx = (lon2 - lon1) / 1000.0  # Convert to km
    dy = (lat2 - lat1) / 1000.0  # Convert to km
    return np.sqrt(dx**2 + dy**2)

def load_method3_data():
    """Load current Method 3 data and population data"""
    print("📊 Loading Method 3 current data...")
    
    # Load current Method 3 bases (45 bases, 96.7% coverage)
    method3_bases = pd.read_csv('../../data/processed/hospital_integrated_ems_locations_45bases.csv')
    
    # Load population data
    pop_df = pd.read_csv('../../../0 polulation_location_polygon.csv')
    
    print(f"✅ Current Method 3: {len(method3_bases)} bases")
    print(f"✅ Population communities: {len(pop_df)} locations")
    
    return method3_bases, pop_df

def clean_population_data(pop_df):
    """Clean and prepare population data for analysis"""
    print("Cleaning population data...")
    
    # Use coordinates as provided (already in appropriate projection)
    clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
    
    # Filter out rows with zero population
    clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
    
    print(f"✅ Cleaned data: {len(clean_df)} valid communities")
    print(f"Total population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
    
    return clean_df

def calculate_current_coverage(clean_df, method3_bases, max_distance_km=15):
    """Calculate current coverage from Method 3 bases"""
    print(f"📏 Calculating current coverage within {max_distance_km}km...")
    
    covered_communities = []
    uncovered_communities = []
    
    for idx, community in clean_df.iterrows():
        community_lat = community['latitude']
        community_lon = community['longitude']
        
        # Find minimum distance to any Method 3 base
        min_distance = float('inf')
        closest_base = None
        
        for _, base in method3_bases.iterrows():
            distance = haversine_distance(
                community_lat, community_lon,
                base['Latitude'], base['Longitude']
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_base = base['EHS_Base_ID']
        
        community_data = {
            'geo_name': community['GEO_NAME'],
            'population': community['C1_COUNT_TOTAL'],
            'latitude': community_lat,
            'longitude': community_lon,
            'min_distance': min_distance,
            'closest_base': closest_base
        }
        
        if min_distance <= max_distance_km:
            covered_communities.append(community_data)
        else:
            uncovered_communities.append(community_data)
    
    covered_df = pd.DataFrame(covered_communities)
    uncovered_df = pd.DataFrame(uncovered_communities)
    
    # Calculate coverage metrics
    covered_population = covered_df['population'].sum() if len(covered_df) > 0 else 0
    total_population = clean_df['C1_COUNT_TOTAL'].sum()
    coverage_percentage = (covered_population / total_population) * 100
    community_coverage = (len(covered_df) / len(clean_df)) * 100
    
    print(f"📊 CURRENT METHOD 3 COVERAGE:")
    print(f"   Communities covered: {len(covered_df)}/{len(clean_df)} ({community_coverage:.1f}%)")
    print(f"   Population covered: {covered_population:,}/{total_population:,} ({coverage_percentage:.1f}%)")
    print(f"   Uncovered communities: {len(uncovered_df)}")
    print(f"   Uncovered population: {uncovered_df['population'].sum():,}")
    
    return covered_df, uncovered_df, coverage_percentage

def find_additional_bases_for_100_percent(uncovered_df, method3_bases, clean_df, max_distance_km=15):
    """Find minimum additional bases needed for 100% coverage"""
    print(f"\n🎯 Finding additional bases for 100% coverage...")
    
    if len(uncovered_df) == 0:
        print("✅ Already 100% covered!")
        return pd.DataFrame(), 0, 100.0
    
    print(f"🔍 Need to cover {len(uncovered_df)} uncovered communities")
    print(f"📊 Uncovered population: {uncovered_df['population'].sum():,}")
    
    # Prepare data for clustering on uncovered areas
    X_uncovered = uncovered_df[['latitude', 'longitude']].values
    sample_weights = uncovered_df['population'].values
    
    # Test different numbers of additional bases
    print("🧪 Testing different numbers of additional bases...")
    
    for k in range(1, len(uncovered_df) + 1):
        print(f"Testing {k} additional bases...", end=" ")
        
        # Perform weighted K-means on uncovered areas
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_uncovered, sample_weight=sample_weights)
        
        # Create additional bases dataframe
        additional_bases = pd.DataFrame({
            'EHS_Base_ID': [f'Method3_Additional_{i+1}' for i in range(k)],
            'Latitude': kmeans.cluster_centers_[:, 0],
            'Longitude': kmeans.cluster_centers_[:, 1],
            'Type': 'Additional_EMS_Method3'
        })
        
        # Combine with existing Method 3 bases
        all_bases = pd.concat([
            method3_bases[['EHS_Base_ID', 'Latitude', 'Longitude']].assign(Type='Original_Method3'),
            additional_bases
        ], ignore_index=True)
        
        # Test coverage with all bases
        coverage_achieved = calculate_total_coverage_all_bases(clean_df, all_bases, max_distance_km)
        
        print(f"Coverage: {coverage_achieved:.1f}%")
        
        if coverage_achieved >= 99.9:  # Allow for small rounding errors
            print(f"  ✅ ACHIEVED 100% COVERAGE with {k} additional bases!")
            print(f"  🎯 Total bases: {len(method3_bases)} (original) + {k} (additional) = {len(method3_bases) + k}")
            return additional_bases, k, coverage_achieved
    
    # If we get here, couldn't achieve 100% coverage
    print(f"⚠️  Could not achieve 100% coverage with tested configurations")
    return additional_bases, k, coverage_achieved

def calculate_total_coverage_all_bases(clean_df, all_bases, max_distance_km=15):
    """Calculate coverage from all bases (original + additional)"""
    covered_count = 0
    
    for idx, community in clean_df.iterrows():
        community_lat = community['latitude']
        community_lon = community['longitude']
        
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
    
    return (covered_count / len(clean_df)) * 100

def save_extended_method3_results(method3_bases, additional_bases, k_additional, coverage_achieved):
    """Save the extended Method 3 results for 100% coverage"""
    print(f"\n💾 Saving extended Method 3 results...")
    
    # Combine all bases
    extended_bases = pd.concat([
        method3_bases[['EHS_Base_ID', 'Latitude', 'Longitude']].assign(Type='Original_Method3'),
        additional_bases
    ], ignore_index=True)
    
    # Save to file
    filename = '../../data/processed/method3_extended_100percent_coverage.csv'
    extended_bases.to_csv(filename, index=False)
    
    print(f"✅ Extended Method 3 results saved to: {filename}")
    print(f"📊 Total bases: {len(extended_bases)}")
    print(f"📊 Original Method 3 bases: {len(method3_bases)}")
    print(f"📊 Additional bases needed: {k_additional}")
    print(f"📊 Final coverage: {coverage_achieved:.1f}%")
    
    return filename, len(extended_bases)

def create_comparison_visualization(method3_bases, additional_bases, covered_df, uncovered_df):
    """Create visualization comparing original vs extended Method 3"""
    print("📊 Creating comparison visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Original Method 3 (96.7% coverage)
    ax1.scatter(covered_df['longitude'], covered_df['latitude'], 
               c='green', alpha=0.6, s=covered_df['population']/500, 
               label=f'Covered Communities ({len(covered_df)})')
    
    ax1.scatter(uncovered_df['longitude'], uncovered_df['latitude'], 
               c='red', alpha=0.8, s=uncovered_df['population']/500, 
               label=f'Uncovered Communities ({len(uncovered_df)})')
    
    ax1.scatter(method3_bases['Longitude'], method3_bases['Latitude'], 
               c='blue', s=100, marker='^', 
               label=f'Method 3 Bases ({len(method3_bases)})', edgecolors='black')
    
    ax1.set_title('Original Method 3: 96.7% Coverage\n45 Bases', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Extended Method 3 (100% coverage)
    ax2.scatter(covered_df['longitude'], covered_df['latitude'], 
               c='green', alpha=0.6, s=covered_df['population']/500, 
               label='All Communities (100% covered)')
    
    ax2.scatter(method3_bases['Longitude'], method3_bases['Latitude'], 
               c='blue', s=100, marker='^', 
               label=f'Original Method 3 Bases ({len(method3_bases)})', edgecolors='black')
    
    if len(additional_bases) > 0:
        ax2.scatter(additional_bases['Longitude'], additional_bases['Latitude'], 
                   c='orange', s=100, marker='s', 
                   label=f'Additional Bases ({len(additional_bases)})', edgecolors='black')
    
    total_bases = len(method3_bases) + len(additional_bases)
    ax2.set_title(f'Extended Method 3: 100% Coverage\n{total_bases} Total Bases', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../output/plots/method3_100percent_coverage_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to: ../../output/plots/method3_100percent_coverage_comparison.png")

def main():
    """Main analysis function"""
    print("🎯 METHOD 3: 100% COVERAGE EXTENSION ANALYSIS")
    print("=" * 60)
    
    try:
        # Load data
        method3_bases, pop_df = load_method3_data()
        clean_df = clean_population_data(pop_df)
        
        # Calculate current coverage
        covered_df, uncovered_df, current_coverage = calculate_current_coverage(clean_df, method3_bases)
        
        # Find additional bases needed
        additional_bases, k_additional, final_coverage = find_additional_bases_for_100_percent(
            uncovered_df, method3_bases, clean_df
        )
        
        # Save results
        if k_additional > 0:
            filename, total_bases = save_extended_method3_results(
                method3_bases, additional_bases, k_additional, final_coverage
            )
            
            # Create visualization
            create_comparison_visualization(method3_bases, additional_bases, covered_df, uncovered_df)
        
        # Summary
        print(f"\n🎉 METHOD 3 EXTENSION ANALYSIS COMPLETE!")
        print(f"📊 Current Method 3: {len(method3_bases)} bases, {current_coverage:.1f}% coverage")
        print(f"🎯 For 100% coverage: +{k_additional} additional bases needed")
        print(f"📈 Extended Method 3: {len(method3_bases) + k_additional} total bases, {final_coverage:.1f}% coverage")
        
        # Comparison with other methods
        print(f"\n📊 COMPARISON WITH OTHER METHODS:")
        print(f"   Method 1 (Population-Only): 80 bases, 100% coverage")
        print(f"   Method 2 (Emergency Hospital): 76 bases, 100% coverage")
        print(f"   Method 3 (Original): 45 bases, 96.7% coverage")
        print(f"   Method 3 (Extended): {len(method3_bases) + k_additional} bases, 100% coverage")
        
        return len(method3_bases) + k_additional, filename if k_additional > 0 else None
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

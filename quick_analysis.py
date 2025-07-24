import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and clean data
pop_df = pd.read_csv('0 polulation_location_polygon.csv')
clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
clean_df = clean_df[
    (clean_df['latitude'] >= 43.0) & (clean_df['latitude'] <= 47.0) &
    (clean_df['longitude'] >= -67.0) & (clean_df['longitude'] <= -59.0)
]

# Clustering with k=5 (good default for Nova Scotia)
X = clean_df[['longitude', 'latitude']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
weights = clean_df['C1_COUNT_TOTAL'].values

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled, sample_weight=weights)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Create results
ems_locations = []
for i, center in enumerate(centers):
    ems_locations.append({
        'EHS_Base_ID': f'EHS-{i+1}',
        'Longitude': center[0],
        'Latitude': center[1]
    })

# Save files
pd.DataFrame(ems_locations).to_csv('proposed_ems_locations.csv', index=False)

clean_df['cluster'] = kmeans.labels_
output_cols = ['GEO_NAME', 'latitude', 'longitude', 'C1_COUNT_TOTAL', 'cluster']
clean_df[output_cols].to_csv('community_assignments.csv', index=False)

print("Analysis complete!")
print(f"Communities: {len(clean_df)}")
print(f"Population: {clean_df['C1_COUNT_TOTAL'].sum():,}")
print("EHS Locations:")
for loc in ems_locations:
    print(f"  {loc['EHS_Base_ID']}: {loc['Latitude']:.6f}, {loc['Longitude']:.6f}")

# Nova Scotia EHS Base Location Analysis

This project analyzes Nova Scotia population and hospital data to determine optimal Emergency Health Services (EHS) base locations using K-Means clustering with population-weighted sampling and 15-minute coverage optimization.

## 🎯 Project Objective

Find optimal EHS base locations in Nova Scotia that:
- **Ensure 15-minute coverage** for all 95 communities
- **Maximize coverage** for areas with higher population density
- **Minimize average response times** across the province
- **Use population weighting** in K-Means clustering algorithm
- Consider existing hospital infrastructure

## 📊 Key Results

### Updated Analysis (15-Minute Coverage)
- **Communities Analyzed**: 95 Nova Scotia communities
- **Total Population**: 923,598 residents
- **Optimal Configuration**: 12 EHS bases (updated from 5)
- **Coverage Achievement**: 100% of communities within 15km
- **Method**: Population-weighted K-Means with distance optimization

### EHS Base Locations
| Base | Region | Coverage Area |
|------|--------|---------------|
| EHS-01 | Halifax Metro East | Halifax, Dartmouth |
| EHS-02 | Halifax Metro West | Bedford, Sackville |
| EHS-03 | Kentville/Valley | Kings County |
| EHS-04 | Bridgewater | South Shore |
| EHS-05 | Yarmouth | Southwest Nova |
| EHS-06 | Digby | Bay of Fundy |
| EHS-07 | Truro | Central Nova |
| EHS-08 | New Glasgow | Pictou County |
| EHS-09 | Antigonish | Eastern Mainland |
| EHS-10 | Sydney | Cape Breton Regional |
| EHS-11 | North Sydney | Northern Cape Breton |
| EHS-12 | Glace Bay | Eastern Cape Breton |

## Project Structure

```
📁 Analysis/
├── 📊 0 polulation_location_polygon.csv    # Nova Scotia population data
├── 📊 1 Hospitals.geojson                  # Hospital location data
├── 📓 EMS_Location_Analysis.ipynb          # Main analysis notebook
├── 🐍 ems_location_analysis.py             # Python script version
├── 📋 requirements.txt                     # Required Python packages
├── 📝 README.md                           # This file
└── 📄 Capstone Project.docx               # Project documentation
```

## Objective

Find optimal EHS base locations in Nova Scotia that:
- Maximize coverage for areas with higher population density
- Minimize average response times across the province
- Consider existing hospital infrastructure
- Use data-driven clustering methodology

## Methodology

1. **Data Loading**: Load population data (95+ communities) and hospital locations (47 facilities)
2. **Data Cleaning**: Remove invalid coordinates and zero-population areas
3. **Population Weighting**: Use population as sample weights in K-Means clustering
4. **Cluster Optimization**: Find optimal number of clusters using silhouette analysis
5. **Coverage Analysis**: Calculate distance and response time metrics
6. **Visualization**: Create maps and charts showing optimal locations

## Installation and Setup

### Option 1: Using Jupyter Notebook (Recommended)

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook EMS_Location_Analysis.ipynb
   ```

3. **Run all cells** to perform the complete analysis

### Option 2: Using Python Script

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis script:**
   ```bash
   python ems_location_analysis.py
   ```

## Required Packages

- pandas >= 1.3.0
- geopandas >= 0.10.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- folium (for interactive maps)
- shapely >= 1.8.0
- fiona >= 1.8.0

## Input Data

### Population Data (`0 polulation_location_polygon.csv`)
- Community-level population data for Nova Scotia
- Contains coordinates (latitude/longitude) for each community
- Population counts used as sample weights in clustering
- Includes geometry data for spatial analysis

### Hospital Data (`1 Hospitals.geojson`)
- Location data for all hospitals in Nova Scotia
- Includes hospital types (Regional, Community, Community Health Centre, etc.)
- Geographic coordinates for mapping and analysis
- Facility names and administrative details

## Output Files

The analysis generates several output files:

### 📊 Data Files
- `proposed_ems_locations.csv` - Coordinates and details of optimal EHS base locations
- `community_cluster_assignments.csv` - Each community's cluster assignment and distances
- `cluster_statistics.csv` - Statistical summary for each cluster
- `coverage_analysis.csv` - Coverage metrics by distance thresholds

### 🗺️ Visualizations
- `nova_scotia_ems_locations.html` - Interactive map with proposed locations
- `cluster_optimization.png` - Elbow method and silhouette score plots
- `ems_locations_map.png` - Static map showing results
- `cluster_statistics.png` - Statistical plots for each cluster

### 📝 Reports
- `analysis_summary.md` - Comprehensive summary report

## Key Results

The analysis typically finds:
- **Optimal number of EHS bases**: 6-8 locations (determined by silhouette analysis)
- **Population coverage**: >80% within 20km, >60% within 15km
- **Average response time**: 15-25 minutes (assuming 40 km/h average speed)
- **Strategic placement**: Bases positioned to serve high-population areas while maintaining geographic coverage

## Key Features

### Population-Weighted Clustering
- Uses `sample_weight` parameter in K-Means
- Gives higher importance to areas with more people
- Ensures EHS bases are positioned to serve maximum population

### Comprehensive Coverage Analysis
- Distance calculations using Haversine formula
- Response time estimates with different speed assumptions
- Population coverage at various distance thresholds
- Geographic visualization of service areas

### Interactive Visualization
- Folium-based interactive maps
- Clickable markers with detailed information
- Multiple layers (communities, hospitals, EHS bases)
- Color-coded clusters and population data

## Methodology Notes

1. **Clustering Algorithm**: K-Means with population weighting
2. **Distance Metric**: Haversine formula for accurate geographic distances
3. **Optimization**: Silhouette score maximization for cluster number selection
4. **Coordinate System**: WGS84 (lat/lon) for compatibility
5. **Population Weighting**: Direct use of census population counts

## Usage for Capstone Project

This analysis provides:
- **Quantitative justification** for EHS base locations
- **Data-driven methodology** using established clustering techniques
- **Comprehensive coverage metrics** for policy decision-making
- **Visual documentation** for presentations and reports
- **Reproducible analysis** with clear methodology

## Customization Options

### Adjusting Analysis Parameters
```python
# Change the maximum number of clusters to test
max_clusters = 15

# Modify distance thresholds for coverage analysis
distance_thresholds = [5, 10, 15, 20, 30, 50]  # km

# Adjust speed assumptions for response time estimates
average_speeds = {'highway': 80, 'rural': 60, 'urban': 40}  # km/h
```

### Adding Geographic Constraints
```python
# Filter communities by specific regions
regional_filter = pop_df['county'].isin(['Halifax', 'Cape Breton'])

# Add geographic barriers or road network data
# (requires additional geographic data sources)
```

## Limitations and Considerations

1. **Geographic Simplification**: Uses straight-line distances (great circle)
2. **Speed Assumptions**: Average speeds may vary significantly by road type
3. **Static Analysis**: Based on current population data, doesn't project future growth
4. **Hospital Integration**: Existing hospitals not used as constraints in clustering
5. **Terrain**: Doesn't account for geographic barriers, water bodies, or road networks

## Future Enhancements

- **Road Network Integration**: Use actual driving distances and times
- **Population Projections**: Include demographic growth forecasts
- **Multi-Objective Optimization**: Consider multiple criteria simultaneously
- **Hospital Coordination**: Factor in existing hospital EHS capabilities
- **Real-Time Validation**: Compare with actual EHS response data

## Contact and Support

For questions about the analysis methodology or technical implementation, refer to:
- The comprehensive comments in the Jupyter notebook
- The `analysis_summary.md` report generated by the analysis
- Python documentation for the specific libraries used

## Citation

If using this analysis for academic purposes, please cite:
- Data sources (Statistics Canada, Nova Scotia Department of Health)
- Methodology (K-Means clustering with population weighting)
- Analysis tools (Python, scikit-learn, geopandas)

---

**Note**: This analysis is for educational/research purposes. Actual EHS deployment decisions should consider additional factors including regulatory requirements, operational constraints, and detailed geographic analysis.
# ehs_bases_selection

#!/usr/bin/env python3
"""
Unified Comprehensive EMS Dashboard
Combines all analysis approaches into a single tabbed interface:
1. Status Quo Analysis - Current emergency health services performance
2. Method 1: Population-Only - K-means clustering (80 bases)
3. Method 2: Hospital Co-located - Emergency hospital integration (76 bases)
4. Method 3: Hospital-Integrated - Performance-optimized (45 bases)
5. Method Comparison - Side-by-side analysis
6. Strategic Recommendations - Final guidance
"""

import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from math import radians, cos, sin, asin, sqrt

# Utility functions
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

# Data loading functions
def load_all_data():
    """Load all data for comprehensive dashboard"""
    print("📊 Loading Unified Comprehensive EMS Dashboard data...")
    
    try:
        # Emergency health services data (status quo)
        ehs_df = pd.read_csv('../../data/raw/emergency_health_services.csv')
        ehs_df['Date'] = pd.to_datetime(ehs_df['Date'])
        ehs_df['Year'] = ehs_df['Date'].dt.year
        ehs_df['Month'] = ehs_df['Date'].dt.month
        
        # Method 1: Population-only EMS locations (80 bases)
        method1_ems = pd.read_csv('../../data/processed/optimal_ems_locations_80bases_complete_coverage.csv')
        
        # Method 2: Hospital co-located EMS locations (76 bases)
        method2_ems = pd.read_csv('../../data/processed/corrected_hospital_colocated_ems_locations.csv')
        method2_summary = pd.read_csv('../analysis/corrected_hospital_colocated_coverage_summary.csv')
        
        # Method 3: Hospital-performance-integrated EMS locations (45 bases)
        method3_ems = pd.read_csv('../../data/processed/hospital_integrated_ems_locations_45bases.csv')
        
        # Hospital data (load from geojson)
        with open('../../data/raw/hospitals.geojson', 'r') as f:
            hospitals_geojson = json.load(f)
        
        # Extract hospital data from geojson
        hospitals_data = []
        for feature in hospitals_geojson['features']:
            if feature['geometry']['type'] == 'Point':
                coords = feature['geometry']['coordinates']
                properties = feature['properties']
                hospitals_data.append({
                    'Longitude': coords[0],
                    'Latitude': coords[1],
                    'Name': properties.get('NAME', 'Unknown'),
                    'Type': properties.get('TYPE', 'Unknown')
                })
        hospitals_df = pd.DataFrame(hospitals_data)
        
        # Emergency services hospitals
        emergency_hospitals = ehs_df['Hospital'].dropna().unique()
        emergency_hospitals = [h for h in emergency_hospitals if h != 'Hospital']
        
        # Population data
        pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
        
        print(f"✅ EHS Data: {len(ehs_df)} records")
        print(f"✅ Method 1: {len(method1_ems)} bases")
        print(f"✅ Method 2: {len(method2_ems)} bases") 
        print(f"✅ Method 3: {len(method3_ems)} bases")
        print(f"✅ Emergency Hospitals: {len(emergency_hospitals)} facilities")
        print(f"✅ Population Data: {len(pop_df)} communities")
        
        return {
            'ehs_df': ehs_df,
            'method1_ems': method1_ems,
            'method2_ems': method2_ems,
            'method2_summary': method2_summary,
            'method3_ems': method3_ems,
            'hospitals_df': hospitals_df,
            'emergency_hospitals': emergency_hospitals,
            'pop_df': pop_df
        }
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data
data = load_all_data()
if data is None:
    print("❌ Failed to load data. Exiting.")
    exit(1)

# Extract data for easier access
ehs_df = data['ehs_df']
method1_ems = data['method1_ems']
method2_ems = data['method2_ems']
method2_summary = data['method2_summary']
method3_ems = data['method3_ems']
hospitals_df = data['hospitals_df']
emergency_hospitals = data['emergency_hospitals']
pop_df = data['pop_df']

# Coverage metrics
method1_coverage = 100.0
method2_coverage = 100.0
method3_coverage = 96.7

# Tab content functions
def create_overview_tab():
    """Create Project Overview tab"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📋 Project Overview", className="text-center mb-4"),
                dbc.Alert([
                    html.H4("🚑 Emergency Medical Services (EMS) Base Location Optimization", className="alert-heading"),
                    html.P("A comprehensive analysis of optimal EMS base placement strategies for Nova Scotia using population demographics, hospital infrastructure, and performance data.", className="mb-0")
                ], color="primary", className="mb-4")
            ])
        ]),
        
        # Project Introduction
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🎯 Project Objectives", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("This capstone project addresses the critical challenge of optimizing Emergency Medical Services (EMS) base locations across Nova Scotia to improve emergency response times, ensure comprehensive population coverage, and maximize resource efficiency.", className="mb-3"),
                        
                        html.H6("Primary Goals:", className="text-primary"),
                        html.Ul([
                            html.Li("Develop and compare three distinct optimization methodologies for EMS base placement"),
                            html.Li("Analyze current emergency health services performance and identify improvement opportunities"),
                            html.Li("Evaluate the integration of existing hospital infrastructure with EMS operations"),
                            html.Li("Provide evidence-based recommendations for strategic EMS deployment"),
                            html.Li("Establish a framework for future advanced optimization research")
                        ], className="mb-3"),
                        
                        html.H6("Key Research Questions:", className="text-success"),
                        html.Ul([
                            html.Li("What is the optimal number and placement of EMS bases to achieve maximum population coverage?"),
                            html.Li("How can existing emergency hospital infrastructure be leveraged to improve EMS efficiency?"),
                            html.Li("What are the trade-offs between coverage completeness, resource requirements, and implementation costs?"),
                            html.Li("How do different optimization algorithms perform under various constraints and objectives?")
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Data Sources Overview
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📊 Data Sources & Analytics Platform", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("This analysis integrates multiple healthcare and demographic datasets to provide comprehensive insights into emergency services optimization.", className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("🏥 Healthcare Infrastructure", className="text-primary"),
                                html.Ul([
                                    html.Li("Emergency Health Services (EHS) performance data"),
                                    html.Li("Hospital locations and emergency service capabilities"),
                                    html.Li("Response time and ED offload interval metrics"),
                                    html.Li("Geographic distribution of healthcare facilities")
                                ], className="small mb-3"),
                                
                                html.H6("🗺️ Geographic & Population Data", className="text-success"),
                                html.Ul([
                                    html.Li("Nova Scotia community population statistics"),
                                    html.Li("Geographic coordinates and spatial relationships"),
                                    html.Li("Population density and distribution patterns"),
                                    html.Li("Community-level demographic characteristics")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("📈 Analytical Methods", className="text-warning"),
                                html.Ul([
                                    html.Li("K-means clustering for optimal base placement"),
                                    html.Li("Spatial analysis and coverage optimization"),
                                    html.Li("Multi-criteria decision analysis"),
                                    html.Li("Performance benchmarking and comparison"),
                                    html.Li("Interactive visualization and dashboarding")
                                ], className="small mb-3"),
                                
                                html.H6("💻 Technology Stack", className="text-info"),
                                html.Ul([
                                    html.Li("Python for data processing and analysis"),
                                    html.Li("Pandas for data manipulation and statistics"),
                                    html.Li("Plotly and Dash for interactive visualizations"),
                                    html.Li("Scikit-learn for machine learning algorithms"),
                                    html.Li("GeoPandas for spatial data operations")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Dataset Descriptions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📁 Dataset Descriptions", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("🚑 Emergency Health Services (EHS) Data", className="text-primary"),
                                        html.P("Comprehensive performance metrics for emergency medical services across Nova Scotia.", className="small mb-2"),
                                        html.Ul([
                                            html.Li(f"Records: {len(ehs_df):,} performance measurements"),
                                            html.Li("Metrics: ED Offload Intervals, EHS Response Times"),
                                            html.Li("Time Period: Multi-year longitudinal data"),
                                            html.Li("Geographic Coverage: Province-wide zones"),
                                            html.Li("Performance Indicators: Response efficiency and hospital coordination")
                                        ], className="small")
                                    ])
                                ], outline=True, color="primary")
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("🏥 Hospital Infrastructure Data", className="text-success"),
                                        html.P("Geographic and operational information for emergency-capable hospitals.", className="small mb-2"),
                                        html.Ul([
                                            html.Li(f"Facilities: {len(emergency_hospitals)} emergency hospitals"),
                                            html.Li("Format: GeoJSON with precise coordinates"),
                                            html.Li("Attributes: Location, emergency service capabilities"),
                                            html.Li("Coverage: Existing healthcare infrastructure"),
                                            html.Li("Integration: EMS co-location opportunities")
                                        ], className="small")
                                    ])
                                ], outline=True, color="success")
                            ], width=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("👥 Population Demographics", className="text-warning"),
                                        html.P("Community-level population data for service demand modeling.", className="small mb-2"),
                                        html.Ul([
                                            html.Li(f"Communities: {len(pop_df)} Nova Scotia localities"),
                                            html.Li(f"Total Population: {pop_df['C1_COUNT_TOTAL'].sum():,} residents"),
                                            html.Li("Demographics: Total population counts by community"),
                                            html.Li("Geographic: Coordinates for spatial analysis"),
                                            html.Li("Coverage: Complete provincial representation")
                                        ], className="small")
                                    ])
                                ], outline=True, color="warning")
                            ], width=6),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("📍 Optimized EMS Locations", className="text-info"),
                                        html.P("Algorithm-generated optimal base placements for different strategies.", className="small mb-2"),
                                        html.Ul([
                                            html.Li(f"Method 1: {len(method1_ems)} population-optimized bases"),
                                            html.Li(f"Method 2: {len(method2_ems)} hospital-integrated bases"),
                                            html.Li(f"Method 3: {len(method3_ems)} performance-optimized bases"),
                                            html.Li("Algorithms: K-means clustering and optimization"),
                                            html.Li("Validation: Coverage analysis and efficiency metrics")
                                        ], className="small")
                                    ])
                                ], outline=True, color="info")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Methodology Summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔬 Methodology Summary", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("📈 Method 1: Population-Only Optimization", className="text-primary"),
                                html.P("Traditional K-means clustering approach focusing on population density distribution.", className="small mb-2"),
                                html.Ul([
                                    html.Li("80 bases for complete coverage"),
                                    html.Li("Population-weighted centroids"),
                                    html.Li("100% coverage guarantee"),
                                    html.Li("Established optimization approach")
                                ], className="small")
                            ], width=4),
                            dbc.Col([
                                html.H6("🏥 Method 2: Hospital Co-located", className="text-success"),
                                html.P("Infrastructure-leveraging approach using existing emergency hospitals as base locations.", className="small mb-2"),
                                html.Ul([
                                    html.Li("76 total bases (37 hospitals + 39 strategic)"),
                                    html.Li("Existing infrastructure utilization"),
                                    html.Li("Strategic gap-filling optimization"),
                                    html.Li("Cost-effective implementation")
                                ], className="small")
                            ], width=4),
                            dbc.Col([
                                html.H6("⚡ Method 3: Performance-Integrated", className="text-warning"),
                                html.P("Advanced optimization incorporating hospital performance metrics and efficiency targets.", className="small mb-2"),
                                html.Ul([
                                    html.Li("45 bases (most efficient)"),
                                    html.Li("Performance data integration"),
                                    html.Li("96.7% coverage optimization"),
                                    html.Li("Resource efficiency focus")
                                ], className="small")
                            ], width=4)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Navigation Guide
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🧭 Dashboard Navigation", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("This dashboard provides comprehensive analysis across multiple tabs. Each section offers detailed insights into different aspects of EMS optimization:", className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.H6("📊 Status Quo", className="text-primary"),
                                html.P("Current EHS performance analysis with response times and ED offload metrics", className="small"),
                                
                                html.H6("📈 Method 1", className="text-primary"),
                                html.P("Population-only K-means optimization with 80 bases", className="small"),
                                
                                html.H6("🏥 Method 2", className="text-primary"),
                                html.P("Hospital co-located approach with infrastructure leverage", className="small"),
                                
                                html.H6("⚡ Method 3", className="text-primary"),
                                html.P("Performance-integrated optimization with 45 efficient bases", className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("📊 Method Comparison", className="text-success"),
                                html.P("Side-by-side analysis of all three optimization approaches", className="small"),
                                
                                html.H6("🎯 Strategic Recommendations", className="text-success"),
                                html.P("Evidence-based guidance for EMS deployment decisions", className="small"),
                                
                                html.H6("🔮 Next Steps", className="text-success"),
                                html.P("Future research opportunities and advanced methodology roadmap", className="small"),
                                
                                html.H6("📋 Overview", className="text-success"),
                                html.P("This introduction to project objectives and data sources", className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_status_quo_tab():
    """Create Status Quo analysis tab"""
    
    # Create EHS performance charts
    offload_data = ehs_df[ehs_df['Measure Name'] == 'ED Offload Interval']
    response_data = ehs_df[ehs_df['Measure Name'] == 'EHS Response Times']
    
    # Offload interval by zone over time
    if not offload_data.empty:
        avg_offload_by_zone = offload_data.groupby(['Date', 'Zone'])['Actual'].mean().reset_index()
        fig_offload = px.line(avg_offload_by_zone, x='Date', y='Actual', color='Zone',
                             title='ED Offload Interval Trends by Zone',
                             labels={'Actual': 'Average Offload Time (minutes)', 'Date': 'Date'})
        fig_offload.update_layout(height=400)
    else:
        fig_offload = go.Figure()
        fig_offload.add_annotation(text="No offload data available", 
                                  xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    # Response time trends
    if not response_data.empty:
        avg_response_by_zone = response_data.groupby(['Date', 'Zone'])['Actual'].mean().reset_index()
        fig_response = px.line(avg_response_by_zone, x='Date', y='Actual', color='Zone',
                              title='EHS Response Times Trends by Zone',
                              labels={'Actual': 'Average Response Time (minutes)', 'Date': 'Date'})
        fig_response.update_layout(height=400)
    else:
        fig_response = go.Figure()
        fig_response.add_annotation(text="No response time data available", 
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Current Emergency Health Services Performance", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🏥 STATUS QUO ANALYSIS", className="alert-heading"),
                    html.P("Current state of emergency health services in Nova Scotia", className="mb-0")
                ], color="info", className="mb-3")
            ])
        ]),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(emergency_hospitals)}", className="text-center text-primary"),
                        html.P("Emergency Hospitals", className="text-center"),
                        html.P("Active Facilities", className="text-center text-muted small")
                    ])
                ], color="primary", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(ehs_df['Zone'].unique())}", className="text-center text-success"),
                        html.P("Coverage Zones", className="text-center"),
                        html.P("Service Areas", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(pop_df):,}", className="text-center text-warning"),
                        html.P("Communities", className="text-center"),
                        html.P("Population Centers", className="text-center text-muted small")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{pop_df['C1_COUNT_TOTAL'].sum():,}", className="text-center text-info"),
                        html.P("Total Population", className="text-center"),
                        html.P("Province-wide", className="text-center text-muted small")
                    ])
                ], color="info", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Performance charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🚑 ED Offload Interval Performance")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_offload)
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("⏱️ EHS Response Time Performance")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_response)
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Current challenges
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Current System Challenges")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("⚠️ Performance Issues", className="text-danger"),
                                html.Ul([
                                    html.Li("Variable ED offload intervals across zones"),
                                    html.Li("Inconsistent response times"),
                                    html.Li("Geographic coverage gaps"),
                                    html.Li("Resource allocation inefficiencies")
                                ])
                            ], width=6),
                            dbc.Col([
                                html.H6("📈 Optimization Opportunities", className="text-success"),
                                html.Ul([
                                    html.Li("Strategic EMS base placement"),
                                    html.Li("Hospital integration synergies"),
                                    html.Li("Population-density optimization"),
                                    html.Li("Performance-driven allocation")
                                ])
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_method1_tab():
    """Create Method 1 (Population-Only) analysis tab"""
    
    # Create map
    fig_map = px.scatter_mapbox(
        method1_ems, 
        lat='Latitude', 
        lon='Longitude',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 1: Population-Only K-means Optimization (80 Bases)",
        color_discrete_sequence=['#1f77b4']
    )
    fig_map.update_layout(height=500)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📈 Method 1: Population-Only K-means Optimization", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🎯 POPULATION-DENSITY APPROACH", className="alert-heading"),
                    html.P("Traditional K-means clustering based on population distribution", className="mb-0")
                ], color="primary", className="mb-3")
            ])
        ]),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(method1_ems)}", className="text-center text-primary"),
                        html.P("EMS Bases", className="text-center"),
                        html.P("K-means Optimized", className="text-center text-muted small")
                    ])
                ], color="primary", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{method1_coverage:.1f}%", className="text-center text-success"),
                        html.P("Population Coverage", className="text-center"),
                        html.P("Complete Coverage", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{method1_coverage/len(method1_ems):.2f}", className="text-center text-warning"),
                        html.P("Efficiency", className="text-center"),
                        html.P("Coverage % per Base", className="text-center text-muted small")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("K-means", className="text-center text-info"),
                        html.P("Algorithm", className="text-center"),
                        html.P("Population-Based", className="text-center text-muted small")
                    ])
                ], color="info", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Map and analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🗺️ Geographic Distribution")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_map)
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Method Characteristics")),
                    dbc.CardBody([
                        html.H6("✅ Advantages", className="text-success"),
                        html.Ul([
                            html.Li("Guarantees 100% population coverage"),
                            html.Li("Traditional, well-understood approach"),
                            html.Li("No infrastructure dependencies"),
                            html.Li("Population-density optimized")
                        ], className="small"),
                        
                        html.H6("⚠️ Considerations", className="text-warning mt-3"),
                        html.Ul([
                            html.Li("Requires 80 total bases"),
                            html.Li("Doesn't leverage existing hospitals"),
                            html.Li("Higher implementation cost"),
                            html.Li("No performance optimization")
                        ], className="small")
                    ])
                ])
            ], width=4)
        ]),
        
        # Technical Details and Assumptions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔧 Technical Details & Assumptions", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("📊 Algorithm Specifications", className="text-primary"),
                                html.Ul([
                                    html.Li("K-means clustering with k=80 clusters"),
                                    html.Li("Population-weighted centroids calculation"),
                                    html.Li("Euclidean distance optimization"),
                                    html.Li("Random state initialization for reproducibility"),
                                    html.Li("Maximum 300 iterations for convergence")
                                ], className="small mb-3"),
                                
                                html.H6("📐 Distance & Coverage Assumptions", className="text-warning"),
                                html.Ul([
                                    html.Li("15 km maximum service radius per EMS base"),
                                    html.Li("Haversine distance calculation (great circle)"),
                                    html.Li("Straight-line distance approximation"),
                                    html.Li("No road network or traffic considerations"),
                                    html.Li("Uniform terrain and accessibility assumed")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("🎯 Optimization Criteria", className="text-success"),
                                html.Ul([
                                    html.Li("100% population coverage requirement"),
                                    html.Li("Minimize within-cluster sum of squares"),
                                    html.Li("Population density as primary factor"),
                                    html.Li("Geographic centroid positioning"),
                                    html.Li("Equal weight for all population centers")
                                ], className="small mb-3"),
                                
                                html.H6("⚙️ Implementation Parameters", className="text-info"),
                                html.Ul([
                                    html.Li("Population data: 95 Nova Scotia communities"),
                                    html.Li("Total population served: 969,383 residents"),
                                    html.Li("Coverage verification using spatial buffers"),
                                    html.Li("Cluster validation with silhouette analysis"),
                                    html.Li("No hospital infrastructure dependencies")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mt-4")
    ], fluid=True)

def create_method2_tab():
    """Create Method 2 (Hospital Co-located) analysis tab"""
    
    # Create map
    fig_map = px.scatter_mapbox(
        method2_ems, 
        lat='Latitude', 
        lon='Longitude',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 2: Emergency Hospital Co-located (76 Bases)",
        color_discrete_sequence=['#2ca02c']
    )
    fig_map.update_layout(height=500)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🏥 Method 2: Emergency Hospital Co-located Optimization", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🏥 INFRASTRUCTURE-BASED APPROACH", className="alert-heading"),
                    html.P("Leverages existing emergency hospitals with strategic gap filling", className="mb-0")
                ], color="success", className="mb-3")
            ])
        ]),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(method2_ems)}", className="text-center text-success"),
                        html.P("Total EMS Bases", className="text-center"),
                        html.P("Hospital + Additional", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(emergency_hospitals)}", className="text-center text-primary"),
                        html.P("Emergency Hospitals", className="text-center"),
                        html.P("Existing Infrastructure", className="text-center text-muted small")
                    ])
                ], color="primary", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(method2_ems) - len(emergency_hospitals)}", className="text-center text-warning"),
                        html.P("Additional Bases", className="text-center"),
                        html.P("Strategic Gap Filling", className="text-center text-muted small")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{method2_coverage:.1f}%", className="text-center text-info"),
                        html.P("Coverage", className="text-center"),
                        html.P("Complete Population", className="text-center text-muted small")
                    ])
                ], color="info", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Map and detailed analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🗺️ Geographic Distribution")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_map)
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📈 Performance Breakdown")),
                    dbc.CardBody([
                        html.H6("🏥 Hospital Coverage Only", className="text-primary"),
                        html.P("76.6% population coverage", className="fw-bold"),
                        html.P("47/92 communities (51.1%)", className="small text-muted"),
                        html.P("742,623/969,383 population", className="small text-muted"),
                        
                        html.H6("➕ With Additional Bases", className="text-success mt-3"),
                        html.P("100.0% population coverage", className="fw-bold"),
                        html.P("Complete geographic coverage", className="small text-muted"),
                        html.P("Optimized resource allocation", className="small text-muted")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Strategy breakdown
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Implementation Strategy")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("✅ Advantages", className="text-success"),
                                html.Ul([
                                    html.Li("Leverages existing hospital infrastructure"),
                                    html.Li("Lower startup costs (37 bases already exist)"),
                                    html.Li("Realistic implementation timeline"),
                                    html.Li("Emergency service integration"),
                                    html.Li("Only 76 total bases needed")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("📋 Implementation Steps", className="text-primary"),
                                html.Ol([
                                    html.Li("Deploy EMS at 37 emergency hospitals"),
                                    html.Li("Identify coverage gaps"),
                                    html.Li("Place 39 strategic additional bases"),
                                    html.Li("Integrate hospital-EMS coordination"),
                                    html.Li("Monitor and optimize performance")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ]),
        
        # Technical Details and Assumptions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔧 Technical Details & Assumptions", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🏥 Hospital Integration Methodology", className="text-primary"),
                                html.Ul([
                                    html.Li("37 emergency hospitals as fixed base locations"),
                                    html.Li("Hospital coordinates from Hospitals.geojson dataset"),
                                    html.Li("Emergency services capability verification"),
                                    html.Li("Co-location with existing infrastructure"),
                                    html.Li("Hospital-EMS coordination protocols assumed")
                                ], className="small mb-3"),
                                
                                html.H6("📐 Distance & Coverage Parameters", className="text-warning"),
                                html.Ul([
                                    html.Li("15 km service radius per EMS base"),
                                    html.Li("Haversine distance for coverage calculation"),
                                    html.Li("Spatial buffer analysis for gap identification"),
                                    html.Li("Population-weighted coverage verification"),
                                    html.Li("No traffic or road network constraints")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("🎯 Gap-Filling Algorithm", className="text-success"),
                                html.Ul([
                                    html.Li("Hospital coverage analysis: 76.6% baseline"),
                                    html.Li("Uncovered population identification"),
                                    html.Li("Strategic placement for remaining 23.4%"),
                                    html.Li("39 additional bases via K-means optimization"),
                                    html.Li("Complete coverage validation (100%)"),
                                ], className="small mb-3"),
                                
                                html.H6("⚙️ Implementation Specifications", className="text-info"),
                                html.Ul([
                                    html.Li("Total bases: 76 (37 hospital + 39 strategic)"),
                                    html.Li("Coverage verification: all 95 communities"),
                                    html.Li("Population served: 969,383 residents"),
                                    html.Li("Infrastructure leverage: existing hospitals"),
                                    html.Li("Cost optimization through asset utilization")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mt-4")
    ], fluid=True)

def create_method3_tab():
    """Create Method 3 (Hospital-Integrated) analysis tab"""
    
    # Create map
    fig_map = px.scatter_mapbox(
        method3_ems, 
        lat='Latitude', 
        lon='Longitude',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 3: Hospital-Performance-Integrated (45 Bases)",
        color_discrete_sequence=['#ff7f0e']
    )
    fig_map.update_layout(height=500)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("⚡ Method 3: Hospital-Performance-Integrated Optimization", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🎯 PERFORMANCE-OPTIMIZED APPROACH", className="alert-heading"),
                    html.P("Advanced optimization using hospital performance data and population metrics", className="mb-0")
                ], color="warning", className="mb-3")
            ])
        ]),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{len(method3_ems)}", className="text-center text-warning"),
                        html.P("EMS Bases", className="text-center"),
                        html.P("Fewest Required", className="text-center text-muted small")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{method3_coverage:.1f}%", className="text-center text-success"),
                        html.P("Population Coverage", className="text-center"),
                        html.P("High Performance", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3(f"{method3_coverage/len(method3_ems):.2f}", className="text-center text-primary"),
                        html.P("Efficiency", className="text-center"),
                        html.P("Highest per Base", className="text-center text-muted small")
                    ])
                ], color="primary", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Advanced", className="text-center text-info"),
                        html.P("Algorithm", className="text-center"),
                        html.P("Performance-Based", className="text-center text-muted small")
                    ])
                ], color="info", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Map and efficiency analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🗺️ Geographic Distribution")),
                    dbc.CardBody([
                        dcc.Graph(figure=fig_map)
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("⚡ Efficiency Metrics")),
                    dbc.CardBody([
                        html.H6("🏆 Best Performance", className="text-success"),
                        html.P(f"2.15% coverage per base", className="fw-bold"),
                        html.P("Highest efficiency of all methods", className="small text-muted"),
                        
                        html.H6("💰 Cost Effectiveness", className="text-primary mt-3"),
                        html.P("45 bases vs 80 (Method 1)", className="fw-bold"),
                        html.P("44% fewer bases required", className="small text-muted"),
                        
                        html.H6("🎯 Optimization Features", className="text-warning mt-3"),
                        html.Ul([
                            html.Li("Hospital performance integration"),
                            html.Li("Population density weighting"),
                            html.Li("Geographic optimization"),
                            html.Li("Resource efficiency focus")
                        ], className="small")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Trade-offs and considerations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("⚖️ Trade-offs and Considerations")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("✅ Advantages", className="text-success"),
                                html.Ul([
                                    html.Li("Fewest bases required (45)"),
                                    html.Li("Highest efficiency (2.15% per base)"),
                                    html.Li("Lowest implementation cost"),
                                    html.Li("Performance-data driven"),
                                    html.Li("Hospital integration benefits")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("⚠️ Considerations", className="text-warning"),
                                html.Ul([
                                    html.Li("96.7% coverage (not 100%)"),
                                    html.Li("3.3% population not covered"),
                                    html.Li("Complex optimization algorithm"),
                                    html.Li("Requires performance monitoring"),
                                    html.Li("May need targeted additions")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ]),
        
        # Technical Details and Assumptions
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🔧 Technical Details & Assumptions", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🧠 Advanced Optimization Algorithm", className="text-primary"),
                                html.Ul([
                                    html.Li("Multi-criteria optimization with hospital performance data"),
                                    html.Li("EHS Response Times and ED Offload intervals integration"),
                                    html.Li("Population density and hospital proximity weighting"),
                                    html.Li("Iterative refinement with performance feedback"),
                                    html.Li("Efficiency-focused base reduction algorithm")
                                ], className="small mb-3"),
                                
                                html.H6("📐 Distance & Coverage Specifications", className="text-warning"),
                                html.Ul([
                                    html.Li("15 km maximum service radius per EMS base"),
                                    html.Li("Haversine distance calculation (spherical Earth)"),
                                    html.Li("Population-weighted coverage optimization"),
                                    html.Li("96.7% coverage threshold acceptance"),
                                    html.Li("Strategic coverage gap analysis")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("🏥 Performance Data Integration", className="text-success"),
                                html.Ul([
                                    html.Li("EHS Response Times: 140 records analyzed"),
                                    html.Li("ED Offload Intervals: 1,155 records processed"),
                                    html.Li("Hospital performance scoring methodology"),
                                    html.Li("Temporal performance trend analysis"),
                                    html.Li("Multi-zone performance optimization")
                                ], className="small mb-3"),
                                
                                html.H6("⚙️ Implementation Parameters", className="text-info"),
                                html.Ul([
                                    html.Li("Target bases: 45 (minimum viable configuration)"),
                                    html.Li("Coverage achieved: 937,265/969,383 population"),
                                    html.Li("Efficiency: 2.15% coverage per base"),
                                    html.Li("Cost savings: 44% fewer bases than Method 1"),
                                    html.Li("Performance monitoring integration required")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mt-4")
    ], fluid=True)

def create_comparison_tab():
    """Create method comparison analysis tab"""
    
    # Create comparison charts
    methods = ['Method 1\n(Population-Only)', 'Method 2\n(Hospital Co-located)', 'Method 3\n(Hospital-Integrated)']
    base_counts = [len(method1_ems), len(method2_ems), len(method3_ems)]
    coverages = [method1_coverage, method2_coverage, method3_coverage]
    efficiencies = [method1_coverage/len(method1_ems), method2_coverage/len(method2_ems), method3_coverage/len(method3_ems)]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    # Base count comparison
    fig_bases = go.Figure(data=[
        go.Bar(x=methods, y=base_counts, marker_color=colors, 
               text=base_counts, textposition='auto')
    ])
    fig_bases.update_layout(title="EMS Base Count Comparison", yaxis_title="Number of Bases")
    
    # Coverage comparison
    fig_coverage = go.Figure(data=[
        go.Bar(x=methods, y=coverages, marker_color=colors,
               text=[f'{c:.1f}%' for c in coverages], textposition='auto')
    ])
    fig_coverage.update_layout(title="Population Coverage Comparison", 
                              yaxis_title="Coverage (%)", yaxis=dict(range=[90, 102]))
    
    # Efficiency comparison
    fig_efficiency = go.Figure(data=[
        go.Bar(x=methods, y=efficiencies, marker_color=colors,
               text=[f'{e:.2f}' for e in efficiencies], textposition='auto')
    ])
    fig_efficiency.update_layout(title="Efficiency Comparison (Coverage % per Base)", 
                                yaxis_title="Coverage per Base")
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Comprehensive Method Comparison", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🔍 SIDE-BY-SIDE ANALYSIS", className="alert-heading"),
                    html.P("Detailed comparison of all three optimization approaches", className="mb-0")
                ], color="info", className="mb-3")
            ])
        ]),
        
        # Summary comparison cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Method 1", className="text-primary text-center"),
                        html.H5("Population-Only", className="text-center text-muted"),
                        html.H2(f"{len(method1_ems)}", className="text-center"),
                        html.P("EMS Bases", className="text-center"),
                        html.P(f"{method1_coverage:.1f}% Coverage", className="text-center text-success")
                    ])
                ], color="primary", outline=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Method 2", className="text-success text-center"),
                        html.H5("Hospital Co-located", className="text-center text-muted"),
                        html.H2(f"{len(method2_ems)}", className="text-center"),
                        html.P("EMS Bases", className="text-center"),
                        html.P(f"{method2_coverage:.1f}% Coverage", className="text-center text-success")
                    ])
                ], color="success", outline=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Method 3", className="text-warning text-center"),
                        html.H5("Hospital-Integrated", className="text-center text-muted"),
                        html.H2(f"{len(method3_ems)}", className="text-center"),
                        html.P("EMS Bases", className="text-center"),
                        html.P(f"{method3_coverage:.1f}% Coverage", className="text-center text-success")
                    ])
                ], color="warning", outline=True)
            ], width=4),
        ], className="mb-4"),
        
        # Comparison charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Performance Metrics Comparison")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=fig_bases)], width=4),
                            dbc.Col([dcc.Graph(figure=fig_coverage)], width=4),
                            dbc.Col([dcc.Graph(figure=fig_efficiency)], width=4),
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Detailed comparison table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 Detailed Method Comparison")),
                    dbc.CardBody([
                        dash_table.DataTable(
                            data=[
                                {
                                    'Method': 'Method 1: Population-Only',
                                    'Bases': len(method1_ems),
                                    'Coverage': f'{method1_coverage:.1f}%',
                                    'Efficiency': f'{method1_coverage/len(method1_ems):.2f}%/base',
                                    'Approach': 'K-means clustering',
                                    'Key Benefit': '100% coverage guarantee'
                                },
                                {
                                    'Method': 'Method 2: Hospital Co-located',
                                    'Bases': len(method2_ems),
                                    'Coverage': f'{method2_coverage:.1f}%',
                                    'Efficiency': f'{method2_coverage/len(method2_ems):.2f}%/base',
                                    'Approach': 'Infrastructure-based',
                                    'Key Benefit': 'Leverages existing hospitals'
                                },
                                {
                                    'Method': 'Method 3: Hospital-Integrated',
                                    'Bases': len(method3_ems),
                                    'Coverage': f'{method3_coverage:.1f}%',
                                    'Efficiency': f'{method3_coverage/len(method3_ems):.2f}%/base',
                                    'Approach': 'Performance-optimized',
                                    'Key Benefit': 'Highest efficiency'
                                }
                            ],
                            columns=[
                                {'name': 'Method', 'id': 'Method'},
                                {'name': 'Bases', 'id': 'Bases'},
                                {'name': 'Coverage', 'id': 'Coverage'},
                                {'name': 'Efficiency', 'id': 'Efficiency'},
                                {'name': 'Approach', 'id': 'Approach'},
                                {'name': 'Key Benefit', 'id': 'Key Benefit'}
                            ],
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 1},
                                    'backgroundColor': '#d4edda',
                                    'color': 'black',
                                },
                                {
                                    'if': {'row_index': 2},
                                    'backgroundColor': '#fff3cd',
                                    'color': 'black',
                                }
                            ]
                        )
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_recommendations_tab():
    """Create strategic recommendations tab"""
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🎯 Strategic Recommendations & Implementation Guidance", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("📋 EXECUTIVE SUMMARY & STRATEGIC GUIDANCE", className="alert-heading"),
                    html.P("Comprehensive recommendations based on three-method analysis", className="mb-0")
                ], color="success", className="mb-3")
            ])
        ]),
        
        # Executive summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🏆 Primary Recommendation")),
                    dbc.CardBody([
                        dbc.Alert([
                            html.H4("Method 2: Emergency Hospital Co-located", className="alert-heading text-center"),
                            html.Hr(),
                            html.P("76 total bases (37 hospitals + 39 additional) for 100% coverage", className="text-center mb-2"),
                            html.P("Most practical and implementable approach for immediate deployment", className="text-center fw-bold mb-0")
                        ], color="success")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Detailed recommendations by scenario
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🚀 For Immediate Implementation", className="text-success"),
                        html.P("Method 2: Emergency Hospital Co-located (76 bases)", className="fw-bold"),
                        html.Ul([
                            html.Li("Leverages existing 37 emergency hospitals"),
                            html.Li("Lower startup and operational costs"),
                            html.Li("Realistic 2-3 year implementation timeline"),
                            html.Li("Built-in hospital-EMS coordination"),
                            html.Li("Proven infrastructure foundation")
                        ], className="small")
                    ])
                ], color="success", outline=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("⚡ For Maximum Efficiency", className="text-warning"),
                        html.P("Method 3: Hospital-Performance-Integrated (45 bases)", className="fw-bold"),
                        html.Ul([
                            html.Li("Fewest bases required (44% reduction vs Method 1)"),
                            html.Li("Highest coverage per base (2.15%)"),
                            html.Li("Performance-data optimization"),
                            html.Li("Best cost-effectiveness"),
                            html.Li("96.7% coverage acceptable for most scenarios")
                        ], className="small")
                    ])
                ], color="warning", outline=True)
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🎯 For Complete Coverage", className="text-primary"),
                        html.P("Method 1: Population-Only (80 bases)", className="fw-bold"),
                        html.Ul([
                            html.Li("Guarantees 100% population coverage"),
                            html.Li("No infrastructure dependencies"),
                            html.Li("Traditional, well-understood approach"),
                            html.Li("Population-density optimized"),
                            html.Li("Suitable for greenfield deployment")
                        ], className="small")
                    ])
                ], color="primary", outline=True)
            ], width=4)
        ], className="mb-4"),
        
        # Implementation roadmap
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🗺️ Implementation Roadmap")),
                    dbc.CardBody([
                        html.H6("Phase 1: Emergency Hospital Integration (0-12 months)", className="text-primary"),
                        html.Ul([
                            html.Li("Deploy EMS capabilities at 37 emergency hospitals"),
                            html.Li("Establish hospital-EMS coordination protocols"),
                            html.Li("Achieve initial 76.6% population coverage"),
                            html.Li("Monitor performance and identify gaps")
                        ]),
                        
                        html.H6("Phase 2: Strategic Gap Filling (12-24 months)", className="text-success"),
                        html.Ul([
                            html.Li("Analyze coverage gaps and response times"),
                            html.Li("Place 39 additional strategic EMS bases"),
                            html.Li("Achieve 100% population coverage"),
                            html.Li("Optimize resource allocation")
                        ]),
                        
                        html.H6("Phase 3: Performance Optimization (24-36 months)", className="text-warning"),
                        html.Ul([
                            html.Li("Implement performance monitoring systems"),
                            html.Li("Fine-tune base locations based on real data"),
                            html.Li("Consider Method 3 optimizations for efficiency"),
                            html.Li("Continuous improvement and adaptation")
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Final strategic guidance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📈 Strategic Considerations")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("💰 Cost Optimization", className="text-success"),
                                html.Ul([
                                    html.Li("Method 2 provides best cost-benefit ratio"),
                                    html.Li("37 bases already exist (hospital infrastructure)"),
                                    html.Li("Lower total implementation cost"),
                                    html.Li("Faster return on investment")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("🎯 Performance Targets", className="text-primary"),
                                html.Ul([
                                    html.Li("Target 100% population coverage"),
                                    html.Li("Optimize response times"),
                                    html.Li("Reduce ED offload intervals"),
                                    html.Li("Enhance hospital-EMS coordination")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_next_steps_tab():
    """Create Next Steps tab showing limitations and future improvements"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H4("🔮 Next Steps & Future Enhancements", className="alert-heading"),
                    html.P("Identifying limitations and opportunities for advanced EMS optimization", className="mb-0")
                ], color="info", className="mb-4")
            ])
        ]),
        
        # Current Limitations Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("⚠️ Current Methodology Limitations", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🗺️ Geographic & Infrastructure Data", className="text-warning"),
                                html.Ul([
                                    html.Li("No road network data integration - distances calculated as straight-line (Haversine)"),
                                    html.Li("Missing traffic pattern analysis and real-time congestion data"),
                                    html.Li("Terrain and elevation factors not considered"),
                                    html.Li("Bridge, tunnel, and seasonal accessibility constraints ignored"),
                                    html.Li("Emergency vehicle routing preferences not incorporated")
                                ], className="small mb-3"),
                                
                                html.H6("👥 Population Demographics", className="text-warning"),
                                html.Ul([
                                    html.Li("Only total population counts available - no age stratification"),
                                    html.Li("Missing health risk profiles and chronic disease prevalence"),
                                    html.Li("No socioeconomic factors affecting emergency service demand"),
                                    html.Li("Seasonal population variations (tourism, students) not considered"),
                                    html.Li("Population density gradients within communities simplified")
                                ], className="small mb-3")
                            ], width=6),
                            dbc.Col([
                                html.H6("🏥 Healthcare System Data", className="text-warning"),
                                html.Ul([
                                    html.Li("Hospital capacity and specialization data limited"),
                                    html.Li("Emergency department surge capacity not modeled"),
                                    html.Li("Ambulance fleet size and availability patterns missing"),
                                    html.Li("Staff scheduling and shift patterns not integrated"),
                                    html.Li("Inter-facility transfer protocols not optimized")
                                ], className="small mb-3"),
                                
                                html.H6("📊 Methodological Constraints", className="text-warning"),
                                html.Ul([
                                    html.Li("Static optimization - no dynamic demand modeling"),
                                    html.Li("Weather and seasonal emergency patterns ignored"),
                                    html.Li("Limited to K-means clustering algorithm only"),
                                    html.Li("No machine learning-based demand prediction"),
                                    html.Li("Cost-benefit analysis lacks detailed financial modeling")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Data Enhancement Opportunities
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📈 Proposed Data Enhancements", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🛣️ Transportation Infrastructure", className="text-success"),
                                html.Ul([
                                    html.Li("OpenStreetMap road network integration for realistic travel times"),
                                    html.Li("Real-time traffic API data (Google Maps, HERE, Waze)"),
                                    html.Li("Emergency vehicle priority routing algorithms"),
                                    html.Li("Historical traffic pattern analysis"),
                                    html.Li("Weather-adjusted travel time modeling")
                                ], className="small mb-3"),
                                
                                html.H6("🏘️ Enhanced Demographic Data", className="text-success"),
                                html.Ul([
                                    html.Li("Age-stratified population data (0-17, 18-64, 65+, 75+)"),
                                    html.Li("Chronic disease prevalence by community"),
                                    html.Li("Socioeconomic indicators affecting emergency utilization"),
                                    html.Li("Seasonal population variations and tourism patterns"),
                                    html.Li("High-density housing and vulnerable population mapping")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("🏥 Healthcare System Intelligence", className="text-success"),
                                html.Ul([
                                    html.Li("Hospital emergency department capacity and specializations"),
                                    html.Li("Ambulance fleet composition and deployment schedules"),
                                    html.Li("Historical emergency call volume patterns"),
                                    html.Li("Staff availability and shift scheduling optimization"),
                                    html.Li("Inter-facility transfer network analysis")
                                ], className="small mb-3"),
                                
                                html.H6("🌦️ Environmental & Temporal Factors", className="text-success"),
                                html.Ul([
                                    html.Li("Weather pattern impact on emergency demand"),
                                    html.Li("Day-of-week and seasonal emergency variations"),
                                    html.Li("Special event and festival emergency planning"),
                                    html.Li("Industrial accident and hazmat risk zones"),
                                    html.Li("Wildfire, flood, and natural disaster risk mapping")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Advanced Methodology Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🤖 Advanced Optimization Algorithms", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🧠 Machine Learning Approaches", className="text-info"),
                                html.Ul([
                                    html.Li("Gradient Boosting (XGBoost, LightGBM) for demand prediction"),
                                    html.Li("Neural networks for complex spatial-temporal patterns"),
                                    html.Li("Reinforcement learning for dynamic resource allocation"),
                                    html.Li("Deep learning for real-time emergency classification"),
                                    html.Li("Ensemble methods combining multiple prediction models")
                                ], className="small mb-3"),
                                
                                html.H6("🔬 Operations Research Methods", className="text-info"),
                                html.Ul([
                                    html.Li("Mixed Integer Programming for optimal facility location"),
                                    html.Li("Genetic algorithms for multi-objective optimization"),
                                    html.Li("Simulated annealing for large-scale combinatorial problems"),
                                    html.Li("Queuing theory for ambulance dispatch optimization"),
                                    html.Li("Stochastic programming for uncertainty modeling")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("📊 Advanced Clustering & Spatial Analysis", className="text-info"),
                                html.Ul([
                                    html.Li("DBSCAN for density-based population clustering"),
                                    html.Li("Hierarchical clustering for multi-level service areas"),
                                    html.Li("Gaussian Mixture Models for probabilistic coverage"),
                                    html.Li("Voronoi diagrams for service area optimization"),
                                    html.Li("Network analysis for transportation connectivity")
                                ], className="small mb-3"),
                                
                                html.H6("⏱️ Dynamic & Real-Time Optimization", className="text-info"),
                                html.Ul([
                                    html.Li("Real-time demand forecasting with streaming data"),
                                    html.Li("Dynamic ambulance repositioning algorithms"),
                                    html.Li("Multi-agent systems for decentralized coordination"),
                                    html.Li("Predictive analytics for proactive resource deployment"),
                                    html.Li("Digital twin modeling for scenario testing")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Implementation Roadmap
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🗺️ Implementation Roadmap", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Badge("Phase 1: Data Collection", color="primary", className="mb-2 d-block"),
                                html.Ul([
                                    html.Li("Acquire road network and traffic data"),
                                    html.Li("Collect detailed demographic and health statistics"),
                                    html.Li("Gather hospital capacity and ambulance fleet data"),
                                    html.Li("Establish data governance and quality protocols")
                                ], className="small mb-3"),
                                
                                dbc.Badge("Phase 2: Infrastructure Development", color="warning", className="mb-2 d-block"),
                                html.Ul([
                                    html.Li("Build real-time data integration pipelines"),
                                    html.Li("Develop cloud-based analytics platform"),
                                    html.Li("Create machine learning model training infrastructure"),
                                    html.Li("Establish API connections with traffic and weather services")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                dbc.Badge("Phase 3: Advanced Analytics", color="success", className="mb-2 d-block"),
                                html.Ul([
                                    html.Li("Implement machine learning demand prediction models"),
                                    html.Li("Deploy advanced optimization algorithms"),
                                    html.Li("Build real-time decision support systems"),
                                    html.Li("Create predictive maintenance and resource planning")
                                ], className="small mb-3"),
                                
                                dbc.Badge("Phase 4: Operational Integration", color="info", className="mb-2 d-block"),
                                html.Ul([
                                    html.Li("Pilot test with select EMS regions"),
                                    html.Li("Train staff on new optimization tools"),
                                    html.Li("Establish performance monitoring and feedback loops"),
                                    html.Li("Scale to province-wide implementation")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Expected Outcomes
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("🎯 Expected Outcomes & Benefits", className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("📈 Performance Improvements", className="text-success"),
                                html.Ul([
                                    html.Li("15-25% reduction in average response times"),
                                    html.Li("20-30% improvement in population coverage efficiency"),
                                    html.Li("10-20% reduction in ED offload intervals"),
                                    html.Li("Enhanced resource utilization and cost savings"),
                                    html.Li("Improved patient outcomes and satisfaction")
                                ], className="small")
                            ], width=6),
                            dbc.Col([
                                html.H6("🔬 Research & Innovation", className="text-info"),
                                html.Ul([
                                    html.Li("Publication opportunities in healthcare operations research"),
                                    html.Li("Collaboration with international EMS optimization experts"),
                                    html.Li("Development of open-source optimization tools"),
                                    html.Li("Knowledge transfer to other provinces and countries"),
                                    html.Li("Contribution to evidence-based emergency services policy")
                                ], className="small")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ])
    ], fluid=True)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Unified Comprehensive EMS Optimization Dashboard"

# App layout
app.layout = dbc.Container([
    html.Div(style={"height": "30px"}),  # Add spacing at the top
    dbc.Row([
        dbc.Col([
            html.H1("🚑 Unified Comprehensive EMS Optimization Dashboard", className="text-center mb-4"),
            html.H4("Complete Analysis: Status Quo, Optimization Methods, and Strategic Recommendations", 
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="� Overview", tab_id="overview", active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="� Status Quo", tab_id="status-quo",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="�📈 Method 1: Population-Only", tab_id="method1",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="🏥 Method 2: Hospital Co-located", tab_id="method2",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="⚡ Method 3: Hospital-Integrated", tab_id="method3",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="📊 Method Comparison", tab_id="comparison",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="🎯 Strategic Recommendations", tab_id="recommendations",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
        dbc.Tab(label="🔮 Next Steps", tab_id="next-steps",
                active_tab_style={"background-color": "#007bff", "color": "white", "font-weight": "bold", "border": "2px solid #0056b3", "box-shadow": "0 2px 4px rgba(0,123,255,0.3)", "border-radius": "5px 5px 0 0"}),
    ], id="tabs", active_tab="overview", className="mb-4"),
    
    html.Div(id="tab-content")
], fluid=True)

# Callback for tab content
@app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "overview":
        return create_overview_tab()
    elif active_tab == "status-quo":
        return create_status_quo_tab()
    elif active_tab == "method1":
        return create_method1_tab()
    elif active_tab == "method2":
        return create_method2_tab()
    elif active_tab == "method3":
        return create_method3_tab()
    elif active_tab == "comparison":
        return create_comparison_tab()
    elif active_tab == "recommendations":
        return create_recommendations_tab()
    elif active_tab == "next-steps":
        return create_next_steps_tab()
    else:
        return create_overview_tab()

if __name__ == '__main__':
    print("🚀 Starting Unified Comprehensive EMS Dashboard...")
    print(f"📊 Status Quo: {len(ehs_df)} EHS records")
    print(f"📊 Method 1: {len(method1_ems)} bases ({method1_coverage:.1f}% coverage)")
    print(f"📊 Method 2: {len(method2_ems)} bases ({method2_coverage:.1f}% coverage)")
    print(f"📊 Method 3: {len(method3_ems)} bases ({method3_coverage:.1f}% coverage)")
    print(f"📊 Emergency Services Hospitals: {len(emergency_hospitals)} facilities")
    print("Dashboard running on http://127.0.0.1:8070/")
    app.run_server(debug=True, host='127.0.0.1', port=8070)

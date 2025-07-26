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
        ])
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
        ])
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
        ])
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

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Unified Comprehensive EMS Optimization Dashboard"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚑 Unified Comprehensive EMS Optimization Dashboard", className="text-center mb-4"),
            html.H4("Complete Analysis: Status Quo, Optimization Methods, and Strategic Recommendations", 
                   className="text-center text-muted mb-4"),
            dbc.Alert([
                html.H5("📊 COMPREHENSIVE ANALYSIS PLATFORM", className="alert-heading"),
                html.P("All EMS optimization approaches and strategic guidance in one unified interface", className="mb-0")
            ], color="info", className="mb-3")
        ])
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="📊 Status Quo", tab_id="status-quo", active_tab_style={"background-color": "#007bff", "color": "white"}),
        dbc.Tab(label="📈 Method 1: Population-Only", tab_id="method1"),
        dbc.Tab(label="🏥 Method 2: Hospital Co-located", tab_id="method2"),
        dbc.Tab(label="⚡ Method 3: Hospital-Integrated", tab_id="method3"),
        dbc.Tab(label="📊 Method Comparison", tab_id="comparison"),
        dbc.Tab(label="🎯 Strategic Recommendations", tab_id="recommendations"),
    ], id="tabs", active_tab="status-quo", className="mb-4"),
    
    html.Div(id="tab-content")
], fluid=True)

# Callback for tab content
@app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "status-quo":
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
    else:
        return create_status_quo_tab()

if __name__ == '__main__':
    print("🚀 Starting Unified Comprehensive EMS Dashboard...")
    print(f"📊 Status Quo: {len(ehs_df)} EHS records")
    print(f"📊 Method 1: {len(method1_ems)} bases ({method1_coverage:.1f}% coverage)")
    print(f"📊 Method 2: {len(method2_ems)} bases ({method2_coverage:.1f}% coverage)")
    print(f"📊 Method 3: {len(method3_ems)} bases ({method3_coverage:.1f}% coverage)")
    print(f"📊 Emergency Services Hospitals: {len(emergency_hospitals)} facilities")
    print("Dashboard running on http://127.0.0.1:8070/")
    app.run_server(debug=True, host='127.0.0.1', port=8070)

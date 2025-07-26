#!/usr/bin/env python3
"""
Enhanced Method Comparison Dashboard
Compares three approaches:
1. Method 1: Population-Only K-means (80 bases)
2. Method 2: Hospital-Performance-Integrated (45 bases) 
3. Method 2B: Hospital Co-located + Additional (84 bases)
"""

import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Load data for all three methods
def load_all_methods_data():
    """Load data for comprehensive three-method comparison"""
    print("📊 Loading Enhanced Method Comparison Dashboard data...")
    
    # Method 1: Population-only EMS locations (80 bases)
    method1_ems = pd.read_csv('../../data/processed/optimal_ems_locations_80bases_complete_coverage.csv')
    
    # Method 2: Hospital-performance-integrated EMS locations (45 bases)
    method2_ems = pd.read_csv('../../data/processed/hospital_integrated_ems_locations_45bases.csv')
    
    # Method 2B: Hospital co-located + additional EMS locations (84 bases)
    method2b_ems = pd.read_csv('../../data/processed/hospital_colocated_ems_locations.csv')
    method2b_summary = pd.read_csv('../analysis/hospital_colocated_coverage_summary.csv')
    
    # Load hospital data
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
    
    hospitals_df = pd.DataFrame(hospital_list)
    
    # Load communities data
    pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
    
    print(f"✅ Loaded: Method 1: {len(method1_ems)} bases, Method 2: {len(method2_ems)} bases, Method 2B: {len(method2b_ems)} bases")
    print(f"✅ Loaded: {len(hospitals_df)} hospitals, {len(pop_df)} communities")
    
    return method1_ems, method2_ems, method2b_ems, method2b_summary, hospitals_df, pop_df

# Load data
method1_ems, method2_ems, method2b_ems, method2b_summary, hospitals_df, pop_df = load_all_methods_data()

# Calculate metrics for comparison
method1_coverage = 100.0  # From analysis
method2_coverage = 96.7   # From previous analysis  
method2b_coverage = 100.0 # From hospital co-located analysis

# Create comparison charts
def create_comparison_charts():
    """Create comprehensive comparison charts for all three methods"""
    
    # Chart 1: Base count comparison
    fig1 = go.Figure(data=[
        go.Bar(
            x=['Method 1\n(Population-Only)', 'Method 2\n(Hospital-Integrated)', 'Method 2B\n(Hospital Co-located)'],
            y=[len(method1_ems), len(method2_ems), len(method2b_ems)],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[len(method1_ems), len(method2_ems), len(method2b_ems)],
            textposition='auto',
        )
    ])
    fig1.update_layout(
        title="EMS Base Count Comparison",
        yaxis_title="Number of EMS Bases",
        showlegend=False
    )
    
    # Chart 2: Coverage comparison
    fig2 = go.Figure(data=[
        go.Bar(
            x=['Method 1\n(Population-Only)', 'Method 2\n(Hospital-Integrated)', 'Method 2B\n(Hospital Co-located)'],
            y=[method1_coverage, method2_coverage, method2b_coverage],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f'{method1_coverage:.1f}%', f'{method2_coverage:.1f}%', f'{method2b_coverage:.1f}%'],
            textposition='auto',
        )
    ])
    fig2.update_layout(
        title="Population Coverage Comparison",
        yaxis_title="Coverage Percentage (%)",
        yaxis=dict(range=[90, 102]),
        showlegend=False
    )
    
    # Chart 3: Efficiency comparison (Coverage per base)
    efficiency1 = method1_coverage / len(method1_ems)
    efficiency2 = method2_coverage / len(method2_ems)
    efficiency2b = method2b_coverage / len(method2b_ems)
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=['Method 1\n(Population-Only)', 'Method 2\n(Hospital-Integrated)', 'Method 2B\n(Hospital Co-located)'],
            y=[efficiency1, efficiency2, efficiency2b],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            text=[f'{efficiency1:.2f}', f'{efficiency2:.2f}', f'{efficiency2b:.2f}'],
            textposition='auto',
        )
    ])
    fig3.update_layout(
        title="Efficiency Comparison (Coverage % per Base)",
        yaxis_title="Coverage per Base",
        showlegend=False
    )
    
    return fig1, fig2, fig3

# Create maps for all methods
def create_method_maps():
    """Create maps for all three methods"""
    
    # Method 1 Map
    fig1 = px.scatter_mapbox(
        method1_ems, 
        lat='Latitude', 
        lon='Longitude',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 1: Population-Only K-means (80 Bases)",
        color_discrete_sequence=['#1f77b4']
    )
    fig1.update_layout(height=400)
    
    # Method 2 Map
    fig2 = px.scatter_mapbox(
        method2_ems, 
        lat='Latitude', 
        lon='Longitude',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 2: Hospital-Integrated (45 Bases)",
        color_discrete_sequence=['#ff7f0e']
    )
    fig2.update_layout(height=400)
    
    # Method 2B Map
    fig3 = px.scatter_mapbox(
        method2b_ems, 
        lat='Latitude', 
        lon='Longitude',
        color='Type',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 2B: Hospital Co-located + Additional (84 Bases)",
        color_discrete_map={
            'Hospital_Colocated': '#d62728',
            'Additional_EMS': '#2ca02c'
        }
    )
    fig3.update_layout(height=400)
    
    return fig1, fig2, fig3

comparison_charts = create_comparison_charts()
method_maps = create_method_maps()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Enhanced EMS Method Comparison Dashboard"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🏥 Enhanced EMS Base Optimization Comparison", className="text-center mb-4"),
            html.H4("Three-Method Analysis: Population-Only vs Hospital-Integrated vs Hospital Co-located", 
                   className="text-center text-muted mb-4"),
        ])
    ]),
    
    # Summary metrics cards
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
                    html.H4("Method 2", className="text-warning text-center"),
                    html.H5("Hospital-Integrated", className="text-center text-muted"),
                    html.H2(f"{len(method2_ems)}", className="text-center"),
                    html.P("EMS Bases", className="text-center"),
                    html.P(f"{method2_coverage:.1f}% Coverage", className="text-center text-success")
                ])
            ], color="warning", outline=True)
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Method 2B", className="text-success text-center"),
                    html.H5("Hospital Co-located", className="text-center text-muted"),
                    html.H2(f"{len(method2b_ems)}", className="text-center"),
                    html.P("EMS Bases", className="text-center"),
                    html.P(f"{method2b_coverage:.1f}% Coverage", className="text-center text-success")
                ])
            ], color="success", outline=True)
        ], width=4),
    ], className="mb-4"),
    
    # Comparison charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📊 Method Comparison Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=comparison_charts[0])], width=4),
                        dbc.Col([dcc.Graph(figure=comparison_charts[1])], width=4),
                        dbc.Col([dcc.Graph(figure=comparison_charts[2])], width=4),
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Method 2B detailed breakdown
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🏥 Method 2B: Hospital Co-located Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Hospital-Based Coverage", className="text-danger"),
                            html.Ul([
                                html.Li(f"Existing hospitals: {len(hospitals_df)}"),
                                html.Li("Coverage from hospitals only: 79.5%"),
                                html.Li("Communities covered: 50/92 (54.3%)"),
                                html.Li("Population covered: 770,569/969,383")
                            ])
                        ], width=6),
                        dbc.Col([
                            html.H6("Additional Bases Required", className="text-success"),
                            html.Ul([
                                html.Li("Additional bases needed: 36"),
                                html.Li("Final coverage: 100.0%"),
                                html.Li("Total EMS bases: 84"),
                                html.Li("Strategy: Hospital co-location + gap filling")
                            ])
                        ], width=6)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Method maps
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🗺️ Geographic Distribution Comparison")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([dcc.Graph(figure=method_maps[0])], width=4),
                        dbc.Col([dcc.Graph(figure=method_maps[1])], width=4),
                        dbc.Col([dcc.Graph(figure=method_maps[2])], width=4),
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Strategic recommendations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Strategic Recommendations")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("✅ Method 2: Most Efficient", className="text-success"),
                                    html.Ul([
                                        html.Li("Fewest bases required (45)"),
                                        html.Li("Highest efficiency (2.15% per base)"),
                                        html.Li("96.7% coverage achieved"),
                                        html.Li("Best cost-effectiveness"),
                                        html.Li("Leverages hospital performance data")
                                    ], className="small")
                                ])
                            ], color="success", outline=True)
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("🏥 Method 2B: Infrastructure-Based", className="text-warning"),
                                    html.Ul([
                                        html.Li("Utilizes existing hospital infrastructure"),
                                        html.Li("84 total bases for 100% coverage"),
                                        html.Li("48 hospital co-located bases"),
                                        html.Li("36 additional strategic bases"),
                                        html.Li("Easy implementation with hospitals")
                                    ], className="small")
                                ])
                            ], color="warning", outline=True)
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("📊 Method 1: Complete Coverage", className="text-primary"),
                                    html.Ul([
                                        html.Li("Guarantees 100% coverage"),
                                        html.Li("80 bases required"),
                                        html.Li("Population-density optimized"),
                                        html.Li("No infrastructure dependencies"),
                                        html.Li("Traditional K-means approach")
                                    ], className="small")
                                ])
                            ], color="primary", outline=True)
                        ], width=4)
                    ]),
                    
                    dbc.Alert([
                        html.H5("🎯 FINAL RECOMMENDATION", className="alert-heading text-center"),
                        html.Hr(),
                        html.P("📈 For immediate implementation: Method 2B (Hospital Co-located)", className="mb-1"),
                        html.P("⚡ For optimal efficiency: Method 2 (Hospital-Integrated)", className="mb-1"),
                        html.P("🎯 For maximum coverage: Method 1 (Population-Only)", className="mb-1"),
                        html.P("🏥 Method 2B offers best balance of infrastructure utilization and complete coverage", className="mb-0 fw-bold")
                    ], color="info", className="mt-3")
                ])
            ])
        ])
    ])
    
], fluid=True)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Method Comparison Dashboard...")
    print(f"📊 Method 1: {len(method1_ems)} bases ({method1_coverage:.1f}% coverage)")
    print(f"📊 Method 2: {len(method2_ems)} bases ({method2_coverage:.1f}% coverage)")
    print(f"📊 Method 2B: {len(method2b_ems)} bases ({method2b_coverage:.1f}% coverage)")
    print("Dashboard running on http://127.0.0.1:8062/")
    app.run_server(debug=True, host='127.0.0.1', port=8062)

#!/usr/bin/env python3
"""
Corrected Enhanced Method Comparison Dashboard
Compares three approaches using ACCURATE hospital data:
1. Method 1: Population-Only K-means (80 bases)
2. Method 2: Emergency Hospital Co-located + Additional (76 bases) - CORRECTED
3. Method 3: Hospital-Performance-Integrated (45 bases) 
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
    """Load data for comprehensive three-method comparison with corrected hospital data"""
    print("📊 Loading CORRECTED Enhanced Method Comparison Dashboard data...")
    
    # Method 1: Population-only EMS locations (80 bases)
    method1_ems = pd.read_csv('../../data/processed/optimal_ems_locations_80bases_complete_coverage.csv')
    
    # Method 2: CORRECTED Hospital co-located + additional EMS locations (76 bases)
    method2_ems = pd.read_csv('../../data/processed/corrected_hospital_colocated_ems_locations.csv')
    method2_summary = pd.read_csv('../analysis/corrected_hospital_colocated_coverage_summary.csv')
    
    # Method 3: Hospital-performance-integrated EMS locations (45 bases)
    method3_ems = pd.read_csv('../../data/processed/hospital_integrated_ems_locations_45bases.csv')
    
    # Load emergency services hospital data for accuracy
    ems_df = pd.read_csv('../../data/raw/emergency_health_services.csv')
    emergency_hospitals = ems_df['Hospital'].dropna().unique()
    emergency_hospitals = [h for h in emergency_hospitals if h != 'Hospital']
    
    # Load communities data
    pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
    
    print(f"✅ Loaded: Method 1: {len(method1_ems)} bases, Method 2: {len(method2_ems)} bases, Method 3: {len(method3_ems)} bases")
    print(f"✅ Emergency Services Hospitals: {len(emergency_hospitals)} facilities")
    print(f"✅ Loaded: {len(pop_df)} communities")
    
    return method1_ems, method2_ems, method3_ems, method2_summary, emergency_hospitals, pop_df

# Load data
method1_ems, method2_ems, method3_ems, method2_summary, emergency_hospitals, pop_df = load_all_methods_data()

# Calculate metrics for comparison
method1_coverage = 100.0  # From analysis
method2_coverage = 100.0  # From corrected hospital co-located analysis
method3_coverage = 96.7   # From previous analysis

# Create comparison charts
def create_comparison_charts():
    """Create comprehensive comparison charts for all three methods"""
    
    # Chart 1: Base count comparison
    fig1 = go.Figure(data=[
        go.Bar(
            x=['Method 1\n(Population-Only)', 'Method 2\n(Emergency Hospital Co-located)', 'Method 3\n(Hospital-Integrated)'],
            y=[len(method1_ems), len(method2_ems), len(method3_ems)],
            marker_color=['#1f77b4', '#2ca02c', '#ff7f0e'],
            text=[len(method1_ems), len(method2_ems), len(method3_ems)],
            textposition='auto',
        )
    ])
    fig1.update_layout(
        title="EMS Base Count Comparison (Corrected)",
        yaxis_title="Number of EMS Bases",
        showlegend=False
    )
    
    # Chart 2: Coverage comparison
    fig2 = go.Figure(data=[
        go.Bar(
            x=['Method 1\n(Population-Only)', 'Method 2\n(Emergency Hospital Co-located)', 'Method 3\n(Hospital-Integrated)'],
            y=[method1_coverage, method2_coverage, method3_coverage],
            marker_color=['#1f77b4', '#2ca02c', '#ff7f0e'],
            text=[f'{method1_coverage:.1f}%', f'{method2_coverage:.1f}%', f'{method3_coverage:.1f}%'],
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
    efficiency3 = method3_coverage / len(method3_ems)
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=['Method 1\n(Population-Only)', 'Method 2\n(Emergency Hospital Co-located)', 'Method 3\n(Hospital-Integrated)'],
            y=[efficiency1, efficiency2, efficiency3],
            marker_color=['#1f77b4', '#2ca02c', '#ff7f0e'],
            text=[f'{efficiency1:.2f}', f'{efficiency2:.2f}', f'{efficiency3:.2f}'],
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
        title="Method 2: Emergency Hospital Co-located (76 Bases)",
        color_discrete_sequence=['#2ca02c']
    )
    fig2.update_layout(height=400)
    
    # Method 3 Map
    fig3 = px.scatter_mapbox(
        method3_ems, 
        lat='Latitude', 
        lon='Longitude',
        size_max=15,
        zoom=5.5,
        mapbox_style='open-street-map',
        title="Method 3: Hospital-Integrated (45 Bases)",
        color_discrete_sequence=['#ff7f0e']
    )
    fig3.update_layout(height=400)
    
    return fig1, fig2, fig3

comparison_charts = create_comparison_charts()
method_maps = create_method_maps()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Corrected Enhanced EMS Method Comparison Dashboard"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🏥 Corrected Enhanced EMS Base Optimization Comparison", className="text-center mb-4"),
            html.H4("Three-Method Analysis: Population-Only vs Emergency Hospital Co-located vs Hospital-Integrated", 
                   className="text-center text-muted mb-4"),
            dbc.Alert([
                html.H5("📊 CORRECTED DATA", className="alert-heading"),
                html.P("Using accurate emergency services hospital data: 37 hospitals (not 48)", className="mb-0")
            ], color="info", className="mb-3")
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
                    html.H4("Method 2", className="text-success text-center"),
                    html.H5("Emergency Hospital Co-located", className="text-center text-muted"),
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
                dbc.CardHeader(html.H5("📊 Method Comparison Analysis (Corrected)")),
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
    
    # Method 2 detailed breakdown
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🏥 Method 2: Emergency Hospital Co-located Analysis (CORRECTED)")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Emergency Services Hospital Coverage", className="text-danger"),
                            html.Ul([
                                html.Li(f"Emergency services hospitals: {len(emergency_hospitals)}"),
                                html.Li("Coverage from emergency hospitals only: 76.6%"),
                                html.Li("Communities covered: 47/92 (51.1%)"),
                                html.Li("Population covered: 742,623/969,383"),
                                html.Li("Uses hospitals with actual emergency departments")
                            ])
                        ], width=6),
                        dbc.Col([
                            html.H6("Additional Bases Required", className="text-success"),
                            html.Ul([
                                html.Li("Additional bases needed: 39"),
                                html.Li("Final coverage: 100.0%"),
                                html.Li("Total EMS bases: 76"),
                                html.Li("Strategy: Emergency hospital co-location + gap filling"),
                                html.Li("4 fewer bases than original analysis")
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
    
    # Correction details
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔍 Data Correction Details")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("❌ Original Analysis Issues", className="text-danger"),
                                    html.Ul([
                                        html.Li("Used all 48 hospitals from GeoJSON"),
                                        html.Li("Included non-emergency facilities"),
                                        html.Li("Rehabilitation centers, mental health facilities"),
                                        html.Li("Nursing homes and specialized care centers"),
                                        html.Li("Result: 84 total bases for 100% coverage")
                                    ], className="small")
                                ])
                            ], color="danger", outline=True)
                        ], width=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("✅ Corrected Analysis", className="text-success"),
                                    html.Ul([
                                        html.Li("Uses only 37 emergency services hospitals"),
                                        html.Li("Hospitals with actual emergency departments"),
                                        html.Li("Facilities that report ED offload intervals"),
                                        html.Li("Aligned with emergency_health_services.csv"),
                                        html.Li("Result: 76 total bases for 100% coverage")
                                    ], className="small")
                                ])
                            ], color="success", outline=True)
                        ], width=6)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Strategic recommendations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Updated Strategic Recommendations")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("🏥 Method 2: Infrastructure-Based (CORRECTED)", className="text-success"),
                                    html.Ul([
                                        html.Li("Uses 37 emergency services hospitals"),
                                        html.Li("76 total bases for 100% coverage"),
                                        html.Li("39 additional strategic bases"),
                                        html.Li("76.6% initial hospital coverage"),
                                        html.Li("More realistic implementation approach")
                                    ], className="small")
                                ])
                            ], color="success", outline=True)
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("✅ Method 3: Most Efficient", className="text-warning"),
                                    html.Ul([
                                        html.Li("Fewest bases required (45)"),
                                        html.Li("Highest efficiency (2.15% per base)"),
                                        html.Li("96.7% coverage achieved"),
                                        html.Li("Best cost-effectiveness"),
                                        html.Li("Leverages hospital performance data")
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
                        html.H5("🎯 UPDATED RECOMMENDATION", className="alert-heading text-center"),
                        html.Hr(),
                        html.P("📈 For immediate implementation: Method 2 (Emergency Hospital Co-located - 76 bases)", className="mb-1"),
                        html.P("⚡ For optimal efficiency: Method 3 (Hospital-Integrated - 45 bases)", className="mb-1"),
                        html.P("🎯 For maximum coverage guarantee: Method 1 (Population-Only - 80 bases)", className="mb-1"),
                        html.P("🏥 Corrected Method 2 is 4 bases more efficient than original while maintaining emergency focus", className="mb-0 fw-bold")
                    ], color="info", className="mt-3")
                ])
            ])
        ])
    ])
    
], fluid=True)

if __name__ == '__main__':
    print("🚀 Starting CORRECTED Enhanced Method Comparison Dashboard...")
    print(f"📊 Method 1: {len(method1_ems)} bases ({method1_coverage:.1f}% coverage)")
    print(f"📊 Method 2: {len(method2_ems)} bases ({method2_coverage:.1f}% coverage)")
    print(f"📊 Method 3 (CORRECTED): {len(method3_ems)} bases ({method3_coverage:.1f}% coverage)")
    print(f"📊 Emergency Services Hospitals: {len(emergency_hospitals)} facilities")
    print("Dashboard running on http://127.0.0.1:8063/")
    app.run_server(debug=True, host='127.0.0.1', port=8063)

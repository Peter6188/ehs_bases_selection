#!/usr/bin/env python3
"""
Comprehensive Multi-Tab EMS Base Optimization Dashboard
Combines all analyses into a single tabbed interface:
- Tab 1: Three-Method Comparison Overview
- Tab 2: Method 1: Population-Only Analysis
- Tab 3: Method 2: Emergency Hospital Co-located Analysis
- Tab 4: Method 3: Hospital Performance-Integrated Analysis
- Tab 5: Strategic Recommendations
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Load data for all analyses
def load_all_data():
    """Load data for comprehensive multi-tab dashboard"""
    print("📊 Loading Comprehensive Multi-Tab Dashboard data...")
    
    # Method 1: Population-only EMS locations (80 bases)
    method1_ems = pd.read_csv('../../data/processed/optimal_ems_locations_80bases_complete_coverage.csv')
    
    # Method 2: CORRECTED Hospital co-located + additional EMS locations (76 bases)
    method2_ems = pd.read_csv('../../data/processed/corrected_hospital_colocated_ems_locations.csv')
    
    # Method 3: Hospital-performance-integrated EMS locations (45 bases)
    method3_ems = pd.read_csv('../../data/processed/hospital_integrated_ems_locations_45bases.csv')
    
    # Load emergency services hospital data
    ems_df = pd.read_csv('../../data/raw/emergency_health_services.csv')
    emergency_hospitals = ems_df['Hospital'].dropna().unique()
    emergency_hospitals = [h for h in emergency_hospitals if h != 'Hospital']
    
    # Load communities data
    pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
    
    print(f"✅ Loaded: Method 1: {len(method1_ems)} bases, Method 2: {len(method2_ems)} bases, Method 3: {len(method3_ems)} bases")
    print(f"✅ Emergency Services Hospitals: {len(emergency_hospitals)} facilities")
    
    return method1_ems, method2_ems, method3_ems, emergency_hospitals, pop_df

# Load data
method1_ems, method2_ems, method3_ems, emergency_hospitals, pop_df = load_all_data()

# Calculate metrics
method1_coverage = 100.0
method2_coverage = 100.0
method3_coverage = 96.7

# Create comparison charts
def create_overview_charts():
    """Create overview comparison charts"""
    
    # Chart 1: Base count comparison
    methods = ['Method 1\n(Population-Only)', 'Method 2\n(Emergency Hospital\nCo-located)', 'Method 3\n(Hospital-Integrated\n96.7%)']
    base_counts = [len(method1_ems), len(method2_ems), len(method3_ems)]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    fig1 = go.Figure(data=[
        go.Bar(
            x=methods,
            y=base_counts,
            marker_color=colors,
            text=base_counts,
            textposition='auto',
        )
    ])
    fig1.update_layout(
        title="EMS Base Count Comparison - All Methods",
        yaxis_title="Number of EMS Bases",
        showlegend=False,
        height=400
    )
    
    # Chart 2: Coverage comparison
    coverages = [method1_coverage, method2_coverage, method3_coverage, 100.0]
    fig2 = go.Figure(data=[
        go.Bar(
            x=methods,
            y=coverages,
            marker_color=colors,
            text=[f'{c:.1f}%' for c in coverages],
            textposition='auto',
        )
    ])
    fig2.update_layout(
        title="Population Coverage Comparison",
        yaxis_title="Coverage Percentage (%)",
        yaxis=dict(range=[90, 102]),
        showlegend=False,
        height=400
    )
    
    # Chart 3: Efficiency comparison (Coverage per base)
    efficiencies = [
        method1_coverage / len(method1_ems),
        method2_coverage / len(method2_ems),
        method3_coverage / len(method3_ems)
    ]
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=methods,
            y=efficiencies,
            marker_color=colors,
            text=[f'{e:.2f}' for e in efficiencies],
            textposition='auto',
        )
    ])
    fig3.update_layout(
        title="Efficiency Comparison (Coverage % per Base)",
        yaxis_title="Coverage per Base",
        showlegend=False,
        height=400
    )
    
    return fig1, fig2, fig3

# Create method-specific maps
def create_method_maps():
    """Create maps for all methods"""
    
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
    fig1.update_layout(height=500)
    
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
    fig2.update_layout(height=500)
    
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
    fig3.update_layout(height=500)
    
    return fig1, fig2, fig3

overview_charts = create_overview_charts()
method_maps = create_method_maps()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Comprehensive EMS Base Optimization Dashboard"

# Tab content functions
def create_overview_tab():
    """Create the overview tab content"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🏥 EMS Base Optimization: Three-Method Comparison", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("📊 COMPREHENSIVE ANALYSIS", className="alert-heading"),
                    html.P("Complete comparison of three optimization approaches with corrected emergency services data", className="mb-0")
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
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Method 2", className="text-success text-center"),
                        html.H5("Emergency Hospital Co-located", className="text-center text-muted"),
                        html.H2(f"{len(method2_ems)}", className="text-center"),
                        html.P("EMS Bases", className="text-center"),
                        html.P("(37 are current emergency hospitals)", className="text-center text-muted small"),
                        html.P(f"{method2_coverage:.1f}% Coverage", className="text-center text-success")
                    ])
                ], color="success", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Method 3", className="text-warning text-center"),
                        html.H5("Hospital-Integrated", className="text-center text-muted"),
                        html.H2(f"{len(method3_ems)}", className="text-center"),
                        html.P("EMS Bases", className="text-center"),
                        html.P("(All 45 are new bases)", className="text-center text-muted small"),
                        html.P(f"{method3_coverage:.1f}% Coverage", className="text-center text-success")
                    ])
                ], color="warning", outline=True)
            ], width=4),
        ], className="mb-4"),
        
        # Comparison charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 Comprehensive Method Comparison")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=overview_charts[0])], width=4),
                            dbc.Col([dcc.Graph(figure=overview_charts[1])], width=4),
                            dbc.Col([dcc.Graph(figure=overview_charts[2])], width=4),
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Key insights
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Key Insights & Strategic Recommendations")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🏆 Most Efficient (95% Coverage)", className="text-warning"),
                                html.P("Method 3: 45 bases (2.15% coverage per base)", className="fw-bold"),
                                html.Ul([
                                    html.Li("Best cost-effectiveness for 95% target"),
                                    html.Li("All new bases, performance-optimized"),
                                    html.Li("Leverages hospital performance data"),
                                    html.Li("Ideal for quality-focused deployment")
                                ], className="small")
                            ], width=3),
                            
                            dbc.Col([
                                html.H6("🏥 Most Practical (100% Coverage)", className="text-success"),
                                html.P("Method 2: 76 bases (1.32% coverage per base)", className="fw-bold"),
                                html.Ul([
                                    html.Li("Leverages existing emergency infrastructure"),
                                    html.Li("37 hospitals + 39 additional bases"),
                                    html.Li("Realistic implementation timeline"),
                                    html.Li("Lower operational startup costs")
                                ], className="small")
                            ], width=3),
                            
                            dbc.Col([
                                html.H6("📊 Most Comprehensive", className="text-primary"),
                                html.P("Method 1: 80 bases (1.25% coverage per base)", className="fw-bold"),
                                html.Ul([
                                    html.Li("Pure population-density optimization"),
                                    html.Li("No infrastructure dependencies"),
                                    html.Li("Guaranteed 100% coverage"),
                                    html.Li("Traditional K-means approach")
                                ], className="small")
                            ], width=3),
                            
                        ])
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_method1_tab():
    """Create Method 1 detailed analysis tab"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("📊 Method 1: Population-Only K-means Analysis", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🎯 PURE POPULATION OPTIMIZATION", className="alert-heading"),
                    html.P("Traditional K-means clustering based solely on population density and geographic distribution", className="mb-0")
                ], color="primary", className="mb-3")
            ])
        ]),
        
        # Method 1 metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("80", className="text-center text-primary"),
                        html.P("Total EMS Bases", className="text-center"),
                    ])
                ], color="primary", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("100.0%", className="text-center text-success"),
                        html.P("Population Coverage", className="text-center"),
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("1.25", className="text-center text-info"),
                        html.P("Coverage % per Base", className="text-center"),
                    ])
                ], color="info", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("All New", className="text-center text-warning"),
                        html.P("Base Type", className="text-center"),
                    ])
                ], color="warning", outline=True)
            ], width=3),
        ], className="mb-4"),
        
        # Method 1 map and details
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=method_maps[0])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 Method 1 Characteristics")),
                    dbc.CardBody([
                        html.H6("✅ Advantages", className="text-success"),
                        html.Ul([
                            html.Li("Guaranteed 100% coverage"),
                            html.Li("No infrastructure dependencies"),
                            html.Li("Optimal population distribution"),
                            html.Li("Well-established methodology"),
                            html.Li("Predictable performance")
                        ], className="small"),
                        
                        html.H6("⚠️ Considerations", className="text-warning mt-3"),
                        html.Ul([
                            html.Li("Requires 80 new bases"),
                            html.Li("Higher initial infrastructure cost"),
                            html.Li("Ignores existing hospital resources"),
                            html.Li("No performance optimization"),
                            html.Li("May not align with current facilities")
                        ], className="small"),
                        
                        html.H6("🎯 Best Use Case", className="text-primary mt-3"),
                        html.P("When complete geographic coverage is mandatory and existing infrastructure cannot be leveraged", className="small")
                    ])
                ])
            ], width=4)
        ])
    ], fluid=True)

def create_method2_tab():
    """Create Method 2 detailed analysis tab"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🏥 Method 2: Emergency Hospital Co-located Analysis", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🏥 INFRASTRUCTURE-LEVERAGED OPTIMIZATION", className="alert-heading"),
                    html.P("Strategically leverages existing emergency hospital infrastructure with 37 emergency services hospitals", className="mb-0")
                ], color="success", className="mb-3")
            ])
        ]),
        
        # Method 2 metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("76", className="text-center text-success"),
                        html.P("Total EMS Bases", className="text-center"),
                        html.P("37 + 39", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("37", className="text-center text-danger"),
                        html.P("Emergency Hospitals", className="text-center"),
                        html.P("Existing Infrastructure", className="text-center text-muted small")
                    ])
                ], color="danger", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("39", className="text-center text-warning"),
                        html.P("Additional Bases", className="text-center"),
                        html.P("Strategic Gap Filling", className="text-center text-muted small")
                    ])
                ], color="warning", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("100.0%", className="text-center text-success"),
                        html.P("Population Coverage", className="text-center"),
                    ])
                ], color="success", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("76.6%", className="text-center text-info"),
                        html.P("Hospital-Only Coverage", className="text-center"),
                    ])
                ], color="info", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("1.32", className="text-center text-primary"),
                        html.P("Coverage % per Base", className="text-center"),
                    ])
                ], color="primary", outline=True)
            ], width=2),
        ], className="mb-4"),
        
        # Method 2 breakdown
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=method_maps[1])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 Method 2 Breakdown")),
                    dbc.CardBody([
                        html.H6("🏥 Emergency Hospital Coverage", className="text-danger"),
                        html.Ul([
                            html.Li("37 emergency services hospitals"),
                            html.Li("742,623/969,383 population covered"),
                            html.Li("47/92 communities covered"),
                            html.Li("Uses hospitals with actual EDs"),
                            html.Li("Based on ED offload interval data")
                        ], className="small"),
                        
                        html.H6("➕ Additional Strategic Bases", className="text-success mt-3"),
                        html.Ul([
                            html.Li("39 additional bases needed"),
                            html.Li("Covers remaining 226,760 population"),
                            html.Li("Fills geographic gaps"),
                            html.Li("Ensures 100% coverage"),
                            html.Li("4 fewer bases than original analysis")
                        ], className="small"),
                        
                        html.H6("🎯 Implementation Advantages", className="text-primary mt-3"),
                        html.P("Lower startup costs, faster deployment, leverages existing emergency infrastructure", className="small")
                    ])
                ])
            ], width=4)
        ])
    ], fluid=True)

def create_method3_tab():
    """Create Method 3 detailed analysis tab"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("⚡ Method 3: Hospital-Performance-Integrated Analysis", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("🎯 PERFORMANCE-OPTIMIZED EFFICIENCY", className="alert-heading"),
                    html.P("Advanced optimization leveraging hospital performance data for maximum efficiency at 95% coverage target", className="mb-0")
                ], color="warning", className="mb-3")
            ])
        ]),
        
        # Method 3 metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("45", className="text-center text-warning"),
                        html.P("Total EMS Bases", className="text-center"),
                        html.P("Fewest Required", className="text-center text-muted small")
                    ])
                ], color="warning", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("96.7%", className="text-center text-success"),
                        html.P("Population Coverage", className="text-center"),
                        html.P("Target: 95%", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("2.15", className="text-center text-success"),
                        html.P("Coverage % per Base", className="text-center"),
                        html.P("Highest Efficiency", className="text-center text-muted small")
                    ])
                ], color="success", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("0", className="text-center text-info"),
                        html.P("Existing Hospitals Used", className="text-center"),
                        html.P("All New Bases", className="text-center text-muted small")
                    ])
                ], color="info", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("2", className="text-center text-primary"),
                        html.P("Close to Hospitals", className="text-center"),
                        html.P("≤2km Distance", className="text-center text-muted small")
                    ])
                ], color="primary", outline=True)
            ], width=2),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H3("126", className="text-center text-danger"),
                        html.P("For 100% Coverage", className="text-center"),
                        html.P("+81 Additional", className="text-center text-muted small")
                    ])
                ], color="danger", outline=True)
            ], width=2),
        ], className="mb-4"),
        
        # Method 3 detailed analysis
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=method_maps[2])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 Method 3 Analysis")),
                    dbc.CardBody([
                        html.H6("⚡ Performance Optimization", className="text-warning"),
                        html.Ul([
                            html.Li("Leverages hospital performance data"),
                            html.Li("Optimized for 95% coverage target"),
                            html.Li("Highest efficiency (2.15% per base)"),
                            html.Li("Strategic placement independent of infrastructure"),
                            html.Li("Quality-focused approach")
                        ], className="small"),
                        
                        html.H6("📍 Base Distribution", className="text-info mt-3"),
                        html.Ul([
                            html.Li("45 strategically placed new bases"),
                            html.Li("2 bases close to emergency hospitals (≤2km)"),
                            html.Li("8 bases nearby hospitals (2-10km)"),
                            html.Li("35 bases distant from hospitals (>10km)"),
                            html.Li("Performance-driven placement")
                        ], className="small"),
                        
                        html.H6("🎯 Best Use Case", className="text-success mt-3"),
                        html.P("When 95% coverage is acceptable and maximum efficiency is required", className="small")
                    ])
                ])
            ], width=4)
        ])
    ], fluid=True)

def create_recommendations_tab():
    """Create final recommendations and summary tab"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H2("🎯 Strategic Recommendations & Executive Summary", className="text-center mb-4"),
                dbc.Alert([
                    html.H5("📋 FINAL ANALYSIS & RECOMMENDATIONS", className="alert-heading"),
                    html.P("Comprehensive strategic guidance based on three-method comparison with corrected emergency services data", className="mb-0")
                ], color="info", className="mb-3")
            ])
        ]),
        
        # Executive summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📈 Executive Summary")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🏆 RECOMMENDED APPROACH", className="text-success"),
                                dbc.Alert([
                                    html.H5("Method 2: Emergency Hospital Co-located", className="alert-heading text-success"),
                                    html.P("76 bases (37 hospitals + 39 additional) for 100% coverage", className="mb-1"),
                                    html.P("Most practical and cost-effective for immediate implementation", className="mb-0 fw-bold")
                                ], color="success")
                            ], width=6),
                            dbc.Col([
                                html.H6("⚡ ALTERNATIVE FOR EFFICIENCY", className="text-warning"),
                                dbc.Alert([
                                    html.H5("Method 3: Hospital-Performance-Integrated", className="alert-heading text-warning"),
                                    html.P("45 bases for 96.7% coverage", className="mb-1"),
                                    html.P("Most efficient if 95% coverage target is acceptable", className="mb-0 fw-bold")
                                ], color="warning")
                            ], width=6)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Detailed recommendations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Strategic Decision Framework")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("🚀 For Immediate Implementation", className="text-success"),
                                        html.P("Method 2: Emergency Hospital Co-located", className="fw-bold text-success"),
                                        html.Ul([
                                            html.Li("Leverages existing emergency infrastructure"),
                                            html.Li("Lower startup and operational costs"),
                                            html.Li("Faster deployment timeline"),
                                            html.Li("76 bases for 100% coverage"),
                                            html.Li("Most realistic implementation path")
                                        ], className="small")
                                    ])
                                ], color="success", outline=True)
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("⚡ For Maximum Efficiency", className="text-warning"),
                                        html.P("Method 3: Hospital-Performance-Integrated", className="fw-bold text-warning"),
                                        html.Ul([
                                            html.Li("Most cost-effective (45 bases)"),
                                            html.Li("Quality-focused optimization"),
                                            html.Li("96.7% coverage achieved"),
                                            html.Li("Best performance metrics"),
                                            html.Li("Ideal for budget constraints")
                                        ], className="small")
                                    ])
                                ], color="warning", outline=True)
                            ], width=4),
                            
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("📊 For Complete Coverage", className="text-primary"),
                                        html.P("Method 1: Population-Only", className="fw-bold text-primary"),
                                        html.Ul([
                                            html.Li("Guaranteed 100% coverage"),
                                            html.Li("No infrastructure dependencies"),
                                            html.Li("Traditional proven approach"),
                                            html.Li("80 bases required"),
                                            html.Li("When hospitals cannot be leveraged")
                                        ], className="small")
                                    ])
                                ], color="primary", outline=True)
                            ], width=4)
                        ])
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Implementation roadmap
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🗺️ Implementation Roadmap")),
                    dbc.CardBody([
                        html.H6("Phase 1: Infrastructure Assessment (Months 1-3)", className="text-primary"),
                        html.Ul([
                            html.Li("Verify emergency hospital capacity and readiness"),
                            html.Li("Assess 37 emergency hospitals for EMS co-location"),
                            html.Li("Conduct site surveys and feasibility studies"),
                            html.Li("Develop partnerships with hospital administrators")
                        ], className="small mb-3"),
                        
                        html.H6("Phase 2: Strategic Base Deployment (Months 4-12)", className="text-success"),
                        html.Ul([
                            html.Li("Deploy EMS services at 37 emergency hospitals"),
                            html.Li("Identify and secure locations for 39 additional bases"),
                            html.Li("Begin construction/setup of additional facilities"),
                            html.Li("Train and deploy EMS personnel")
                        ], className="small mb-3"),
                        
                        html.H6("Phase 3: Optimization & Monitoring (Months 13-18)", className="text-warning"),
                        html.Ul([
                            html.Li("Monitor coverage and response time metrics"),
                            html.Li("Optimize base operations and resource allocation"),
                            html.Li("Assess performance against targets"),
                            html.Li("Consider Method 3 transition for high-efficiency areas")
                        ], className="small")
                    ])
                ])
            ])
        ])
    ], fluid=True)

# Main app layout with tabs
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🏥 Comprehensive EMS Base Optimization Dashboard", className="text-center mb-4"),
            html.P("Multi-Method Analysis with Emergency Services Data Integration", className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="overview", active_tab_style={"background-color": "#007bff", "color": "white"}),
        dbc.Tab(label="📈 Method 1: Population-Only", tab_id="method1"),
        dbc.Tab(label="🏥 Method 2: Hospital Co-located", tab_id="method2"),
        dbc.Tab(label="⚡ Method 3: Performance-Integrated", tab_id="method3"),
        dbc.Tab(label="🎯 Recommendations", tab_id="recommendations"),
    ], id="tabs", active_tab="overview", className="mb-4"),
    
    html.Div(id="tab-content")
], fluid=True)

# Callback for tab content
@app.callback(Output("tab-content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "overview":
        return create_overview_tab()
    elif active_tab == "method1":
        return create_method1_tab()
    elif active_tab == "method2":
        return create_method2_tab()
    elif active_tab == "method3":
        return create_method3_tab()
    elif active_tab == "recommendations":
        return create_recommendations_tab()
    else:
        return create_overview_tab()

if __name__ == '__main__':
    print("🚀 Starting Comprehensive Multi-Tab EMS Dashboard...")
    print(f"📊 Method 1: {len(method1_ems)} bases ({method1_coverage:.1f}% coverage)")
    print(f"📊 Method 2: {len(method2_ems)} bases ({method2_coverage:.1f}% coverage)")
    print(f"📊 Method 3: {len(method3_ems)} bases ({method3_coverage:.1f}% coverage)")
    print(f"📊 Emergency Services Hospitals: {len(emergency_hospitals)} facilities")
    print("Dashboard running on http://127.0.0.1:8064/")
    app.run_server(debug=True, host='127.0.0.1', port=8064)

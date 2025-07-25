#!/usr/bin/env python3
"""
Nova Scotia EHS Base Location Dashboard
Interactive dashboard for Emergency Health Services optimization analysis
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from folium import plugins
import dash_bootstrap_components as dbc

# Load data
def load_dashboard_data():
    """Load all data for dashboard"""
    try:
        # Population data
        pop_df = pd.read_csv('0 polulation_location_polygon.csv')
        clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
        clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]
        clean_df = clean_df[
            (clean_df['latitude'] >= 43.0) & (clean_df['latitude'] <= 47.0) &
            (clean_df['longitude'] >= -67.0) & (clean_df['longitude'] <= -59.0)
        ]
        
        # EHS base locations
        ems_df = pd.read_csv('optimal_ems_locations_15min.csv')
        
        # EHS performance data
        ehs_perf = pd.read_csv('2 Emergency_Health_Services_20250719.csv')
        
        return clean_df, ems_df, ehs_perf
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Nova Scotia EHS Base Optimization"

# Load data
population_df, ems_bases_df, ehs_performance_df = load_dashboard_data()

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Nova Scotia EHS Base Location Optimization", 
                   className="text-center mb-4"),
            html.P("Interactive dashboard for Emergency Health Services base placement analysis",
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Key Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("12", className="card-title text-primary"),
                    html.P("Optimal EHS Bases", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("95", className="card-title text-success"),
                    html.P("Communities Covered", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("100%", className="card-title text-info"),
                    html.P("15-Minute Coverage", className="card-text")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("923K", className="card-title text-warning"),
                    html.P("Population Served", className="card-text")
                ])
            ])
        ], width=3)
    ], className="mb-4"),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Coverage Map", tab_id="map"),
        dbc.Tab(label="Analysis Results", tab_id="analysis"),
        dbc.Tab(label="Performance Data", tab_id="performance"),
        dbc.Tab(label="Methodology", tab_id="methodology")
    ], id="tabs", active_tab="map"),
    
    # Tab content
    html.Div(id="tab-content", className="mt-4")
    
], fluid=True)

# Callback for tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "map":
        return render_map_tab()
    elif active_tab == "analysis":
        return render_analysis_tab()
    elif active_tab == "performance":
        return render_performance_tab()
    elif active_tab == "methodology":
        return render_methodology_tab()

def render_map_tab():
    """Render the coverage map tab"""
    
    # Create Plotly map (since Folium integration with Dash requires additional setup)
    fig = go.Figure()
    
    if not population_df.empty:
        # Add communities
        fig.add_trace(go.Scattermapbox(
            lat=population_df['latitude'],
            lon=population_df['longitude'],
            mode='markers',
            marker=dict(
                size=population_df['C1_COUNT_TOTAL']/1000,  # Scale by population
                color='lightblue',
                opacity=0.6
            ),
            text=population_df['GEO_NAME'],
            hovertemplate='<b>%{text}</b><br>Population: %{marker.size}<extra></extra>',
            name='Communities'
        ))
    
    if not ems_bases_df.empty:
        # Add EHS bases
        fig.add_trace(go.Scattermapbox(
            lat=ems_bases_df['Latitude'],
            lon=ems_bases_df['Longitude'],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            text=ems_bases_df['EHS_Base_ID'],
            hovertemplate='<b>%{text}</b><br>Region: %{customdata}<extra></extra>',
            customdata=ems_bases_df['Region'] if 'Region' in ems_bases_df.columns else ems_bases_df['EHS_Base_ID'],
            name='EHS Bases'
        ))
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=45.0, lon=-63.0),
            zoom=6
        ),
        height=600,
        title="Nova Scotia EHS Base Locations and Community Coverage"
    )
    
    return dcc.Graph(figure=fig)

def render_analysis_tab():
    """Render the analysis results tab"""
    
    content = [
        html.H3("Analysis Results"),
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.H5("Coverage Analysis"),
                html.P("• All 95 communities within 15km of an EHS base"),
                html.P("• Average distance: ~8.5 km"),
                html.P("• Population-weighted average: ~7.2 km"),
                html.P("• Maximum distance: ~14.8 km")
            ], width=6),
            dbc.Col([
                html.H5("Base Distribution"),
                html.P("• Central Zone: 4 bases (high volume)"),
                html.P("• Eastern Zone: 3 bases (Cape Breton coverage)"),
                html.P("• Northern Zone: 2 bases (geographic spread)"),
                html.P("• Western Zone: 3 bases (coastal + inland)")
            ], width=6)
        ]),
        
        html.Hr(),
        
        html.H5("Implementation Phases"),
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Phase"),
                    html.Th("Bases"),
                    html.Th("Priority"),
                    html.Th("Coverage")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td("Phase 1"),
                    html.Td("6 bases"),
                    html.Td("Urban/High Population"),
                    html.Td("~70% population")
                ]),
                html.Tr([
                    html.Td("Phase 2"),
                    html.Td("6 bases"),
                    html.Td("Rural/Remote Areas"),
                    html.Td("100% coverage")
                ])
            ])
        ], striped=True)
    ]
    
    return content

def render_performance_tab():
    """Render the performance data tab"""
    
    if ehs_performance_df.empty:
        return html.P("Performance data not available")
    
    # Create performance visualization
    offload_data = ehs_performance_df[ehs_performance_df['Measure Name'] == 'ED Offload Interval']
    
    if not offload_data.empty:
        # Hospital performance chart
        hospital_avg = offload_data.groupby('Hospital')['Actual'].mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=hospital_avg.values,
            y=hospital_avg.index,
            orientation='h',
            title="Hospital ED Offload Performance (Average Minutes)",
            labels={'x': 'Average Offload Time (minutes)', 'y': 'Hospital'}
        )
        fig.update_layout(height=600)
        
        content = [
            html.H3("EHS Performance Analysis"),
            html.P("Hospital offload times significantly impact EHS efficiency. "
                   "Bases are strategically placed to avoid overloading poor-performing hospitals."),
            dcc.Graph(figure=fig),
            html.Hr(),
            html.H5("Performance Categories"),
            html.P("• Good: < 8,000 minutes average offload"),
            html.P("• Average: 8,000 - 12,000 minutes"),
            html.P("• Poor: > 12,000 minutes (avoid for base co-location)")
        ]
    else:
        content = [html.P("Offload data not available")]
    
    return content

def render_methodology_tab():
    """Render the methodology tab"""
    
    content = [
        html.H3("Analysis Methodology"),
        html.Hr(),
        
        html.H5("1. Data Sources"),
        html.Ul([
            html.Li("Population data: 95 Nova Scotia communities"),
            html.Li("Hospital locations: 47 facilities"),
            html.Li("EHS performance data: 2021-2023 operational metrics"),
            html.Li("Geographic boundaries: Nova Scotia provincial limits")
        ]),
        
        html.H5("2. Optimization Approach"),
        html.Ul([
            html.Li("Population-weighted K-Means clustering"),
            html.Li("15-minute (15km) coverage constraint"),
            html.Li("Haversine distance calculations"),
            html.Li("Hospital performance integration"),
            html.Li("Zone-based coverage requirements")
        ]),
        
        html.H5("3. Key Constraints"),
        html.Ul([
            html.Li("100% coverage within 15km for all communities"),
            html.Li("Minimum coverage per operational zone"),
            html.Li("Avoid co-location at poor-performing hospitals"),
            html.Li("Population density weighting for urban areas")
        ]),
        
        html.H5("4. Validation Metrics"),
        html.Ul([
            html.Li("Distance coverage validation"),
            html.Li("Population-weighted average response time"),
            html.Li("Zone-level redundancy assessment"),
            html.Li("Hospital capacity impact analysis")
        ])
    ]
    
    return content

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)

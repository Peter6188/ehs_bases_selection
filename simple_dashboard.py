#!/usr/bin/env python3
"""
Simple Dashboard Launcher for Nova Scotia EHS Analysis
"""

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc

# Load data
print("Loading data...")
pop_df = pd.read_csv('0 polulation_location_polygon.csv')
clean_df = pop_df.dropna(subset=['latitude', 'longitude', 'C1_COUNT_TOTAL'])
clean_df = clean_df[clean_df['C1_COUNT_TOTAL'] > 0]

ems_df = pd.read_csv('optimal_ems_locations_15min.csv')
print(f"Loaded {len(clean_df)} communities and {len(ems_df)} EMS bases")

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Simple layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚑 Nova Scotia EHS Base Optimization Dashboard", 
                   className="text-center mb-4"),
            html.H4("Interactive Analysis Results", 
                   className="text-center text-muted mb-5")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Key Results", className="card-title"),
                    html.P(f"Communities Analyzed: {len(clean_df)}", className="card-text"),
                    html.P(f"Optimal EMS Bases: {len(ems_df)}", className="card-text"),
                    html.P("Coverage: 100% within 15 minutes", className="card-text text-success"),
                ])
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🗺️ Geographic Distribution"),
                    dcc.Graph(
                        figure=px.scatter_mapbox(
                            clean_df, 
                            lat="latitude", 
                            lon="longitude",
                            size="C1_COUNT_TOTAL",
                            color_discrete_sequence=["blue"],
                            zoom=6,
                            height=400,
                            title="Population Centers"
                        ).update_layout(
                            mapbox_style="open-street-map",
                            margin={"r":0,"t":0,"l":0,"b":0}
                        ).add_scattermapbox(
                            lat=ems_df['Latitude'],
                            lon=ems_df['Longitude'],
                            mode="markers",
                            marker=dict(size=15, color="red", symbol="hospital"),
                            name="EMS Bases",
                            text=ems_df['EHS_Base_ID'] + " - " + ems_df['Region']
                        )
                    )
                ])
            ])
        ], width=8)
    ])
], fluid=True)

if __name__ == '__main__':
    print("🚀 Starting EHS Dashboard...")
    print("📱 Dashboard will be available at: http://127.0.0.1:8050")
    print("🔄 Press Ctrl+C to stop the server")
    app.run_server(debug=True, host='127.0.0.1', port=8050)

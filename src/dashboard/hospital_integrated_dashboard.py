#!/usr/bin/env python3
"""
Enhanced EHS Dashboard with Hospital Performance Integration
Displays the hospital-performance-integrated EMS base optimization results
"""

import dash
from dash import dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from pyproj import Transformer

# Load data
def load_and_process_data():
    """Load all required data for the enhanced dashboard"""
    print("📊 Loading hospital-integrated EMS optimization data...")
    
    # Load optimal EMS locations (hospital-integrated)
    ems_df = pd.read_csv('../../data/processed/hospital_integrated_ems_locations_45bases.csv')
    
    # Load community assignments
    communities_df = pd.read_csv('../analysis/hospital_integrated_community_assignments_45bases.csv')
    
    # Load hospital performance analysis
    hospitals_df = pd.read_csv('../analysis/hospital_performance_analysis.csv')
    
    # Load original population data for full context
    pop_df = pd.read_csv('../../data/raw/population_location_polygon.csv')
    
    # Transform coordinates
    transformer = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
    lon_deg, lat_deg = transformer.transform(pop_df['longitude'].values, pop_df['latitude'].values)
    pop_df['lat_deg'] = lat_deg
    pop_df['lon_deg'] = lon_deg
    
    # Filter for Nova Scotia
    pop_df = pop_df[
        (pop_df['lat_deg'] >= 43.0) & (pop_df['lat_deg'] <= 47.0) &
        (pop_df['lon_deg'] >= -67.0) & (pop_df['lon_deg'] <= -59.0) &
        (pop_df['C1_COUNT_TOTAL'] > 0)
    ].dropna()
    
    print(f"✅ Loaded: {len(ems_df)} EMS bases, {len(hospitals_df)} hospitals, {len(communities_df)} communities")
    
    return ems_df, hospitals_df, communities_df, pop_df

def create_main_map(ems_df, hospitals_df, communities_df, pop_df):
    """Create the main interactive map"""
    
    # Create map figure
    fig = go.Figure()
    
    # Add all population centers (background context)
    fig.add_trace(go.Scattermapbox(
        lat=pop_df['lat_deg'],
        lon=pop_df['lon_deg'],
        mode='markers',
        marker=dict(
            size=np.sqrt(pop_df['C1_COUNT_TOTAL']/1000) * 3,
            color='lightgray',
            opacity=0.4
        ),
        text=pop_df['GEO_NAME'],
        name='All Communities',
        hovertemplate='<b>%{text}</b><br>Population: %{marker.size}<extra></extra>'
    ))
    
    # Add analyzed communities colored by coverage gap
    fig.add_trace(go.Scattermapbox(
        lat=communities_df['latitude'],
        lon=communities_df['longitude'],
        mode='markers',
        marker=dict(
            size=np.sqrt(communities_df['C1_COUNT_TOTAL']/1000) * 4,
            color=communities_df['coverage_gap_score'],
            colorscale='RdYlBu_r',
            colorbar=dict(title="Coverage Gap Score", x=1.02),
            opacity=0.8
        ),
        text=communities_df['GEO_NAME'],
        customdata=np.column_stack((
            communities_df['C1_COUNT_TOTAL'],
            communities_df['coverage_gap_score'],
            communities_df['assigned_ems_base'],
            communities_df['distance_to_ems'],
            communities_df['nearest_hospital_distance']
        )),
        name='Communities (by Coverage Gap)',
        hovertemplate='''
        <b>%{text}</b><br>
        Population: %{customdata[0]:,}<br>
        Coverage Gap Score: %{customdata[1]:.3f}<br>
        Assigned EMS Base: %{customdata[2]}<br>
        Distance to EMS: %{customdata[3]:.1f} km<br>
        Distance to Hospital: %{customdata[4]:.1f} km
        <extra></extra>'''
    ))
    
    # Add hospitals colored by performance
    fig.add_trace(go.Scattermapbox(
        lat=hospitals_df['latitude'],
        lon=hospitals_df['longitude'],
        mode='markers',
        marker=dict(
            size=15,
            color=hospitals_df['overall_performance'],
            colorscale='RdYlGn',
            opacity=0.9,
            colorbar=dict(title="Hospital Performance", x=1.1)
        ),
        text=hospitals_df['facility'],
        customdata=np.column_stack((
            hospitals_df['type'],
            hospitals_df['overall_performance'],
            hospitals_df['ed_offload_avg'],
            hospitals_df['ehs_response_avg']
        )),
        name='Hospitals (by Performance)',
        hovertemplate='''
        <b>%{text}</b><br>
        Type: %{customdata[0]}<br>
        Performance Score: %{customdata[1]:.3f}<br>
        ED Offload Time: %{customdata[2]:.1f} min<br>
        EHS Response Time: %{customdata[3]:.1f} min
        <extra></extra>'''
    ))
    
    # Add optimal EMS bases
    fig.add_trace(go.Scattermapbox(
        lat=ems_df['Latitude'],
        lon=ems_df['Longitude'],
        mode='markers',
        marker=dict(
            size=20,
            color='red',
            opacity=1.0
        ),
        text=ems_df['EHS_Base_ID'],
        customdata=np.column_stack((
            ems_df['Population_Served'],
            ems_df['Communities_Served'],
            ems_df['Avg_Coverage_Gap_Score'],
            ems_df['Nearest_Hospital'],
            ems_df['Priority_Level']
        )),
        name='EMS Bases (Hospital-Integrated)',
        hovertemplate='''
        <b>%{text}</b><br>
        Population Served: %{customdata[0]:,}<br>
        Communities Served: %{customdata[1]}<br>
        Avg Gap Score: %{customdata[2]:.3f}<br>
        Nearest Hospital: %{customdata[3]}<br>
        Priority: %{customdata[4]}
        <extra></extra>'''
    ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=44.65, lon=-63.75),
            zoom=6
        ),
        height=700,
        title="Nova Scotia EHS Optimization with Hospital Performance Integration",
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig

def create_performance_analysis():
    """Create hospital performance analysis charts"""
    ems_df, hospitals_df, communities_df, pop_df = load_and_process_data()
    
    # Hospital performance distribution
    fig1 = px.histogram(
        hospitals_df, 
        x='overall_performance',
        nbins=15,
        title='Hospital Performance Score Distribution',
        labels={'overall_performance': 'Performance Score', 'count': 'Number of Hospitals'}
    )
    fig1.update_layout(height=400)
    
    # Performance by hospital type
    type_perf = hospitals_df.groupby('type')['overall_performance'].mean().sort_values(ascending=True)
    fig2 = px.bar(
        x=type_perf.values,
        y=type_perf.index,
        orientation='h',
        title='Average Hospital Performance by Type',
        labels={'x': 'Average Performance Score', 'y': 'Hospital Type'}
    )
    fig2.update_layout(height=400)
    
    # Coverage gap vs distance to hospital
    fig3 = px.scatter(
        communities_df,
        x='nearest_hospital_distance',
        y='coverage_gap_score',
        size='C1_COUNT_TOTAL',
        title='Coverage Gap vs Distance to Nearest Hospital',
        labels={
            'nearest_hospital_distance': 'Distance to Nearest Hospital (km)',
            'coverage_gap_score': 'Coverage Gap Score',
            'C1_COUNT_TOTAL': 'Population'
        }
    )
    fig3.update_layout(height=400)
    
    return fig1, fig2, fig3

def create_ems_analysis():
    """Create EMS base analysis charts"""
    ems_df, hospitals_df, communities_df, pop_df = load_and_process_data()
    
    # EMS bases by priority level
    priority_counts = ems_df['Priority_Level'].value_counts()
    fig1 = px.pie(
        values=priority_counts.values,
        names=priority_counts.index,
        title='EMS Bases by Priority Level'
    )
    fig1.update_layout(height=400)
    
    # Population served distribution
    fig2 = px.histogram(
        ems_df,
        x='Population_Served',
        nbins=15,
        title='Population Served per EMS Base',
        labels={'Population_Served': 'Population Served', 'count': 'Number of EMS Bases'}
    )
    fig2.update_layout(height=400)
    
    # Coverage gap score distribution
    fig3 = px.histogram(
        ems_df,
        x='Avg_Coverage_Gap_Score',
        nbins=15,
        title='Average Coverage Gap Score per EMS Base',
        labels={'Avg_Coverage_Gap_Score': 'Average Coverage Gap Score', 'count': 'Number of EMS Bases'}
    )
    fig3.update_layout(height=400)
    
    return fig1, fig2, fig3

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data for the app
ems_df, hospitals_df, communities_df, pop_df = load_and_process_data()

# Calculate summary statistics
total_population = communities_df['C1_COUNT_TOTAL'].sum()
coverage_percentage = ((communities_df['distance_to_ems'] <= 15).sum() / len(communities_df)) * 100
avg_gap_score = communities_df['coverage_gap_score'].mean()
high_priority_bases = len(ems_df[ems_df['Priority_Level'] == 'High'])

# Create figures
main_map = create_main_map(ems_df, hospitals_df, communities_df, pop_df)
hospital_perf_dist, hospital_perf_type, gap_vs_distance = create_performance_analysis()
priority_pie, pop_served_hist, gap_score_hist = create_ems_analysis()

# Define the app layout
app.layout = html.Div([
    html.H1("Nova Scotia EHS Optimization Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
    
    html.H2("Hospital Performance Integration Analysis", 
            style={'textAlign': 'center', 'marginBottom': 20, 'color': '#e74c3c'}),
    
    # Key metrics
    html.Div([
        html.Div([
            html.H3(f"{len(ems_df)}", style={'fontSize': 48, 'margin': 0, 'color': '#e74c3c'}),
            html.P("EMS Bases", style={'fontSize': 18, 'margin': 0})
        ], className='metric-box', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#f8f9fa', 'border': '2px solid #e74c3c', 'borderRadius': 10, 'margin': 10}),
        
        html.Div([
            html.H3(f"{coverage_percentage:.1f}%", style={'fontSize': 48, 'margin': 0, 'color': '#27ae60'}),
            html.P("Coverage (≤15km)", style={'fontSize': 18, 'margin': 0})
        ], className='metric-box', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#f8f9fa', 'border': '2px solid #27ae60', 'borderRadius': 10, 'margin': 10}),
        
        html.Div([
            html.H3(f"{total_population:,}", style={'fontSize': 48, 'margin': 0, 'color': '#3498db'}),
            html.P("Total Population", style={'fontSize': 18, 'margin': 0})
        ], className='metric-box', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#f8f9fa', 'border': '2px solid #3498db', 'borderRadius': 10, 'margin': 10}),
        
        html.Div([
            html.H3(f"{high_priority_bases}", style={'fontSize': 48, 'margin': 0, 'color': '#f39c12'}),
            html.P("High Priority Bases", style={'fontSize': 18, 'margin': 0})
        ], className='metric-box', style={'textAlign': 'center', 'padding': 20, 'backgroundColor': '#f8f9fa', 'border': '2px solid #f39c12', 'borderRadius': 10, 'margin': 10}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
    
    # Main map
    html.Div([
        dcc.Graph(figure=main_map)
    ], style={'marginBottom': 30}),
    
    # Hospital Performance Analysis
    html.H3("Hospital Performance Analysis", style={'color': '#2c3e50', 'marginBottom': 20}),
    html.Div([
        html.Div([dcc.Graph(figure=hospital_perf_dist)], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=hospital_perf_type)], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=gap_vs_distance)], style={'width': '33%', 'display': 'inline-block'}),
    ]),
    
    # EMS Base Analysis
    html.H3("EMS Base Analysis", style={'color': '#2c3e50', 'marginBottom': 20, 'marginTop': 30}),
    html.Div([
        html.Div([dcc.Graph(figure=priority_pie)], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=pop_served_hist)], style={'width': '33%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=gap_score_hist)], style={'width': '33%', 'display': 'inline-block'}),
    ]),
    
    # Data tables
    html.H3("Detailed Data", style={'color': '#2c3e50', 'marginBottom': 20, 'marginTop': 30}),
    
    html.H4("EMS Bases Summary", style={'marginBottom': 10}),
    dash_table.DataTable(
        data=ems_df.head(10).to_dict('records'),
        columns=[
            {'name': 'EMS Base ID', 'id': 'EHS_Base_ID'},
            {'name': 'Latitude', 'id': 'Latitude', 'type': 'numeric', 'format': {'specifier': '.3f'}},
            {'name': 'Longitude', 'id': 'Longitude', 'type': 'numeric', 'format': {'specifier': '.3f'}},
            {'name': 'Population Served', 'id': 'Population_Served', 'type': 'numeric', 'format': {'specifier': ','}},
            {'name': 'Communities', 'id': 'Communities_Served'},
            {'name': 'Gap Score', 'id': 'Avg_Coverage_Gap_Score', 'type': 'numeric', 'format': {'specifier': '.3f'}},
            {'name': 'Priority', 'id': 'Priority_Level'},
        ],
        style_cell={'textAlign': 'left'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{Priority_Level} eq High'},
                'backgroundColor': '#ffcccc',
            }
        ]
    ),
    
    html.H4("Hospital Performance Summary", style={'marginBottom': 10, 'marginTop': 20}),
    dash_table.DataTable(
        data=hospitals_df.head(10).to_dict('records'),
        columns=[
            {'name': 'Hospital', 'id': 'facility'},
            {'name': 'Type', 'id': 'type'},
            {'name': 'Performance Score', 'id': 'overall_performance', 'type': 'numeric', 'format': {'specifier': '.3f'}},
            {'name': 'ED Offload (min)', 'id': 'ed_offload_avg', 'type': 'numeric', 'format': {'specifier': '.1f'}},
            {'name': 'EHS Response (min)', 'id': 'ehs_response_avg', 'type': 'numeric', 'format': {'specifier': '.1f'}},
        ],
        style_cell={'textAlign': 'left'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{overall_performance} > 0.7'},
                'backgroundColor': '#ccffcc',
            },
            {
                'if': {'filter_query': '{overall_performance} < 0.5'},
                'backgroundColor': '#ffcccc',
            }
        ]
    ),
    
    # Footer
    html.Div([
        html.P("Nova Scotia Emergency Health Services Optimization with Hospital Performance Integration", 
               style={'textAlign': 'center', 'marginTop': 50, 'color': '#7f8c8d'})
    ])
])

if __name__ == '__main__':
    print("🚀 Starting Enhanced EHS Dashboard with Hospital Integration...")
    print("🔗 Dashboard will be available at: http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)

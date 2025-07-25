#!/usr/bin/env python3
"""
Test script to check dashboard dependencies
"""

try:
    import dash
    print("✅ Dash imported successfully")
    
    import dash_bootstrap_components as dbc
    print("✅ Dash Bootstrap Components imported successfully")
    
    import plotly.express as px
    print("✅ Plotly Express imported successfully")
    
    import pandas as pd
    print("✅ Pandas imported successfully")
    
    # Test data loading
    pop_df = pd.read_csv('0 polulation_location_polygon.csv')
    print(f"✅ Population data loaded: {len(pop_df)} records")
    
    ems_df = pd.read_csv('optimal_ems_locations_15min.csv')
    print(f"✅ EMS bases data loaded: {len(ems_df)} bases")
    
    ehs_perf = pd.read_csv('2 Emergency_Health_Services_20250719.csv')
    print(f"✅ EHS performance data loaded: {len(ehs_perf)} records")
    
    print("\n🎉 All dependencies and data files are ready!")
    print("Dashboard is ready to launch!")
    
except Exception as e:
    print(f"❌ Error: {e}")

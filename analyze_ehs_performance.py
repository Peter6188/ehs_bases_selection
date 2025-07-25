#!/usr/bin/env python3
"""
EHS Performance Data Analysis
Analyzing current EHS response times and hospital performance
to inform optimal base location strategy
"""

import pandas as pd
import numpy as np

def analyze_ehs_data():
    print("=== EHS Performance Data Analysis ===")
    
    # Load the EHS performance data
    ehs_df = pd.read_csv('2 Emergency_Health_Services_20250719.csv')
    
    print(f"Dataset shape: {ehs_df.shape}")
    print(f"Date range: {ehs_df['Date'].min()} to {ehs_df['Date'].max()}")
    print()
    
    print("Available zones:")
    print(ehs_df['Zone'].value_counts())
    print()
    
    print("Available measures:")
    print(ehs_df['Measure Name'].value_counts())
    print()
    
    print("Hospitals in dataset:")
    hospitals = ehs_df['Hospital'].unique()
    print(f"Total hospitals: {len(hospitals)}")
    for i, hospital in enumerate(hospitals[:10]):
        print(f"  {i+1}. {hospital}")
    if len(hospitals) > 10:
        print(f"  ... and {len(hospitals)-10} more")
    print()
    
    # Analyze response time metrics
    response_measures = ehs_df[ehs_df['Measure Name'].str.contains('Response|Time', na=False)]
    if not response_measures.empty:
        print("Response time metrics:")
        print(response_measures['Measure Name'].unique())
        print()
        
        # Get latest response times by zone
        latest_date = response_measures['Date'].max()
        latest_response = response_measures[response_measures['Date'] == latest_date]
        
        if not latest_response.empty:
            print(f"Latest response times ({latest_date}):")
            for zone in latest_response['Zone'].unique():
                zone_data = latest_response[latest_response['Zone'] == zone]
                avg_response = zone_data['Actual'].mean()
                print(f"  {zone}: {avg_response:.1f} minutes average")
            print()
    
    # Analyze offload intervals
    offload_data = ehs_df[ehs_df['Measure Name'] == 'ED Offload Interval']
    if not offload_data.empty:
        print("Hospital offload performance (latest data):")
        latest_offload = offload_data.groupby('Hospital')['Actual'].agg(['mean', 'std']).round(1)
        print(latest_offload.head(10))
        print()
        
        # Identify best and worst performing hospitals
        print("Best performing hospitals (lowest offload times):")
        best_hospitals = latest_offload.nsmallest(5, 'mean')
        for hospital, data in best_hospitals.iterrows():
            print(f"  {hospital}: {data['mean']:.1f} min avg")
        print()
        
        print("Worst performing hospitals (highest offload times):")
        worst_hospitals = latest_offload.nlargest(5, 'mean')
        for hospital, data in worst_hospitals.iterrows():
            print(f"  {hospital}: {data['mean']:.1f} min avg")
        print()
    
    # Zone-based analysis
    print("Performance by Zone:")
    for zone in ehs_df['Zone'].unique():
        zone_data = ehs_df[ehs_df['Zone'] == zone]
        hospitals_in_zone = zone_data['Hospital'].nunique()
        print(f"{zone}: {hospitals_in_zone} hospitals")
    print()
    
    return ehs_df

if __name__ == "__main__":
    ehs_data = analyze_ehs_data()

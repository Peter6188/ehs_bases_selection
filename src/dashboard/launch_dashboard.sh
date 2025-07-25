#!/bin/bash

echo "🚀 Nova Scotia EHS Dashboard Launcher"
echo "======================================="

# Navigate to the project directory
cd "/Users/weilei/Documents/0 Courses/15 5893 Health Data Analytics/Capstone/Analysis"

echo "📁 Current directory: $(pwd)"
echo "📊 Checking data files..."

if [ -f "0 polulation_location_polygon.csv" ]; then
    echo "✅ Population data found"
else
    echo "❌ Population data missing"
    exit 1
fi

if [ -f "optimal_ems_locations_15min.csv" ]; then
    echo "✅ EMS base data found"
else
    echo "❌ EMS base data missing"
    exit 1
fi

echo "🐍 Starting Python dashboard..."
echo "🌐 Dashboard will be available at: http://127.0.0.1:8050"
echo "🔄 Press Ctrl+C to stop the server"
echo ""

# Run the dashboard
"/Users/weilei/Documents/0 Courses/15 5893 Health Data Analytics/Capstone/Analysis/.venv-1/bin/python" robust_dashboard.py

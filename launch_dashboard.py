#!/usr/bin/env python3
"""
Dashboard Launcher Script
"""

import subprocess
import sys
import time
import webbrowser

def launch_dashboard():
    print("🚀 Launching Nova Scotia EHS Dashboard...")
    
    try:
        # Start the dashboard in the background
        python_path = "/Users/weilei/Documents/0 Courses/15 5893 Health Data Analytics/Capstone/Analysis/.venv-1/bin/python"
        
        process = subprocess.Popen([
            python_path,
            "simple_dashboard.py"
        ], cwd="/Users/weilei/Documents/0 Courses/15 5893 Health Data Analytics/Capstone/Analysis")
        
        print("⏳ Starting dashboard server...")
        time.sleep(3)  # Give the server time to start
        
        # Open browser
        dashboard_url = "http://127.0.0.1:8050"
        print(f"🌐 Opening dashboard at: {dashboard_url}")
        webbrowser.open(dashboard_url)
        
        print("✅ Dashboard is running!")
        print("🔄 Press Ctrl+C to stop the server")
        
        # Keep the script running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dashboard...")
            process.terminate()
            
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard()

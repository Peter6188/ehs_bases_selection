#!/bin/bash

echo "=== EHS Base Selection GitHub Push Script ==="
echo "Starting Git repository setup and push process..."
echo ""

# Set up error handling
set -e

# Function to check command success
check_command() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 - SUCCESS"
    else
        echo "❌ $1 - FAILED"
        exit 1
    fi
}

# Show current directory
echo "📁 Current directory:"
pwd
echo ""

# Show files to be uploaded
echo "📋 Files to be uploaded:"
ls -la
echo ""

# Initialize Git repository
echo "🔧 Initializing Git repository..."
git init
check_command "Git init"

# Configure Git (you may need to change the email)
echo "⚙️  Configuring Git user..."
git config user.name "Peter6188"
git config user.email "peter6188@example.com"
check_command "Git config"

# Add README header
echo "📝 Adding README header..."
echo "# ehs_bases_selection" >> README.md
check_command "README header added"

# Add README to staging
echo "📋 Adding README.md to staging..."
git add README.md
check_command "Git add README.md"

# First commit
echo "💾 Making first commit..."
git commit -m "first commit"
check_command "First commit"

# Set main branch
echo "🌿 Setting main branch..."
git branch -M main
check_command "Branch main set"

# Add remote origin
echo "🔗 Adding GitHub remote..."
git remote add origin https://github.com/Peter6188/ehs_bases_selection.git
check_command "Remote origin added"

# Push first commit
echo "🚀 Pushing first commit to GitHub..."
git push -u origin main
check_command "First push"

# Add all remaining files
echo "📁 Adding all analysis files..."
git add .
check_command "Git add all files"

# Show what files are staged
echo "📊 Files staged for commit:"
git status --porcelain
echo ""

# Commit all analysis files
echo "💾 Committing all analysis files..."
git commit -m "Add complete EHS base location analysis

- Population-weighted K-Means clustering analysis
- 12 EHS bases for 15-minute coverage of 95 Nova Scotia communities
- Analysis scripts: ems_location_analysis.py, ems_15min_analysis.py
- Results: EMS_15MIN_RESULTS.md, optimal_ems_locations_15min.csv
- Documentation: README.md, requirements.txt
- Data files: population and hospital data
- Jupyter notebook for interactive analysis"
check_command "Analysis files committed"

# Push all files
echo "🚀 Pushing all files to GitHub..."
git push origin main
check_command "Final push"

# Verify remote repository
echo "🔍 Verifying remote repository..."
git remote -v
echo ""

# Show final status
echo "📊 Final Git status:"
git status
echo ""

echo "✅ SUCCESS! All files have been pushed to GitHub."
echo "🌐 Check your repository at: https://github.com/Peter6188/ehs_bases_selection"
echo ""
echo "📋 Repository should now contain:"
echo "   - ems_location_analysis.py (main analysis script)"
echo "   - EMS_15MIN_RESULTS.md (complete results)"
echo "   - optimal_ems_locations_15min.csv (12 EHS base coordinates)"
echo "   - README.md (project documentation)"
echo "   - All data files and supporting scripts"
echo ""
echo "🎯 Your EHS base location analysis is now on GitHub!"

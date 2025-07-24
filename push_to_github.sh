#!/bin/bash

echo "=== Git Repository Setup and Push ==="

# Initialize repository
echo "Initializing Git repository..."
git init

# Configure Git (you may want to change the email)
echo "Configuring Git..."
git config user.name "Peter6188"
git config user.email "peter6188@example.com"

# Add all files
echo "Adding files to staging..."
git add .

# Check status
echo "Git status:"
git status

# Make initial commit
echo "Making initial commit..."
git commit -m "Initial commit: EHS Base Location Analysis

- Complete population-weighted K-Means clustering analysis
- 12 EHS bases for 15-minute coverage of all 95 Nova Scotia communities
- Includes analysis scripts, results, and documentation
- Optimized for emergency response time requirements"

# Set main branch
echo "Setting main branch..."
git branch -M main

# Add remote origin
echo "Adding GitHub remote..."
git remote add origin https://github.com/Peter6188/ehs_bases_selection.git

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

echo "=== Push completed! ==="
echo "Check your repository at: https://github.com/Peter6188/ehs_bases_selection"

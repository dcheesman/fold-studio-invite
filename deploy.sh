#!/bin/bash

# The Fold Studio Invite - Deployment Script
# This script commits changes and pushes to GitHub, triggering Vercel deployment

echo "🚀 Deploying The Fold Studio Invite..."

# Check if there are any changes
if [ -z "$(git status --porcelain)" ]; then
    echo "❌ No changes to commit"
    exit 1
fi

# Add all changes
echo "📝 Adding changes..."
git add .

# Commit with timestamp
echo "💾 Committing changes..."
git commit -m "deploy: $(date '+%Y-%m-%d %H:%M:%S') - Auto-deploy from local"

# Push to GitHub
echo "🌐 Pushing to GitHub..."
git push origin main

echo "✅ Deployment initiated!"
echo "🔗 Check Vercel dashboard for deployment status"
echo "📱 Your site will be live in 1-2 minutes"

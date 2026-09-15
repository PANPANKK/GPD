#!/bin/bash
# Quick Publish Script for GPD Repository
# This script helps you quickly publish the GPD repository to GitHub

set -e

echo "========================================="
echo "GPD Quick Publish to GitHub"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "LICENSE" ]; then
    echo "❌ Error: Please run this script from the GPD-release directory"
    exit 1
fi

# Ask for GitHub username
echo "Step 1: GitHub Username"
read -p "Enter your GitHub username: " USERNAME

if [ -z "$USERNAME" ]; then
    echo "❌ Error: Username cannot be empty"
    exit 1
fi

echo ""
echo "Step 2: Replacing YOUR_USERNAME with $USERNAME..."

# Replace YOUR_USERNAME in all markdown files
find . -name "*.md" -type f -exec sed -i "s/YOUR_USERNAME/$USERNAME/g" {} +

echo "✅ Replaced YOUR_USERNAME in all markdown files"
echo ""

# Ask for repository name
echo "Step 3: Repository Name"
read -p "Enter repository name (default: GPD): " REPO_NAME
REPO_NAME=${REPO_NAME:-GPD}

echo ""
echo "Step 4: Initialize Git Repository"

# Initialize git if not already
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git repository initialized"
else
    echo "⚠️  Git repository already exists"
fi

echo ""
echo "Step 5: Add all files"
git add .

echo ""
echo "Step 6: Create initial commit"
git commit -m "Initial release v1.0.0

- Complete training and evaluation pipeline
- Pre-configured experiments for video and audio modalities
- Comprehensive documentation
- MuDD dataset access agreement
- GitHub templates and contributing guidelines

Co-authored-by: Yao Liu <liuyao@uestc.edu.cn>" || echo "⚠️  Commit failed or no changes to commit"

echo ""
echo "Step 7: Add remote"
git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git" 2>/dev/null || \
    git remote set-url origin "https://github.com/$USERNAME/$REPO_NAME.git"

echo "✅ Remote added: https://github.com/$USERNAME/$REPO_NAME.git"

echo ""
echo "Step 8: Create main branch"
git branch -M main

echo ""
echo "========================================="
echo "Ready to Push!"
echo "========================================="
echo ""
echo "Repository: https://github.com/$USERNAME/$REPO_NAME"
echo ""
echo "Next steps:"
echo "1. Create the repository on GitHub:"
echo "   https://github.com/new"
echo "   - Name: $REPO_NAME"
echo "   - Description: Official implementation of GPD (MM'26)"
echo "   - Public repository"
echo "   - DO NOT initialize with README"
echo ""
echo "2. After creating the repository on GitHub, run:"
echo "   git push -u origin main"
echo ""
echo "3. Create a release:"
echo "   - Go to: https://github.com/$USERNAME/$REPO_NAME/releases/new"
echo "   - Tag: v1.0.0"
echo "   - Title: GPD v1.0.0 - Initial Release"
echo "   - Copy content from RELEASE_NOTES.md"
echo ""
echo "4. Add topics to your repository:"
echo "   deception-detection, knowledge-distillation, multimodal-learning"
echo "   cross-modal, deep-learning, pytorch, mm2026"
echo ""
echo "========================================="
echo "All set! 🚀"
echo "========================================="

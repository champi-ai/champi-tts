#!/bin/bash

# Champi TTS GitHub Issues Update Script
# This script updates the MVP phase issues with proper labels and milestones

set -e

# GitHub Token (required)
# Get your token from: https://github.com/settings/tokens
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Repository
REPO="champi-ai/champi-tts"

# If no token is provided, use interactive approach
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  No GitHub token provided"
    echo "Please set the GITHUB_TOKEN environment variable:"
    echo "  export GITHUB_TOKEN=your_github_token"
    echo ""
    echo "Or run this script with your token:"
    echo "  GITHUB_TOKEN=your_token ./scripts/update-github-issues.sh"
    echo ""
    echo "📖 See docs/phases/update-github-issues.md for manual instructions"
    exit 1
fi

# Helper function to create milestone
create_milestone() {
    local title="$1"
    local description="$2"

    echo "Creating milestone: $title"

    gh api --method POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        "repos/$REPO/milestones" \
        -d "{\"title\": \"$title\", \"description\": \"$description\"}" > /dev/null 2>&1

    echo "  ✓ $title"
}

# Helper function to update issue
update_issue() {
    local issue_num="$1"
    local milestone="$2"
    local labels="$3"

    echo "Updating issue #$issue_num"

    if [ -n "$milestone" ]; then
        gh issue edit "$issue_num" --milestone "$milestone"
        echo "  ✓ Added milestone: $milestone"
    fi

    if [ -n "$labels" ]; then
        for label in $labels; do
            gh issue edit "$issue_num" --add-label "$label"
        done
        echo "  ✓ Added labels: $labels"
    fi
}

echo "🚀 Starting GitHub Issues Update"
echo "================================"
echo "Repository: $REPO"
echo ""

# Step 1: Create milestones
echo "Step 1: Creating milestones..."
echo "------------------------------"

create_milestone \
    "Phase 1: Comprehensive Test Coverage" \
    "Achieve 90%+ test coverage with unit, integration, and end-to-end tests"

create_milestone \
    "Phase 2: Documentation Completeness" \
    "Complete API reference, user guides, and tutorials"

create_milestone \
    "Phase 3: Security Hardening" \
    "Security audit, vulnerability scanning, and best practices"

create_milestone \
    "Phase 4: Packaging Optimization" \
    "PyPI-ready package, dependency optimization, build process"

create_milestone \
    "Phase 5: Performance Benchmarks" \
    "Establish performance baselines and benchmarks"

create_milestone \
    "Phase 6: User Guides and Examples" \
    "Detailed user guides, tutorials, and practical examples"

create_milestone \
    "Phase 7: Release Process Validation" \
    "Validate complete release pipeline including PyPI publishing"

create_milestone \
    "Phase 8: Pre-Release Preparation" \
    "Final cleanup, dependency updates, and v1.0.0 preparation"

echo ""
echo "Step 2: Verifying labels exist..."
echo "----------------------------------"

# Check for required labels
required_labels=("AI" "size" "type")

for label in "${required_labels[@]}"; do
    if gh label list --search "$label" 2>/dev/null | grep -q "$label"; then
        echo "  ✓ Label exists: $label"
    else
        echo "  ⚠️  Label not found: $label (run: gh label create \"$label\" --color \"XXXXXX\" --description \"...\" --yes)"
    fi
done

echo ""
echo "Step 3: Updating issues..."
echo "--------------------------"

# Update issues with appropriate labels and milestones
update_issue \
    15 \
    "Phase 1: Comprehensive Test Coverage" \
    "AI type:testing size:medium"

update_issue \
    16 \
    "Phase 2: Documentation Completeness" \
    "AI type:documentation size:medium"

update_issue \
    17 \
    "Phase 3: Security Hardening" \
    "AI type:security size:medium"

update_issue \
    18 \
    "Phase 4: Packaging Optimization" \
    "AI type:infrastructure size:medium"

update_issue \
    19 \
    "Phase 5: Performance Benchmarks" \
    "AI type:performance size:medium"

update_issue \
    20 \
    "Phase 6: User Guides and Examples" \
    "AI type:documentation size:medium"

update_issue \
    21 \
    "Phase 7: Release Process Validation" \
    "AI type:release size:small"

update_issue \
    22 \
    "Phase 8: Pre-Release Preparation" \
    "AI type:release size:small"

echo ""
echo "✅ GitHub Issues Update Complete!"
echo "=================================="
echo ""
echo "View updated issues:"
echo "  gh issue list --limit 30 --state open --json number,title,labels,milestone"
echo ""
echo "Or visit: https://github.com/$REPO/issues?q=is:issue+is:open"

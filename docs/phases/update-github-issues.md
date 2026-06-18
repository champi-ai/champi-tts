# GitHub Issues Update Guide

This guide explains how to properly update the GitHub issues with labels and milestones for the MVP phases.

## Current Issues to Update

The following issues were created and need to be updated:
- #15 - [INFRA] Comprehensive Test Coverage
- #16 - [FE] Documentation Completeness
- #17 - [INFRA] Security Hardening
- #18 - [INFRA] Packaging Optimization
- #19 - [BE] Performance Benchmarks
- #20 - [FE] User Guides and Examples
- #21 - [INFRA] Release Process Validation
- #22 - [INFRA] Pre-Release Preparation

## Step 1: Create Milestones (If Not Already Created)

Run these commands to create the 8 milestones for each phase:

```bash
# Phase 1
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 1: Comprehensive Test Coverage", "description": "Achieve 90%+ test coverage with unit, integration, and end-to-end tests"}'

# Phase 2
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 2: Documentation Completeness", "description": "Complete API reference, user guides, and tutorials"}'

# Phase 3
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 3: Security Hardening", "description": "Security audit, vulnerability scanning, and best practices"}'

# Phase 4
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 4: Packaging Optimization", "description": "PyPI-ready package, dependency optimization, build process"}'

# Phase 5
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 5: Performance Benchmarks", "description": "Establish performance baselines and benchmarks"}'

# Phase 6
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 6: User Guides and Examples", "description": "Detailed user guides, tutorials, and practical examples"}'

# Phase 7
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 7: Release Process Validation", "description": "Validate complete release pipeline including PyPI publishing"}'

# Phase 8
gh api --method POST -H "Content-Type: application/json" \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  repos/champi-ai/champi-tts/milestones \
  -d '{"title": "Phase 8: Pre-Release Preparation", "description": "Final cleanup, dependency updates, and v1.0.0 preparation"}'
```

## Step 2: Verify Labels Exist

Ensure these labels exist on the repository:
- `AI` (for AI-related work)
- `size` (for issue size: small, medium, large)
- `type` (for issue type: testing, documentation, security, infrastructure, performance, feature, release)

If they don't exist, create them:

```bash
# Create AI label
gh label create "AI" --color "9e6ac3" --description "Work involving AI or ML components"

# Create size label
gh label create "size" --color "fbca04" --description "Issue size: small, medium, large"

# Create type label
gh label create "type" --color "008672" --description "Issue type: testing, documentation, security, infrastructure, performance, feature, release"
```

## Step 3: Update Issues with Labels and Milestones

### Issue #15: Comprehensive Test Coverage
```bash
gh issue edit 15 --milestone "Phase 1: Comprehensive Test Coverage"
gh issue edit 15 --add-label "AI" --add-label "type:testing" --add-label "size:medium"
```

### Issue #16: Documentation Completeness
```bash
gh issue edit 16 --milestone "Phase 2: Documentation Completeness"
gh issue edit 16 --add-label "AI" --add-label "type:documentation" --add-label "size:medium"
```

### Issue #17: Security Hardening
```bash
gh issue edit 17 --milestone "Phase 3: Security Hardening"
gh issue edit 17 --add-label "AI" --add-label "type:security" --add-label "size:medium"
```

### Issue #18: Packaging Optimization
```bash
gh issue edit 18 --milestone "Phase 4: Packaging Optimization"
gh issue edit 18 --add-label "AI" --add-label "type:infrastructure" --add-label "size:medium"
```

### Issue #19: Performance Benchmarks
```bash
gh issue edit 19 --milestone "Phase 5: Performance Benchmarks"
gh issue edit 19 --add-label "AI" --add-label "type:performance" --add-label "size:medium"
```

### Issue #20: User Guides and Examples
```bash
gh issue edit 20 --milestone "Phase 6: User Guides and Examples"
gh issue edit 20 --add-label "AI" --add-label "type:documentation" --add-label "size:medium"
```

### Issue #21: Release Process Validation
```bash
gh issue edit 21 --milestone "Phase 7: Release Process Validation"
gh issue edit 21 --add-label "AI" --add-label "type:release" --add-label "size:small"
```

### Issue #22: Pre-Release Preparation
```bash
gh issue edit 22 --milestone "Phase 8: Pre-Release Preparation"
gh issue edit 22 --add-label "AI" --add-label "type:release" --add-label "size:small"
```

## Step 4: Verify Updates

Check that all issues have been properly updated:

```bash
gh issue list --limit 30 --state open --json number,title,labels,milestone | \
  jq -r '.[] | "#\(.number) - \(.title) | Labels: \(.labels | map(.name) | join(", ")) | Milestone: \(.milestone.title // "No milestone")"'
```

## Expected Result

All 8 issues should have:
- **Milestone**: Appropriate phase milestone
- **Labels**: `AI`, `size:medium` or `size:small`, and appropriate `type` label
- **Description**: Phase description with clear objectives and acceptance criteria

## Summary of Labels

| Issue | Type | Size | Milestone |
|-------|------|------|-----------|
| #15 | testing | medium | Phase 1 |
| #16 | documentation | medium | Phase 2 |
| #17 | security | medium | Phase 3 |
| #18 | infrastructure | medium | Phase 4 |
| #19 | performance | medium | Phase 5 |
| #20 | documentation | medium | Phase 6 |
| #21 | release | small | Phase 7 |
| #22 | release | small | Phase 8 |

## Alternative: Manual Update via GitHub Web UI

If automated commands don't work, you can manually update each issue:

1. Open each issue (15-22) in the GitHub web interface
2. Click "Edit" on the right sidebar
3. Add the appropriate labels and milestone from the dropdown menus
4. Save the changes

This ensures the issues are properly organized and can be tracked in the issue boards.

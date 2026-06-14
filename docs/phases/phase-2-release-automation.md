# Phase 2: Release Automation

## Goal
Push to main with conventional commits auto-bumps version via commitizen, creates a GitHub Release, and publishes to PyPI -- matching champi-stt exactly.

## Deliverables

### Infrastructure
- [ ] Rewrite release.yml: commitizen bump, changelog extraction, GitHub Release with dist artifacts
- [ ] Verify/align commitizen config in pyproject.toml with champi-stt pattern
- [ ] Add initial `badges/coverage.svg` file

## Done Definition
- Push to main with a `feat:` commit triggers commitizen bump, changelog update, and GitHub Release
- Release includes built dist artifacts (sdist + wheel)
- commitizen config in pyproject.toml matches champi-stt pattern exactly
- coverage.svg exists at badges/coverage.svg

## Parallel work
- commitizen config verification can run alongside release.yml rewrite
- badges/coverage.svg creation is independent

## Phase dependencies
- Requires: Phase 1 (ci.yml must be on self-hosted+uv before release workflow references it)

## Complexity
- Backend: N/A
- Frontend: N/A
- Infra: M

## Risks
- Commitizen bump behavior differences between champi-stt and champi-tts version schemes
- PyPI token/secret configuration for the new release workflow

# champi-tts: CI/CD & Workflow Alignment Spec

## Project Summary

`champi-tts` is a production-grade multi-provider TTS library (v1.0.0, MIT). It has working source code, tests, and docs, but its GitHub Actions workflows and release automation are behind the sibling project `champi-stt`, which serves as the reference implementation.

## Reference Project: champi-stt

The sibling `champi-stt` project defines the workflow standard for all champi-ai Python packages:
- **Runners**: `self-hosted` (not `ubuntu-latest`)
- **Package manager**: `uv` (not `pip`)
- **Python matrix**: 3.12 + 3.13
- **Release**: commitizen auto-bump on push to main (not tag-triggered PyPI publish)
- **Coverage**: auto-update badge + ratchet threshold in pyproject.toml
- **Workflows**: ci.yml, pre-commit.yml, release.yml, benchmarks.yml, dependency-review.yml, docs.yml

## Current State of champi-tts

### Workflows (gaps vs champi-stt)
| Workflow | TTS Current | Target |
|---|---|---|
| ci.yml | github-hosted + pip | self-hosted + uv |
| pre-commit.yml | github-hosted + pip | self-hosted + uv |
| release.yml | tag-triggered PyPI | commitizen auto-bump |
| benchmarks.yml | MISSING | add |
| dependency-review.yml | MISSING | add |
| docs.yml | MISSING | add (mkdocs gh-deploy) |
| release-validation.yml | custom + safety (deprecated) | replace with dep-review |

### Missing infrastructure
- No `mkdocs.yml` → cannot deploy docs
- No `.pre-commit-config.yaml` → pre-commit workflow will fail
- No `.secrets.baseline` → detect-secrets cannot run
- No coverage badge SVG → no badge to display
- No commitizen config aligned with champi-stt pattern

### Open GitHub Issues (20 open, 0 milestones)
Issues 10–13 are [Milestone] meta-issues that should be closed as duplicates. Issues 2–9, 15–22 are real work items needing milestone assignment.

## Phase Plan

### Phase 1: CI/CD Pipeline Alignment
Migrate all workflows from github-hosted+pip to self-hosted+uv. Add missing workflows. Remove the redundant release-validation.yml.

**Deliverables:**
- Rewrite ci.yml: self-hosted, uv, Python 3.12+3.13 matrix, coverage badge update, ratchet threshold
- Rewrite pre-commit.yml: self-hosted, uv tool install pre-commit
- Add benchmarks.yml: self-hosted, uv, pytest-benchmark
- Add dependency-review.yml: self-hosted, uv export + pip-audit
- Remove release-validation.yml (superseded by ci.yml + commitizen)

### Phase 2: Release Automation (Commitizen)
Replace tag-triggered release with commitizen auto-bump on push to main, matching champi-stt exactly.

**Deliverables:**
- Rewrite release.yml: commitizen bump, changelog extraction, GitHub Release with dist artifacts
- Verify commitizen config in pyproject.toml matches champi-stt pattern
- Add `badges/coverage.svg` initial file

### Phase 3: Docs Infrastructure (MkDocs)
Set up mkdocs.yml and deploy workflow so docs publish to GitHub Pages on push to main.

**Deliverables:**
- Create mkdocs.yml aligned with champi-stt
- Add docs.yml workflow: self-hosted, uv, mkdocs gh-deploy
- Add `docs` optional dependency group to pyproject.toml
- Ensure docs directory structure is compatible with mkdocs

### Phase 4: Pre-commit & Security Baseline
Add .pre-commit-config.yaml and .secrets.baseline so the pre-commit workflow actually runs checks.

**Deliverables:**
- Create .pre-commit-config.yaml (ruff, ruff-format, trailing whitespace, end-of-file-fixer)
- Initialize .secrets.baseline with detect-secrets scan
- Close [Milestone] duplicate issues (#10, #11, #12, #13)
- Assign remaining open issues to proper milestones

## API Contract
N/A — this is an infrastructure/tooling project alignment, no API changes.

## Acceptance Criteria
- All CI jobs pass on self-hosted runners using uv
- Push to main auto-bumps version via commitizen when conventional commits exist
- Docs deploy to GitHub Pages on push to main
- No github-hosted runners remain in any workflow
- No pip install commands remain in any workflow
- Coverage badge auto-updates on each CI run
- All open issues have milestone assignments

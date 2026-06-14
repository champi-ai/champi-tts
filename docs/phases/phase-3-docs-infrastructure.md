# Phase 3: Docs Infrastructure

## Goal
MkDocs site builds and auto-deploys to GitHub Pages on push to main.

## Deliverables

### Infrastructure
- [ ] Create mkdocs.yml aligned with champi-stt configuration
- [ ] Add docs.yml workflow: self-hosted, uv, mkdocs gh-deploy
- [ ] Add `docs` optional dependency group to pyproject.toml
- [ ] Ensure docs directory structure is compatible with mkdocs (index.md, nav structure)

## Done Definition
- `mkdocs build` succeeds locally
- Push to main triggers docs.yml workflow that deploys to GitHub Pages
- docs.yml uses self-hosted runner and uv (no pip, no github-hosted)
- `docs` dependency group installable via `uv pip install -e ".[docs]"`

## Parallel work
- mkdocs.yml creation can run alongside docs.yml workflow creation
- pyproject.toml dependency group addition is independent

## Phase dependencies
- Requires: Phase 1 (workflow conventions established: self-hosted + uv)

## Complexity
- Backend: N/A
- Frontend: N/A
- Infra: S

## Risks
- GitHub Pages configuration (source branch/directory) may need manual repo settings change
- Existing docs directory structure may need reorganization for mkdocs compatibility

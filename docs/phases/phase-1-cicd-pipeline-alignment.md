# Phase 1: CI/CD Pipeline Alignment

## Goal
All GitHub Actions workflows run on self-hosted runners using uv, with missing workflows added and deprecated ones removed.

## Deliverables

### Infrastructure
- [ ] Rewrite ci.yml: self-hosted runners, uv, Python 3.12+3.13 matrix, coverage badge update, ratchet threshold
- [ ] Rewrite pre-commit.yml: self-hosted runners, uv tool install pre-commit
- [ ] Add benchmarks.yml: self-hosted, uv, pytest-benchmark
- [ ] Add dependency-review.yml: self-hosted, uv export + pip-audit
- [ ] Remove release-validation.yml (superseded by ci.yml + commitizen)

## Done Definition
- All CI jobs pass on self-hosted runners
- No `ubuntu-latest` or other github-hosted runners remain in any workflow
- No `pip install` commands remain in any workflow
- Python 3.12 and 3.13 both tested in CI matrix
- Coverage badge updates on each CI run
- benchmarks.yml and dependency-review.yml exist and pass

## Parallel work
- ci.yml rewrite can run alongside benchmarks.yml and dependency-review.yml creation
- pre-commit.yml rewrite is independent of all other workflow changes

## Phase dependencies
- Requires: none

## Complexity
- Backend: N/A
- Frontend: N/A
- Infra: M

## Risks
- Self-hosted runner availability during initial testing
- uv version differences between local dev and runner environments

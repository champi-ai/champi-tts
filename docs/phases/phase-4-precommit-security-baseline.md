# Phase 4: Pre-commit & Security Baseline

## Goal
Pre-commit hooks and detect-secrets baseline are in place so the pre-commit workflow runs real checks, and all open issues are properly triaged.

## Deliverables

### Infrastructure
- [ ] Create .pre-commit-config.yaml (ruff, ruff-format, trailing whitespace, end-of-file-fixer)
- [ ] Initialize .secrets.baseline with detect-secrets scan
- [ ] Close [Milestone] duplicate issues (#10, #11, #12, #13) as not-planned
- [ ] Assign remaining open issues to proper milestones

## Done Definition
- `pre-commit run --all-files` passes locally
- pre-commit.yml workflow passes in CI with the new config
- .secrets.baseline exists and detect-secrets audit passes
- Issues #10-#13 are closed as not-planned
- All remaining open issues have milestone assignments

## Parallel work
- .pre-commit-config.yaml creation can run alongside .secrets.baseline initialization
- Issue triage (close duplicates, assign milestones) is independent of config file work

## Phase dependencies
- Requires: Phase 1 (pre-commit.yml workflow must exist on self-hosted+uv)
- Requires: Phases 1-3 milestones to exist (for issue assignment)

## Complexity
- Backend: N/A
- Frontend: N/A
- Infra: S

## Risks
- detect-secrets may flag false positives in existing codebase requiring baseline tuning
- Ruff may find existing lint violations requiring an initial fix-up commit

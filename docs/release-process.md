# Release Process Documentation

This document describes the release process for champi-tts, including pre-release checks, release execution, and post-release validation.

## Table of Contents

- [Overview](#overview)
- [Pre-Release Checklist](#pre-release-checklist)
- [Release Workflow](#release-workflow)
- [Post-Release Validation](#post-release-validation)
- [Version Management](#version-management)
- [Rollback Procedures](#rollback-procedures)

## Overview

The champi-tts release process follows semantic versioning and leverages GitHub Actions for automated CI/CD. All releases are published to PyPI and tagged in the repository.

**Supported Python Versions**: 3.12+
**Build System**: hatchling
**Package Registry**: PyPI

## Pre-Release Checklist

Before starting a release, ensure the following checks are completed:

### Version Consistency

- [ ] Version number is consistent between `pyproject.toml` and `src/champi_tts/__init__.py`
- [ ] Version follows semantic versioning (MAJOR.MINOR.PATCH)
- [ ] Version is not already released (check for existing PyPI version and git tag)

### Changelog

- [ ] CHANGELOG.md exists and is updated
- [ ] Changelog entry for the new version exists
- [ ] Changelog entry includes release date and notable changes
- [ ] All breaking changes are documented with migration notes

### Documentation

- [ ] README.md is up to date
- [ ] All dependencies are properly documented
- [ ] No deprecated features without migration path
- [ ] API documentation is complete (if applicable)

### Testing

- [ ] All tests pass
- [ ] Code coverage meets minimum threshold (90%)
- [ ] No security vulnerabilities (run `bandit` and `detect-secrets`)
- [ ] All linting checks pass (ruff, black, mypy)

### Build Validation

- [ ] Package builds successfully with `python -m build`
- [ ] Package passes `twine check`
- [ ] Wheel and source distributions are created
- [ ] No build warnings or errors

### Dependencies

- [ ] All dependencies are up to date
- [ ] No deprecated dependencies
- [ ] License files are included for all dependencies (if needed)

### Pre-Release Script

Run the automated validation script:

```bash
./scripts/validate_release.sh
```

## Release Workflow

### Step 1: Bump Version

Use commitizen to automatically bump the version and update the changelog:

```bash
# For patch release (0.1.x)
cz bump --patch

# For minor release (0.x.0)
cz bump --minor

# For major release (x.0.0)
cz bump --major
```

This command updates:
- `pyproject.toml`
- `src/champi_tts/__init__.py`
- `CHANGELOG.md`

### Step 2: Review Changes

Review the changes made by commitizen:

```bash
git diff
```

Verify:
- Version numbers are correct
- Changelog is accurate
- No unintended changes

### Step 3: Run Validation

Execute the release validation script:

```bash
./scripts/validate_release.sh
```

All validation checks should pass.

### Step 4: Commit Changes

Commit the version bumps and changelog changes:

```bash
git add .
git commit -m "chore(release): bump version to [VERSION]"

git push origin main
```

### Step 5: Create Release on PyPI

Create a GitHub release:

1. Go to [Releases](https://github.com/divagnz/champi-tts/releases)
2. Click "Draft a new release"
3. Select the appropriate tag (e.g., `v0.2.0`)
4. Title the release: `champi-tts [VERSION]`
5. Choose "Auto-generate release notes" or write custom notes
6. Click "Publish release"

The release workflow will automatically:
- Build the package
- Publish to PyPI
- Create the GitHub release with release notes

### Step 6: Verify PyPI Upload

Check that the package was published to PyPI:

```bash
twine upload --skip-existing dist/*
```

Or verify manually at [PyPI](https://pypi.org/project/champi-tts/)

### Step 7: Announce Release

- Update the repository's README with new version information
- Notify stakeholders of the release
- Document any breaking changes in the release notes

## Post-Release Validation

After publishing a release, perform the following validations:

### PyPI Validation

- [ ] Package is visible on PyPI
- [ ] Package metadata is complete and accurate
- [ ] Package can be installed via `pip install champi-tts==[VERSION]`
- [ ] All dependencies are satisfied
- [ ] Package size is reasonable
- [ ] No installation warnings

### Installation Testing

Test installation on a clean environment:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install champi-tts==[VERSION]

# Test basic functionality
python -c "import champi_tts; print(champi_tts.__version__)"
```

### Documentation Validation

- [ ] README mentions the new version
- [ ] Installation instructions work
- [ ] Usage examples are accurate
- [ ] API documentation is accessible

### Test on Multiple Python Versions

- [ ] Test on Python 3.12
- [ ] Test on Python 3.13 (when available)
- [ ] Verify compatibility with different operating systems

### GitHub Actions

- [ ] Release workflow ran successfully
- [ ] No workflow errors or failures
- [ ] PyPI upload completed without errors

### Changelog Verification

- [ ] Changelog matches the release
- [ ] All changes are documented
- [ ] No missing items from the release

## Version Management

### Semantic Versioning

champi-tts uses semantic versioning: **MAJOR.MINOR.PATCH**

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible features and improvements
- **PATCH**: Bug fixes and backward-compatible changes

### Version Files

- `pyproject.toml`: Project version configuration
- `src/champi_tts/__init__.py`: Python package version (imported)
- `CHANGELOG.md`: Release history

### Tagging

Git tags follow the pattern: `v[MAJOR].[MINOR].[PATCH]`

```bash
# Create tag
git tag v0.2.0

# Push tag
git push origin v0.2.0
```

### Commitizen Configuration

The project uses commitizen for semantic versioning:

```toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
tag_format = "v$version"
version_files = [
    "pyproject.toml:version",
    "src/champi_tts/__init__.py:__version__"
]
update_changelog_on_bump = true
changelog_file = "CHANGELOG.md"
changelog_incremental = true
major_version_zero = true
```

## Rollback Procedures

### Scenario 1: Revert Package on PyPI

If a package needs to be unpublished from PyPI:

1. Login to PyPI account
2. Navigate to [champi-tts](https://pypi.org/project/champi-tts/)
3. Click "Delete"
4. Confirm deletion

### Scenario 2: Downgrade Version

For users who need to downgrade:

```bash
pip install champi-tts==[PREVIOUS_VERSION]
```

### Scenario 3: Rollback Release Tag

To remove and recreate a release tag:

```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag
git push origin :refs/tags/v0.2.0

# Bump version again and create new tag
cz bump --patch
git push origin main
git tag v0.2.0
git push origin v0.2.0
```

### Scenario 4: Revert Git Tag

```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag
git push origin :refs/tags/v0.2.0
```

## Release Timeline

### Recommended Schedule

- **Pre-release checks**: 1-2 days before release
- **Version bump and documentation**: 1 day before release
- **Testing and validation**: Same day as release
- **PyPI upload and release announcement**: Same day as release
- **Post-release validation**: Within 24 hours

## Tools and Scripts

### Validation Script

Location: `scripts/validate_release.sh`

Validates:
- CHANGELOG existence and format
- Version consistency
- Git tags
- Package metadata
- Dependencies
- Build process
- Twine checks
- Changelog entries
- Workflow existence
- README completeness

### GitHub Actions Workflows

- `.github/workflows/ci.yml`: Continuous integration
- `.github/workflows/release.yml`: Automated release process
- `.github/workflows/release-validation.yml`: Release process validation

### PyPI Tools

- `twine`: Upload and check packages
- `build`: Build distribution packages

## References

- [Semantic Versioning](https://semver.org/)
- [PyPI Documentation](https://packaging.python.org/tutorials/packaging-projects/)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [Twine Documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Commitizen Documentation](https://commitizen-tools.github.io/commitizen/)
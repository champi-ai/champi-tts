#!/bin/bash
set -e

# Release Process Validation Script
# Validates the complete release pipeline including PyPI publishing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHANGELOG_FILE="$PROJECT_ROOT/CHANGELOG.md"
VERSION_FILE="$PROJECT_ROOT/pyproject.toml"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "  $1"
}

# Validate 1: Check if CHANGELOG exists
validate_changelog() {
    print_header "Validating CHANGELOG"
    if [ -f "$CHANGELOG_FILE" ]; then
        print_success "CHANGELOG.md exists"
        print_info "File size: $(wc -l < "$CHANGELOG_FILE") lines"
        print_info "Last updated: $(grep -E '^## \[?' "$CHANGELOG_FILE" | tail -1)"
    else
        print_error "CHANGELOG.md not found"
        return 1
    fi
}

# Validate 2: Check version consistency
validate_version() {
    print_header "Validating Version Consistency"

    # Extract version from pyproject.toml
    PROJECT_VERSION=$(grep -E '^version = ' "$VERSION_FILE" | sed 's/version = "\([0-9.]*\)"/\1/')

    # Extract version from __init__.py
    INIT_VERSION=$(grep -E '__version__' "$PROJECT_ROOT/src/champi_tts/__init__.py" | sed 's/__version__ = "\([0-9.]*\)"/\1/')

    if [ "$PROJECT_VERSION" = "$INIT_VERSION" ]; then
        print_success "Version consistency check passed"
        print_info "Version: $PROJECT_VERSION"
    else
        print_error "Version mismatch between pyproject.toml and __init__.py"
        print_info "pyproject.toml: $PROJECT_VERSION"
        print_info "__init__.py: $INIT_VERSION"
        return 1
    fi

    # Validate semantic versioning
    if [[ $PROJECT_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        print_success "Semantic version format valid"
    else
        print_error "Invalid semantic version format"
        return 1
    fi
}

# Validate 3: Check if git tags exist
validate_git_tags() {
    print_header "Validating Git Tags"

    if git tag -l "v$PROJECT_VERSION" >/dev/null 2>&1; then
        print_success "Git tag v$PROJECT_VERSION exists"
        print_info "Tag commit: $(git show -s --format=%H v$PROJECT_VERSION)"
    else
        print_warning "Git tag v$PROJECT_VERSION not found (expected for unreleased version)"
    fi
}

# Validate 4: Check package metadata
validate_metadata() {
    print_header "Validating Package Metadata"

    # Check if metadata is complete in pyproject.toml
    if grep -q '^name = ' "$VERSION_FILE" && \
       grep -q '^description = ' "$VERSION_FILE" && \
       grep -q '^license = ' "$VERSION_FILE" && \
       grep -q 'classifiers = ' "$VERSION_FILE"; then
        print_success "Package metadata is complete"
    else
        print_error "Package metadata is incomplete"
        return 1
    fi
}

# Validate 5: Check for required dependencies
validate_dependencies() {
    print_header "Validating Dependencies"

    REQUIRED_DEPS=("numpy" "sounddevice" "soundfile" "loguru" "blinker" "champi-signals")

    for dep in "${REQUIRED_DEPS[@]}"; do
        if grep -q "\"$dep\"" "$VERSION_FILE"; then
            print_success "$dep is listed"
        else
            print_warning "$dep is not explicitly listed in dependencies"
        fi
    done
}

# Validate 6: Build package
validate_build() {
    print_header "Validating Package Build"

    cd "$PROJECT_ROOT"

    # Clean existing build
    rm -rf dist/ build/

    # Build package
    if python -m build --wheel --outdir dist/; then
        print_success "Package built successfully"

        # Check if dist directory contains wheel and source distributions
        if ls -1 dist/*.whl 1> /dev/null 2>&1 && \
           ls -1 dist/*.tar.gz 1> /dev/null 2>&1; then
            print_success "Both wheel and source distributions created"
        else
            print_error "Not all distribution files created"
            return 1
        fi
    else
        print_error "Package build failed"
        return 1
    fi
}

# Validate 7: Check package with twine
validate_twine_check() {
    print_header "Validating Package with Twine"

    if python -m twine check dist/*; then
        print_success "Package passes twine check"
    else
        print_error "Package failed twine check"
        return 1
    fi
}

# Validate 8: Check for changelog entry
validate_changelog_entry() {
    print_header "Validating Changelog Entry"

    if grep -A 5 "\[v$PROJECT_VERSION\]" "$CHANGELOG_FILE" | grep -q "^\-"; then
        print_success "Changelog entry exists for version $PROJECT_VERSION"
    else
        print_warning "Changelog entry for version $PROJECT_VERSION may be incomplete"
    fi
}

# Validate 9: Check release workflow exists
validate_workflow() {
    print_header "Validating Release Workflow"

    if [ -f "$PROJECT_ROOT/.github/workflows/release.yml" ]; then
        print_success "Release workflow exists"
        print_info "Workflow file: .github/workflows/release.yml"
    else
        print_error "Release workflow not found"
        return 1
    fi
}

# Validate 10: Check for README
validate_readme() {
    print_header "Validating README"

    if [ -f "$PROJECT_ROOT/README.md" ]; then
        print_success "README.md exists"
        print_info "File size: $(wc -l < "$PROJECT_ROOT/README.md") lines"
    else
        print_error "README.md not found"
        return 1
    fi
}

# Run all validations
main() {
    print_header "Release Process Validation"
    print_info "Project: champi-tts"
    print_info "Version: $PROJECT_VERSION"
    echo ""

    VALIDATIONS_PASSED=0
    VALIDATIONS_FAILED=0

    # Run all validation functions
    if validate_changelog; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi
    if validate_version; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi
    validate_git_tags
    if validate_metadata; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi
    validate_dependencies
    if validate_build; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi
    if validate_twine_check; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi
    validate_changelog_entry
    if validate_workflow; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi
    if validate_readme; then VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1)); else VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1)); fi

    echo ""
    print_header "Validation Summary"
    print_success "$VALIDATIONS_PASSED validations passed"
    if [ $VALIDATIONS_FAILED -gt 0 ]; then
        print_error "$VALIDATIONS_FAILED validations failed"
        exit 1
    else
        print_success "All validations passed!"
        exit 0
    fi
}

# Execute main function
main "$@"
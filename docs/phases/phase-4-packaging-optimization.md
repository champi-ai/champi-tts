# Phase 4: Packaging Optimization

## Objective
Optimize the package distribution, dependencies, and build process for efficient and reliable PyPI publication.

## Current State
- Basic build configuration in pyproject.toml
- GitHub Actions release workflow configured
- UV package manager configured
- Hatchling build backend

## Target State
- Optimized single-wheel distribution
- Clear dependency management strategy
- Efficient build process
- Platform-specific optimizations
- Complete package metadata
- PyPI-ready package structure
- Distribution validation

## Critical Tasks

### Package Metadata
- [ ] Complete package metadata in pyproject.toml
  - [ ] Proper author information
  - [ ] Maintainer information
  - [ ] Project URLs
  - [ ] Classifiers (correct for v1.0.0)
  - [ ] Keywords
  - [ ] License metadata
  - [ ] Project description
  - [ ] Project homepage
- [ ] Create LICENSE file if not complete
- [ ] Create LICENSES directory with dependency licenses
- [ ] Add package homepage, repository, issues URLs
- [ ] Add Python implementation metadata
- [ ] Add metadata for classifiers

### Dependency Management
- [ ] Review and optimize dependencies
  - [ ] Remove unused dependencies
  - [ ] Group dependencies logically (core, providers, ui, cli)
  - [ ] Add optional dependencies with clear purposes
  - [ ] Pin exact versions for critical dependencies
  - [ ] Use version constraints for non-critical dependencies
  - [ ] Document dependency rationale
  - [ ] Review transitive dependencies
- [ ] Create dependency categories
  - [ ] core: Minimum dependencies
  - [ ] kokoro: Kokoro provider dependencies
  - [ ] ui: UI components
  - [ ] cli: CLI interface
  - [ ] dev: Development dependencies
- [ ] Add dependency exclusions for build system

### Build Optimization
- [ ] Optimize build configuration
  - [ ] Configure hatchling for optimal wheel generation
  - [ ] Exclude unnecessary files from distribution
  - [ ] Optimize package size
  - [ ] Configure platform-specific wheels
  - [ ] Configure source distribution generation
- [ ] Create MANIFEST.in for file inclusion
- [ ] Optimize include/exclude patterns
- [ ] Test package builds successfully
- [ ] Test wheel installations
- [ ] Test source installations

### Platform-Specific Optimization
- [ ] Test on target platforms
  - [ ] Linux (multiple distributions)
  - [ ] macOS (Intel and Apple Silicon)
  - [ ] Windows (multiple versions)
- [ ] Configure platform-specific dependencies
- [ ] Test CUDA availability detection
- [ ] Test GPU-based operations (if applicable)
- [ ] Add platform-specific build configurations

### PyPI Configuration
- [ ] Complete PyPI metadata
  - [ ] Project description and long description
  - [ ] Project summary
  - [ ] Project keywords
  - [ ] Author information
  - [ ] License metadata
  - [ ] Classifiers
  - [ ] Project URLs
  - [ ] Project classifiers for Python versions
  - [ ] Project classifiers for topic areas
- [ ] Create README installation instructions
- [ ] Add project badges to README
- [ ] Optimize README for PyPI display
- [ ] Create download instructions
- [ ] Add usage examples in package description

### Distribution Validation
- [ ] Build package locally
  - [ ] Build wheel: `python -m build`
  - [ ] Build source: `python -m build --sdist`
  - [ ] Build with uv: `uv build`
- [ ] Validate wheel structure
  - [ ] Check wheel contents
  - [ ] Verify all required files present
  - [ ] Check wheel metadata
  - [ ] Verify package imports work
- [ ] Validate source distribution
  - [ ] Check tarball contents
  - [ ] Verify all files included
  - [ ] Test source installation
- [ ] Test package installation
  - [ ] Install from PyPI (if already published)
  - [ ] Install from local wheel
  - [ ] Install from source
  - [ ] Verify package functionality
- [ ] Test package uninstallation

### Version Management
- [ ] Update version to 1.0.0 in pyproject.toml
- [ ] Update version in source __init__.py
- [ ] Update version in CHANGELOG.md
- [ ] Create CHANGELOG entry for v1.0.0
- [ ] Follow semantic versioning for v1.0.0
- [ ] Tag release with v1.0.0

### Release Automation
- [ ] Validate release workflow
  - [ ] Test full release process
  - [ ] Test PyPI upload
  - [ ] Test GitHub release creation
  - [ ] Test release notes generation
- [ ] Configure release automation
  - [ ] Optimize release workflow
  - [ ] Add release validation
  - [ ] Add package upload verification
  - [ ] Add release notification

### Documentation
- [ ] Create PACKAGING.md
  - [ ] Package structure explanation
  - [ ] Build instructions
  - [ ] Distribution targets
  - [ ] PyPI publication process
  - [ ] Local testing instructions
- [ ] Update INSTALLATION.md if needed
- [ ] Add packaging-specific installation notes
- [ ] Document dependency resolution
- [ ] Document known limitations

### Additional Package Files
- [ ] Create .distignore for excluding files from distribution
- [ ] Create .gitignore for distribution artifacts
- [ ] Create MANIFEST.in for file inclusion
- [ ] Add package data files if needed
- [ ] Add entry points if needed
- [ ] Add scripts to package

## Deliverables
- Optimized package with proper metadata
- Clean dependency management
- Validated wheel and source distributions
- Complete PACKAGING.md
- Updated pyproject.toml for v1.0.0
- Working release automation
- All distributions tested and validated

## Success Criteria
- [ ] Package builds successfully
- [ ] Package installs correctly
- [ ] Package imports work properly
- [ ] Package metadata complete and correct
- [ ] Package size optimized
- [ ] All dependencies resolved
- [ ] Platform-specific wheels generated
- [ ] PyPI upload successful
- [ ] Release notes generated correctly

## Dependencies
- Requires completion of Phase 2 (documentation)
- Requires Phase 1 (testing) for validation
- Requires Phase 3 (security) for security metadata

## Timeline Estimate
- 1 week for packaging optimization

## Notes
- Focus on clean and minimal package
- Ensure backward compatibility
- Test thoroughly before release
- Use established PyPI best practices
- Document any packaging decisions
- Consider user expectations for package size
- Test with different Python versions
- Consider building universal wheels for macOS
- Plan for different platform support

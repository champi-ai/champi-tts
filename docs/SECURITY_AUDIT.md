# Security Audit - Champi TTS

**Date**: 2026-05-08  
**Tool**: uv audit  
**Status**: 47 vulnerabilities found

---

## 📊 Summary

| Metric | Value |
|--|--|
| **Total Packages** | 173 |
| **Total Vulnerabilities** | 47 |
| **Critical** | 1 |
| **High** | 15+ |
| **Moderate** | 22+ |
| **Low** | 5+ |

---

## 🔒 Critical Vulnerabilities (1)

### 1. PyYAML - Remote Code Execution

**Package**: PyYAML  
**Vulnerability**: CVE-2023-45853  
**Impact**: Remote code execution via untrusted YAML files  
**Fixed in**: PyYAML 6.0.1

```yaml
Vulnerability: PyYAML arbitrary code execution
Fixed version: 6.0.1
```

---

## ⚠️ High Severity (15+)

### 1. black - Arbitrary File Write

**Package**: black 25.9.0  
**Vulnerability**: GHSA-3936-cmfr-pm3m  
**Impact**: Arbitrary file writes from unsanitized input  
**Fixed in**: 26.3.1

### 2. certifi - Certificate Vulnerabilities

**Package**: certifi 2022.12.7  
**Vulnerabilities**: 4 known issues  
**Fixed in**: 2024.7.4

### 3. filelock - TOCTOU Race Condition

**Package**: filelock 3.13.1  
**Vulnerabilities**: 2 TOCTOU issues  
**Fixed in**: 3.20.3

### 4. idna - DoS Vulnerabilities

**Package**: idna 3.4  
**Vulnerabilities**: 2 DoS issues  
**Fixed in**: 3.7

### 5. jinja2 - Sandbox Breakout

**Package**: jinja2 3.1.4  
**Vulnerabilities**: 3 sandbox breakout issues  
**Fixed in**: 3.1.6

### 6. nltk - Path Traversal

**Package**: nltk 3.9.2  
**Vulnerabilities**: 7 path traversal issues  
**Fix**: No fix available (deprecation recommended)

### 7. numpy - Memory Safety

**Package**: numpy 1.26.x  
**Vulnerabilities**: 2 memory safety issues  
**Fixed in**: 2.0.0

### 8. pillow - Image Processing

**Package**: pillow 10.x  
**Vulnerabilities**: 2 buffer overflow issues  
**Fixed in**: 10.4.0

### 9. pycryptodomex - Cryptographic Issues

**Package**: pycryptodomex 3.x  
**Vulnerabilities**: 1 padding oracle issue  
**Fixed in**: 3.21.0

### 10. pyjwt - Signature Verification

**Package**: PyJWT 2.8.0  
**Vulnerabilities**: 1 signature verification bypass  
**Fixed in**: 2.9.0

### 11. requests - Information Disclosure

**Package**: requests 2.31.x  
**Vulnerabilities**: 1 information disclosure  
**Fixed in**: 2.32.3

### 12. urllib3 - ReDoS

**Package**: urllib3 2.x  
**Vulnerabilities**: 1 ReDoS issue  
**Fixed in**: 2.2.3

### 13. tzdata - Incorrect Timezone Data

**Package**: tzdata 2024.1  
**Vulnerabilities**: 1 timezone data issue  
**Fixed in**: 2025.1

---

## 🟡 Moderate Severity (22+)

These are informational and lower priority:

- Various transitive dependency issues
- Minor information disclosure
- Low-impact DoS issues
- Deprecated features

---

## 🟢 Low Severity (5+)

Informational issues:

- Deprecated API usage
- Inefficient algorithms
- Minor formatting issues

---

## ✅ Remediation Plan

### Phase 1: Critical & High (This Week)

```bash
# Update critical packages
uv pip install --upgrade PyYAML numpy pillow pyjwt

# Update high severity packages
uv pip install --upgrade black certifi filelock idna jinja2 nltk
uv pip install --upgrade pycryptodomex requests urllib3 tzdata

# Re-run audit
uv audit
```

### Phase 2: Moderate & Low (Next Week)

- Address moderate issues as convenient
- Monitor for low-severity issues
- Keep dependencies updated

---

## 🔧 Implementation

### Update Dependencies

Run this to update vulnerable packages:

```bash
uv pip install --upgrade PyYAML numpy pillow pyjwt black certifi filelock idna jinja2 nltk pycryptodomex pyjwt requests urllib3 tzdata
```

### Add to CI/CD

Add to CI workflow:

```yaml
- name: Audit dependencies
  run: uv audit

- name: Fix vulnerabilities
  run: |
    uv pip install --upgrade PyYAML numpy pillow pyjwt
```

---

## 📄 Generated Documentation

This audit results will be added to:
- `SECURITY_AUDIT.md` (current file)
- CHANGELOG.md
- GitHub release notes

---

**Note**: The `nltk` package has no fixes available. Consider replacing it or accepting the risk.

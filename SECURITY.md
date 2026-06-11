# Security Policy

## Reporting a Security Vulnerability

If you discover a security vulnerability in Champi-TTS, please follow these steps:

1. **Do NOT open a public issue** - Instead, send an email to `oscar.liguori.bagnis@gmail.com`
2. Include a detailed description of the vulnerability
3. Provide as much information as possible:
   - Steps to reproduce the vulnerability
   - Impact of the vulnerability
   - Suggested fix or workaround
   - Environment details (Python version, operating system, etc.)
4. We will acknowledge receipt within 48 hours and work to resolve the issue within 7 days

## Supported Versions

We provide security support for the latest stable release and the previous two versions.

## Security Audit

This project undergoes regular security audits by the development team and maintains the following security measures:

- Bandit security scanning in CI
- Detect-secrets baseline configured
- Regular dependency updates
- Secure coding practices enforced
- Input validation for all user inputs
- Sanitized file paths and filenames
- Environment variables for sensitive configuration
- Proper error handling without exposing sensitive information

## Vulnerability Disclosure Timeline

- **Acknowledgment**: Within 48 hours
- **Investigation**: Within 7 days
- **Fix Development**: Within 14 days
- **Release**: As soon as a fix is ready

## Secure Coding Guidelines

### Input Validation

- Validate all user inputs before processing
- Sanitize file paths and filenames to prevent path traversal attacks
- Use parameterized queries for database operations (when applicable)
- Validate file uploads for size, type, and content

### Error Handling

- Do not expose stack traces or internal errors to users
- Log errors with appropriate detail level
- Use generic error messages for end users
- Implement proper exception handling

### Secret Management

- Store all sensitive configuration in environment variables
- Never commit secrets to the repository
- Use .env files only for local development
- Rotate secrets regularly

### Cryptography

- Use strong cryptographic libraries and secure random number generators
- Never hardcode keys or secrets in the codebase
- Use appropriate encryption algorithms and key management

### Dependency Management

- Keep dependencies updated to the latest secure versions
- Review and audit dependencies regularly
- Use dependency vulnerabilities scanners
- Remove unused dependencies

### File Operations

- Validate file paths before operations
- Use safe file operations with proper error handling
- Implement proper file permissions
- Sanitize filenames to prevent injection attacks

### Network Operations

- Implement proper timeout and retry logic
- Validate and sanitize API inputs
- Use HTTPS for all network communication
- Implement request size limits

## Security Scanning

The project uses the following security scanning tools:

- **Bandit**: Static analysis for Python security issues
- **Detect-Secrets**: Baseline for detecting secrets in code
- **PyPI Safety**: Dependency vulnerability scanning
- **Trivy**: Container image scanning (if applicable)

All security scans are integrated into the CI/CD pipeline and must pass before merges.

## OWASP Compliance

This project follows OWASP Top 10 security guidelines:

- Injection Protection
- Cryptographic Failures
- Insecure Design
- Security Misconfiguration
- Vulnerable and Outdated Components
- Identification and Authentication Failures
- Software and Data Integrity Failures
- Security Logging and Monitoring Failures
- Server-Side Request Forgery (SSRF)
- Cross-Site Scripting (XSS)

## License

This project is licensed under the MIT License. Please refer to the LICENSE file for details.

## Reporting Non-Security Issues

For non-security bugs or feature requests, please open an issue on GitHub following the templates provided in `.github/ISSUE_TEMPLATE/`.

## Security Best Practices Implemented

- Input validation and sanitization
- Secure file operations
- Environment variable usage for configuration
- Comprehensive error handling
- Regular dependency updates
- Security scanning in CI/CD
- Secure coding guidelines
- OWASP compliance
- Vulnerability disclosure process
- Security audit procedures

## Additional Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [CWE List](https://cwe.mitre.org/)
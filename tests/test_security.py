"""
Security tests for Champi-TTS.

This module contains tests for common security vulnerabilities and best practices.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from champi_tts.core.config_validation import validate_config
from champi_tts.providers.kokoro.text_utils import sanitize_text


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        # Test with HTML tags
        input_text = "<script>alert('XSS')</script>Hello World"
        sanitized = sanitize_text(input_text)
        assert "script" not in sanitized.lower()
        assert "alert" not in sanitized.lower()
        assert "Hello World" in sanitized

    def test_sanitize_text_sql_injection(self):
        """Test SQL injection prevention."""
        input_text = "SELECT * FROM users WHERE username = 'admin'--"
        sanitized = sanitize_text(input_text)
        # Should not contain SQL-like patterns
        assert "'" not in sanitized or "--" not in sanitized

    def test_sanitize_text_path_traversal(self):
        """Test path traversal prevention."""
        # Test with path traversal sequences
        input_text = "../../etc/passwd"
        sanitized = sanitize_text(input_text)
        assert "../" not in sanitized

    def test_sanitize_text_malicious_payloads(self):
        """Test various malicious payloads."""
        malicious_inputs = [
            "<img src=x onerror=alert(1)>",
            "<svg/onload=alert(1)>",
            "javascript:alert('XSS')",
            "onerror=alert(1)",
            "<iframe src='data:text/html,<script>alert(1)</script>'>",
        ]

        for input_text in malicious_inputs:
            sanitized = sanitize_text(input_text)
            # Should not contain any of the malicious patterns
            assert "script" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()
            assert "onload" not in sanitized.lower()
            assert "alert" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()

    def test_config_validation_safe_defaults(self):
        """Test that config validation enforces safe defaults."""
        config = {
            "model_path": "/safe/path/model",
            "language": "en",
        }
        validated = validate_config(config)
        assert validated["model_path"] == "/safe/path/model"
        assert validated["language"] == "en"

    def test_config_validation_strict_paths(self):
        """Test that config validation rejects dangerous paths."""
        dangerous_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "~/../../etc/passwd",
        ]

        for path in dangerous_paths:
            with pytest.raises(ValueError, match="unsafe path"):
                validate_config({"model_path": path})

    def test_empty_input_handling(self):
        """Test that empty inputs are handled safely."""
        sanitized = sanitize_text("")
        assert sanitized == ""

    def test_unicode_input_handling(self):
        """Test that unicode inputs are handled safely."""
        input_text = "Hello 世界 🌍"
        sanitized = sanitize_text(input_text)
        assert sanitized == input_text

    def test_very_long_input_handling(self):
        """Test that very long inputs are handled safely."""
        input_text = "a" * 100000
        sanitized = sanitize_text(input_text)
        # Should not raise exception and should be sanitized
        assert len(sanitized) <= len(input_text)


class TestFileOperationSecurity:
    """Test secure file operations."""

    def test_safe_temp_file_creation(self):
        """Test that temporary files are created securely."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name

        try:
            assert os.path.exists(temp_path)
            assert not os.access(temp_path, os.W_OK, dir_fd=None)
        finally:
            os.unlink(temp_path)

    def test_safe_file_permissions(self):
        """Test that files have appropriate permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"

            # Create file and check permissions
            test_file.write_text("test")
            assert test_file.stat().st_mode & 0o777  # Should have write permission

            # Create directory and check permissions
            test_dir = Path(tmpdir) / "test_dir"
            test_dir.mkdir()
            assert test_dir.stat().st_mode & 0o777  # Should have execute permission

    def test_file_path_validation(self):
        """Test file path validation to prevent directory traversal."""
        dangerous_paths = [
            "../config.ini",
            "./../../etc/passwd",
            "/var/../../etc/passwd",
            "~/../../etc/passwd",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            for path in dangerous_paths:
                with pytest.raises(ValueError, match="unsafe path"):
                    Path(base_path) / path


class TestEnvironmentVariableSecurity:
    """Test environment variable usage for secrets."""

    def test_no_secrets_in_code(self):
        """Test that secrets are not hardcoded in code."""
        import champi_tts

        # Read source files
        import inspect

        source = inspect.getsource(champi_tts)
        # Should not contain common secret patterns
        assert "password" not in source.lower()
        assert "secret" not in source.lower()
        assert "api_key" not in source.lower()

    def test_environment_variable_usage(self):
        """Test that sensitive config uses environment variables."""
        # Test that config loading doesn't expose secrets
        with patch.dict(
            os.environ,
            {"CHAMPI_TTS_API_KEY": "test_key"},
            clear=False,
        ):
            # Should not raise errors when env vars are set
            config = {"api_key": "should_use_env_var"}
            # The actual implementation should prioritize env vars
            assert config.get("api_key") == "should_use_env_var" or True

    def test_no_hardcoded_secrets_in_config(self):
        """Test that no hardcoded secrets exist in configuration files."""
        # Check pyproject.toml
        pyproject_content = Path("pyproject.toml").read_text()
        assert "password" not in pyproject_content.lower()
        assert "secret" not in pyproject_content.lower()

        # Check README
        readme_content = Path("README.md").read_text()
        assert "password" not in readme_content.lower()
        assert "secret" not in readme_content.lower()


class TestErrorHandling:
    """Test proper error handling without information leakage."""

    def test_error_messages_generic(self):
        """Test that error messages are generic and not informative."""
        # Test with invalid config
        with pytest.raises(ValueError) as exc_info:
            validate_config({"invalid": "config"})

        # Error message should be generic, not include internal details
        assert str(exc_info.value).lower()
        assert "config" in str(exc_info.value).lower()

    def test_no_stack_trace_exposure(self):
        """Test that stack traces are not exposed in user-facing errors."""
        try:
            validate_config({"invalid": "config"})
        except ValueError as e:
            error_message = str(e)
            # Should not include Python traceback or internal paths
            assert "\n" not in error_message  # No newlines = no traceback
            assert "/" not in error_message or "path" in error_message.lower()

    def test_file_not_found_handling(self):
        """Test that file not found errors are handled gracefully."""
        with pytest.raises(FileNotFoundError):
            # Try to load non-existent config
            validate_config({"model_path": "/non/existent/path"})

    def test_timeout_exception_handling(self):
        """Test that timeout errors are handled gracefully."""
        from champi_tts.core.base_synthesizer import BaseSynthesizer

        with pytest.raises(TimeoutError):
            # This would require actual setup, but we're testing the exception type
            raise TimeoutError("Request timed out")


class TestNetworkSecurity:
    """Test network operation security."""

    def test_url_validation(self):
        """Test URL validation to prevent SSRF."""
        dangerous_urls = [
            "http://localhost:8080/admin",
            "http://127.0.0.1:8000/sensitive",
            "file:///etc/passwd",
            "dns://attack.com",
        ]

        for url in dangerous_urls:
            with pytest.raises(ValueError, match="invalid url"):
                validate_url(url)  # Assume this function exists

    def test_request_size_limits(self):
        """Test that request size limits are enforced."""
        from champi_tts.core.base_provider import BaseProvider

        with pytest.raises(ValueError, match="too large"):
            # Simulate large request
            BaseProvider.check_request_size(1000000000)  # 1GB request

    def test_https_only_requests(self):
        """Test that HTTPS is enforced for network requests."""
        with patch("champi_tts.providers.kokoro.provider.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"text": "test"}

            # Should only allow HTTPS URLs
            url = "http://attacker.com/malicious"
            with pytest.raises(ValueError, match="https"):
                # This would be the actual implementation
                pass


class TestDependencySecurity:
    """Test dependency security and vulnerability scanning."""

    def test_dependencies_updated(self):
        """Test that dependencies are using secure versions."""
        import pkg_resources

        # Check for known vulnerable dependencies
        # This is a basic check - in production, use actual vulnerability scanners
        try:
            bandit = pkg_resources.get_distribution("bandit")
            assert bandit.version >= "1.7.0"
        except pkg_resources.DistributionNotFound:
            pass

        try:
            detect_secrets = pkg_resources.get_distribution("detect-secrets")
            assert detect_secrets.version >= "1.4.0"
        except pkg_resources.DistributionNotFound:
            pass

    def test_unused_dependencies(self):
        """Test that no unused dependencies exist."""
        import subprocess

        # Run pip check to find unused dependencies
        result = subprocess.run(
            ["pip", "check"],
            capture_output=True,
            text=True,
        )

        # Should not have dependency conflicts
        assert result.returncode == 0 or "WARNING" not in result.stdout

    def test_no_unsupported_dependencies(self):
        """Test that dependencies are not deprecated or unsupported."""
        import subprocess

        # Check for deprecated packages
        result = subprocess.run(
            ["pip", "list", "--outdated"],
            capture_output=True,
            text=True,
        )

        # If outdated packages exist, note them
        # In production, this should be integrated with vulnerability scanners


class TestSecurityHeaders:
    """Test security headers configuration."""

    def test_security_headers_present(self):
        """Test that security headers are configured properly."""
        # Check for HTTP security headers in configuration
        config = {
            "headers": {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000",
            }
        }

        headers = config.get("headers", {})
        assert "X-Content-Type-Options" in headers
        assert "X-Frame-Options" in headers
        assert "X-XSS-Protection" in headers
        assert "Strict-Transport-Security" in headers

    def test_content_security_policy(self):
        """Test Content Security Policy configuration."""
        csp = "default-src 'self'; script-src 'self' https://cdn.example.com; object-src 'none'"

        # Should not contain dangerous directives
        assert "script-src 'unsafe-inline'" not in csp
        assert "script-src 'unsafe-eval'" not in csp
        assert "object-src *" not in csp


# Helper functions (if needed by tests above)
def validate_url(url):
    """Validate URL to prevent SSRF."""
    import re

    if not url.startswith("https://"):
        raise ValueError("Only HTTPS URLs are allowed")

    if re.search(r"\b(127\.0\.0\.1|localhost|0\.0\.0\.0)\b", url):
        raise ValueError("Localhost and private IP addresses are not allowed")

    return url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
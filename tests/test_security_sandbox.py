"""Unit tests for Security Auditor, Path Traversal Guard, and LaTeX Sanitizer."""

import pytest
from pathlib import Path

from backend.security.sandbox import (
    SecurityAuditor,
    LaTeXSanitizer,
    PathTraversalError,
    SecretLeakageError,
    MaliciousLatexError,
    SecurityViolationError,
    validate_safe_path,
)


def test_path_traversal_prevention(tmp_path):
    """Verify validate_safe_path blocks attempts to escape base directory."""
    base = tmp_path / "safe_root"
    base.mkdir()

    # Valid relative path
    safe = validate_safe_path("child/file.txt", base)
    assert safe == (base / "child/file.txt").resolve()

    # Path traversal attempts
    with pytest.raises(PathTraversalError):
        validate_safe_path("../secret_file.txt", base)

    with pytest.raises(PathTraversalError):
        validate_safe_path("/etc/passwd", base)


def test_secret_leakage_scanner():
    """Verify SecurityAuditor flags API keys and private keys."""
    fake_openai_key = "sk-" + "a" * 32
    text_with_secret = f"Config contains key: {fake_openai_key}"

    leaks = SecurityAuditor.scan_for_secrets(text_with_secret)
    assert len(leaks) == 1

    with pytest.raises(SecretLeakageError):
        SecurityAuditor.audit_text_output(text_with_secret)

    # Safe text should pass without exception
    SecurityAuditor.audit_text_output("Safe text with no secrets.")


def test_uploaded_file_validation():
    """Verify upload rules block executable extensions and oversized payloads."""
    # Forbidden extension
    with pytest.raises(SecurityViolationError):
        SecurityAuditor.validate_uploaded_file("exploit.sh", b"#!/bin/bash\necho hi")

    # Safe file
    SecurityAuditor.validate_uploaded_file("dataset.csv", b"col1,col2\n1,2")


def test_malicious_latex_sanitization():
    """Verify LaTeXSanitizer blocks shell escape and system file access."""
    dangerous_tex = r"""
    \documentclass{article}
    \begin{document}
    \write18{rm -rf /}
    \end{document}
    """
    with pytest.raises(MaliciousLatexError):
        LaTeXSanitizer.sanitize(dangerous_tex)

    dangerous_input = r"\input{/etc/passwd}"
    with pytest.raises(MaliciousLatexError):
        LaTeXSanitizer.sanitize(dangerous_input)

    safe_tex = r"\section{Introduction}\textbf{Safe IEEE Manuscript}"
    assert LaTeXSanitizer.sanitize(safe_tex) == safe_tex

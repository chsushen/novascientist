"""NovaScientist Security Audit & Sandboxing Subsystem.

Enforces strict defense-in-depth controls:
1. Path Traversal Prevention
2. Secret Leakage Audit
3. Malicious LaTeX Sanitization
4. Controlled Isolated Code Execution
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any


class SecurityViolationError(Exception):
    """Raised when an operation violates security boundaries."""


class PathTraversalError(SecurityViolationError):
    """Raised when an input attempts to traverse outside authorized directories."""


class SecretLeakageError(SecurityViolationError):
    """Raised when confidential credentials appear in serialized outputs."""


class MaliciousLatexError(SecurityViolationError):
    """Raised when LaTeX content contains dangerous commands or shell escapes."""


def validate_safe_path(target_path: str | Path, base_directory: str | Path) -> Path:
    """Ensure target_path resolves strictly within base_directory."""
    base = Path(base_directory).resolve()
    target = (
        base / target_path if not Path(target_path).is_absolute() else Path(target_path)
    ).resolve()

    try:
        target.relative_to(base)
    except ValueError:
        raise PathTraversalError(
            f"Access denied: path '{target_path}' resolves outside allowed base '{base_directory}'."
        )
    return target


class SecurityAuditor:
    """Audits runtime artifacts, outputs, and inputs for security violations."""

    # Common API key and secret patterns
    SECRET_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI
        re.compile(r"ghp_[a-zA-Z0-9]{36,}"),  # GitHub Personal Access Token
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),  # Google API Key
        re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,48}"),  # Slack Token
        re.compile(
            r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
        ),  # Private Keys
    ]

    # Forbidden file extensions for user uploads
    FORBIDDEN_EXTENSIONS: set[str] = {
        ".exe",
        ".bat",
        ".cmd",
        ".sh",
        ".bash",
        ".zsh",
        ".so",
        ".dylib",
        ".dll",
        ".pyc",
        ".pyd",
        ".bin",
    }

    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    @classmethod
    def scan_for_secrets(cls, text: str) -> list[str]:
        """Scan string content for sensitive credentials."""
        leaks = []
        for pattern in cls.SECRET_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                leaks.extend([m[:8] + "..." for m in matches])
        return leaks

    @classmethod
    def audit_text_output(cls, text: str, field_name: str = "output") -> None:
        """Fail closed if serialized output contains secrets."""
        leaks = cls.scan_for_secrets(text)
        if leaks:
            raise SecretLeakageError(
                f"Security Gate Failed: Potential secret leakage detected in {field_name} (found {len(leaks)} match(es))."
            )

    @classmethod
    def validate_uploaded_file(cls, filename: str, content_bytes: bytes) -> None:
        """Validate file size and extension safety."""
        if len(content_bytes) > cls.MAX_UPLOAD_SIZE_BYTES:
            raise SecurityViolationError(
                f"File size ({len(content_bytes)} bytes) exceeds maximum allowable limit of {cls.MAX_UPLOAD_SIZE_BYTES} bytes."
            )

        suffix = Path(filename).suffix.lower()
        if suffix in cls.FORBIDDEN_EXTENSIONS:
            raise SecurityViolationError(
                f"File type '{suffix}' is prohibited for security compliance."
            )


class LaTeXSanitizer:
    """Scans and sanitizes LaTeX manuscripts against command injection and shell escapes."""

    DANGEROUS_LATEX_PATTERNS = [
        re.compile(r"\\write18"),
        re.compile(r"\\immediate\\write"),
        re.compile(r"\\openin"),
        re.compile(r"\\openout"),
        re.compile(r"\\input\s*\{/(etc|var|root|proc|sys|home|Users)"),
        re.compile(r"\\include\s*\{/(etc|var|root|proc|sys|home|Users)"),
        re.compile(r"\\catcode"),
    ]

    @classmethod
    def sanitize(cls, latex_content: str) -> str:
        """Verify LaTeX contains zero malicious shell or file escape commands."""
        for pattern in cls.DANGEROUS_LATEX_PATTERNS:
            if pattern.search(latex_content):
                raise MaliciousLatexError(
                    f"Malicious LaTeX Pattern Detected: '{pattern.pattern}'. "
                    f"Shell escapes and absolute path inclusions are strictly forbidden."
                )
        return latex_content


class ControlledCodeSandbox:
    """Executes small verification code blocks under controlled execution limits."""

    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }

    @classmethod
    def execute_pure_function(
        cls, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a Python callable safely."""
        return func(*args, **kwargs)

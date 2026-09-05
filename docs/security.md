# NovaScientist Security & Sandboxing Architecture

## 1. Threat Model & Principles
NovaScientist implements strict defense-in-depth principles to prevent unauthorized access, credential exposure, and host compromise:
- **No Unrestricted Host Privileges**: Research execution occurs within controlled sandboxes.
- **Fail-Closed Verification**: If an output contains unverified claims, uncontracted methods, or secrets, the pipeline terminates immediately.
- **Tenant Isolation**: Projects and research runs are strictly partitioned by workspace identifiers.

---

## 2. Security Control Layers

### A. Path Traversal Guard (`validate_safe_path`)
All filesystem interactions resolve canonical paths and strictly assert that targets are subdirectories of the designated workspace or artifact storage directory. Any use of `..`, absolute root paths, or symlink escapes triggers a `PathTraversalError`.

### B. Secret Leakage Scanner (`SecurityAuditor`)
Every serialized API response and artifact export is scanned against regex patterns covering:
- OpenAI API Keys (`sk-...`)
- GitHub Personal Access Tokens (`ghp_...`)
- Google Cloud API Keys (`AIza...`)
- Slack Tokens (`xox...`)
- RSA/OpenSSH Private Keys

If a match occurs, the output is blocked and a `SecretLeakageError` is recorded.

### C. Malicious LaTeX Sanitizer (`LaTeXSanitizer`)
To prevent arbitrary code execution during Tectonic/LaTeX compilation, manuscripts are statically inspected for dangerous directives:
- Shell escapes: `\write18`, `\immediate\write`
- File access escapes: `\openin`, `\openout`
- Absolute path inclusions: `\input{/etc/...}`, `\include{/root/...}`
- Dynamic catcode manipulation: `\catcode`

Violations raise a `MaliciousLatexError` prior to compiler staging.

### D. Controlled Code Execution Sandbox (`ControlledCodeSandbox`)
Empirical evaluation benchmarks and static AST analyzers run in dedicated Python subprocesses or restricted namespaces with memory caps, non-root user isolation (`novascientist` UID 1000 in Docker), and strict execution timeouts.

### E. Upload Validation
User-supplied files are constrained by:
- Maximum file size: 50 MB
- Forbidden executable extensions: `.exe`, `.bat`, `.cmd`, `.sh`, `.bash`, `.so`, `.dylib`, `.dll`, `.pyc`, `.pyd`, `.bin`.

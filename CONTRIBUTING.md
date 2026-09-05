# Contributing to NovaScientist 🔬

Thank you for your interest in contributing to NovaScientist! We welcome community contributions in autonomous scientific research, agentic architectures, and empirical machine learning.

## Development Workflow

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/chsushen/novascientist.git
   cd novascientist
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Code Style & Quality**:
   - Write clean, type-annotated Python code (PEP 8).
   - Use deterministic seed handling for any randomized experiments.
   - Avoid hardcoded template heuristics or domain-specific lookups without evidence records.

4. **Running Tests**:
   Before submitting a pull request, ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

5. **Submitting Pull Requests**:
   - Provide a clear PR title and description explaining the rationale.
   - Include regression tests for any newly introduced functionality.

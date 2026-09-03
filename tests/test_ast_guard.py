"""Unit tests for AST Static Analysis Guard."""

import pytest
from backend.core.ast_guard import ASTGuard, DataLeakageError


LEAKY_CODE_FIT_BEFORE_SPLIT = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

# Leaky call: fit before train_test_split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
"""

LEAKY_CODE_FIT_ON_TEST = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Contamination: fit on test partition
scaler = StandardScaler()
scaler.fit(X_test)
"""

UNSAFE_CODE_EVAL = """
import numpy as np
X = eval("np.random.randn(10, 10)")
"""

CLEAN_COMPLIANT_CODE = """
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""


def test_detect_fit_before_split():
    report = ASTGuard.analyze_source(LEAKY_CODE_FIT_BEFORE_SPLIT)
    assert not report.is_valid
    assert any("Data Leakage Error" in v for v in report.violations)
    with pytest.raises(DataLeakageError):
        ASTGuard.enforce(LEAKY_CODE_FIT_BEFORE_SPLIT)


def test_detect_fit_on_test_data():
    report = ASTGuard.analyze_source(LEAKY_CODE_FIT_ON_TEST)
    assert not report.is_valid
    assert any("Contamination Error" in v for v in report.violations)


def test_detect_unsafe_eval():
    report = ASTGuard.analyze_source(UNSAFE_CODE_EVAL)
    assert not report.is_valid
    assert any("Critical Security Violation" in v for v in report.violations)


def test_clean_code_passes():
    report = ASTGuard.analyze_source(CLEAN_COMPLIANT_CODE)
    assert report.is_valid
    assert len(report.violations) == 0
    assert report.seed_initialized

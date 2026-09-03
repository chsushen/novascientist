"""AST Static Analysis Guard for Machine Learning Experiment Scripts.

Statically analyzes experiment scripts to prevent:
1. Data Leakage: Calling .fit() or .fit_transform() on global/test data prior to train_test_split.
2. Contamination: Fitting preprocessing scalers on test partitions.
3. Non-Determinism: Invoking stochastic operations without seed initialization.
4. Security/Safety: Flagging unsafe dynamic evaluation constructs (eval, exec).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


class ASTGuardViolation(Exception):
    """Base exception for AST guard violations."""
    pass


class DataLeakageError(ASTGuardViolation):
    """Raised when data leakage is statically detected."""
    pass


class ASTSecurityError(ASTGuardViolation):
    """Raised when unsafe code execution is detected."""
    pass


@dataclass
class DiagnosticReport:
    """Detailed summary of AST static analysis."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    split_line: Optional[int] = None
    seed_initialized: bool = False
    fitted_variables: Set[str] = field(default_factory=set)


class ExperimentASTVisitor(ast.NodeVisitor):
    """AST Visitor enforcing machine learning evaluation integrity."""

    LEAKY_METHODS = {"fit", "fit_transform", "fit_resample"}
    SPLIT_FUNCTIONS = {"train_test_split", "KFold", "StratifiedKFold", "TimeSeriesSplit", "GroupKFold"}
    SEED_FUNCTIONS = {
        "manual_seed", "seed", "set_seed", "default_rng"
    }
    UNSAFE_CALLS = {"eval", "exec", "__import__"}

    def __init__(self) -> None:
        self.split_detected = False
        self.split_line: Optional[int] = None
        self.split_target_vars: Set[str] = set()
        self.train_vars: Set[str] = set()
        self.test_vars: Set[str] = set()
        self.seed_initialized = False
        self.violations: List[str] = []
        self.warnings: List[str] = []
        self.fitted_entities: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        # Check for unsafe function calls (eval, exec)
        func_name = self._get_func_name(node.func)
        if func_name in self.UNSAFE_CALLS:
            msg = f"[Line {node.lineno}] Critical Security Violation: Call to unsafe dynamic evaluation '{func_name}' is forbidden."
            self.violations.append(msg)

        # Check for seed initialization
        if func_name in self.SEED_FUNCTIONS or (isinstance(node.func, ast.Attribute) and node.func.attr in self.SEED_FUNCTIONS):
            self.seed_initialized = True

        # Check for train_test_split calls
        if func_name in self.SPLIT_FUNCTIONS:
            self.split_detected = True
            self.split_line = node.lineno

        # Check for fit / fit_transform data leakage
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.LEAKY_METHODS:
            method_name = node.func.attr
            caller = self._get_node_repr(node.func.value)
            
            # Check arguments passed to .fit()
            arg_names = [self._get_node_repr(arg) for arg in node.args if isinstance(arg, ast.Name)]
            
            if not self.split_detected:
                # fit() called before dataset splitting -> Data Leakage!
                msg = (
                    f"[Line {node.lineno}] Data Leakage Error: Method '{caller}.{method_name}()' invoked on data "
                    f"({', '.join(arg_names) if arg_names else 'global dataset'}) BEFORE train_test_split(). "
                    f"All preprocessing estimators must be fitted strictly on the training partition."
                )
                self.violations.append(msg)
            else:
                # If splitting occurred, ensure test variables are not passed to fit()
                for arg in arg_names:
                    if "test" in arg.lower() or "val" in arg.lower() or arg in self.test_vars:
                        msg = (
                            f"[Line {node.lineno}] Contamination Error: Method '{caller}.{method_name}()' fitted on test/validation "
                            f"partition '{arg}'. Test data must only be transformed via .transform()."
                        )
                        self.violations.append(msg)
            
            self.fitted_entities.add(caller)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Check if assigning train_test_split output
        if isinstance(node.value, ast.Call):
            func_name = self._get_func_name(node.value.func)
            if func_name in self.SPLIT_FUNCTIONS:
                self.split_detected = True
                self.split_line = node.lineno
                for target in node.targets:
                    if isinstance(target, (ast.Tuple, ast.List)):
                        for idx, elt in enumerate(target.elts):
                            if isinstance(elt, ast.Name):
                                name = elt.id
                                if "train" in name.lower() or idx % 2 == 0:
                                    self.train_vars.add(name)
                                if "test" in name.lower() or "val" in name.lower() or idx % 2 == 1:
                                    self.test_vars.add(name)
        self.generic_visit(node)

    def _get_func_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _get_node_repr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_repr(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return "expr"


class ASTGuard:
    """Public interface for static experiment analysis."""

    @classmethod
    def analyze_source(cls, source_code: str, filename: str = "<experiment>") -> DiagnosticReport:
        """Statically inspect source code and return a DiagnosticReport."""
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            return DiagnosticReport(
                is_valid=False,
                violations=[f"Python Syntax Error in {filename} at line {e.lineno}: {e.msg}"],
            )

        visitor = ExperimentASTVisitor()
        visitor.visit(tree)

        if not visitor.seed_initialized:
            visitor.warnings.append(
                "Stochastic Determinism Warning: No explicit random seed initialization (e.g. torch.manual_seed, np.random.seed) detected."
            )

        is_valid = len(visitor.violations) == 0
        return DiagnosticReport(
            is_valid=is_valid,
            violations=visitor.violations,
            warnings=visitor.warnings,
            split_line=visitor.split_line,
            seed_initialized=visitor.seed_initialized,
            fitted_variables=visitor.fitted_entities,
        )

    @classmethod
    def enforce(cls, source_code: str, filename: str = "<experiment>") -> DiagnosticReport:
        """Analyze source code and raise DataLeakageError if violations are found."""
        report = cls.analyze_source(source_code, filename)
        if not report.is_valid:
            violations_str = "\n".join(report.violations)
            raise DataLeakageError(f"AST Static Analysis Failed for {filename}:\n{violations_str}")
        return report

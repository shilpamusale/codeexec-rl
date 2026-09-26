"""Phase 0 smoke test: proves the package imports and CI collects at least one test.

Exists so `pytest` has something to collect (an empty suite exits non-zero) and so
CI verifies the package is importable. Real tests arrive with Phase 1 (the sandbox).
"""

import importlib


def test_package_imports() -> None:
    """The top-level package imports without error."""
    module = importlib.import_module("codeexec")
    assert module is not None

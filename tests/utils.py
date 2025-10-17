"""Utility helpers for lightweight test execution without pytest.

Provides assertion helpers and file utilities for running tests without
external dependencies. Enables simple, fast test execution."""

import math
import os


def assert_close(actual: float, expected: float, rel: float = 1e-4, msg: str = ""):
    """Assert that two floating point values are approximately equal.
    
    Uses relative tolerance for numerical comparisons to avoid floating-point
    precision issues. Critical for validating simulation calculations.
    """
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=1e-12):
        suffix = f" ({msg})" if msg else ""  # Optional context message
        raise AssertionError(f"Expected {expected} ± {rel}, got {actual}{suffix}")


def expect_raises(exception, func, *args, **kwargs):
    """Assert that a function raises a specific exception.
    
    Used to test error handling and validation logic in configuration
    and simulation components. Ensures proper error responses.
    """
    try:
        func(*args, **kwargs)  # Execute function with provided arguments
    except exception:
        return  # Expected exception was raised - test passes
    raise AssertionError(f"Expected {exception.__name__} to be raised")


def file_exists(path: str) -> bool:
    """Check if a file exists at the given path.
    
    Simple wrapper around os.path.exists for test file validation.
    Used to verify output file generation in metrics tests.
    """
    return os.path.exists(path)  # Return True if file exists, False otherwise

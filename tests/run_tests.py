"""Simple test runner for the Feels simulation suite.

Lightweight test execution without external dependencies like pytest.
Automatically discovers and runs all test functions across test modules.
Provides clear reporting of successes and failures."""

import importlib
import inspect
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path for module imports
ROOT = Path(__file__).resolve().parents[1]  # Navigate to project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # Add to Python path

# Additional path setup for reliability
project_root = Path(__file__).parent.parent  # Alternative path to project root
sys.path.insert(0, str(project_root))        # Ensure project is importable

# Test modules to execute (order matters for dependency resolution)
TEST_MODULES = [
    "tests.test_core",         # Core simulation engine tests
    "tests.test_fees",         # Fee routing and scenario tests
    "tests.test_participants", # Participant behavior tests
    "tests.test_metrics",      # Metrics and analysis tests
]


def iter_tests(module):
    """Discover all test functions in a module.
    
    Finds functions that start with 'test_' prefix following pytest conventions.
    Returns iterator of (name, function) pairs for execution.
    """
    for name, obj in inspect.getmembers(module):  # Examine all module members
        if name.startswith("test_") and inspect.isfunction(obj):  # Test function criteria
            yield name, obj  # Return test name and function object


def main() -> int:
    """Main test execution function.
    
    Discovers and runs all tests across configured modules.
    Returns 0 for success, 1 for failure (standard exit codes).
    """
    failures = []  # Track failed tests for reporting
    total = 0      # Count total tests executed
    
    # Execute tests from each configured module in dependency order
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)  # Import test module
        
        # Run each test function in the module
        for name, test_func in iter_tests(module):
            total += 1  # Increment test counter
            try:
                test_func()  # Execute test function (no arguments)
            except Exception as exc:  # Catch any test failures or assertion errors
                failures.append((module_name, name, exc, traceback.format_exc()))
    
    # Report results
    if failures:
        # Print failure summary
        print(f"\n{len(failures)} test(s) failed out of {total}:\n", file=sys.stderr)
        
        # Print detailed failure information
        for module_name, name, exc, tb in failures:
            print(f"[{module_name}] {name} FAILED: {exc}", file=sys.stderr)
            print(tb, file=sys.stderr)  # Print full traceback for debugging
        return 1  # Exit with failure code
    
    # All tests passed
    print(f"All {total} tests passed.")
    return 0  # Exit with success code


if __name__ == "__main__":
    sys.exit(main())

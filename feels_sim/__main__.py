"""
Entry point for running feels_sim as a module.

Usage:
    python -m feels_sim.cli sweep --sweep-type full
"""

import sys
from .cli import main

if __name__ == "__main__":
    # Add the CLI functionality to the package main
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Remove "cli" from args and run CLI
        sys.argv.pop(1)
        main()
    else:
        print("Feels Protocol Simulation Package")
        print("Usage:")
        print("  python -m feels_sim.cli sweep --help")
        print("  python -m feels_sim.cli sweep --sweep-type full --hours 168")
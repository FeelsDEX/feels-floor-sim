# Feels Floor Price Simulation

# Show available commands
default:
    @echo "Available commands:"
    @echo "  dev     - Enter development environment"
    @echo "  sim     - Run simulation (168 hours)"
    @echo "  sweep   - Run parameter sweep analysis"
    @echo "  test    - Run all tests (pass 'cov' for coverage)"
    @echo "  jupyter - Start Jupyter Lab"

# Enter development environment
dev:
    nix develop

# Run simulation with analysis
sim:
    @nix develop --command python -c "from feels_sim.core import FeelsSimulation; from feels_sim.config import SimulationConfig; sim = FeelsSimulation(SimulationConfig(enable_participant_behavior=False)); result = sim.run(hours=168); print('Simulation completed successfully!'); print('Snapshots:', len(result.snapshots)); print('Final floor price: $' + str(round(result.snapshots[-1].floor_price_usd, 6)))"

# Run all tests
test *ARGS:
    @echo "Testing core modules and dependencies..."
    @nix develop --command python -c "import polars as pl; print(f'✓ polars {pl.__version__} available')"
    @nix develop --command python -c "import numpy as np; print(f'✓ numpy {np.__version__} available')"
    @nix develop --command python -c "import agentpy as ap; print(f'✓ agentpy {ap.version.__version__} available')"
    @nix develop --command python3 tests/run_tests.py

# Run parameter sweep analysis
sweep *ARGS:
    @nix develop --command python -m feels_sim.cli {{ARGS}}

# Start Jupyter Lab
jupyter:
    @nix develop --command jupyter lab

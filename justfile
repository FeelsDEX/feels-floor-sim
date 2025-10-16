# Feels Floor Price Simulation

# Show available commands
default:
    @echo "Available commands:"
    @echo "  dev     - Enter development environment"
    @echo "  sim     - Run simulation (168 hours)"
    @echo "  test    - Run all tests (pass 'cov' for coverage)"
    @echo "  jupyter - Start Jupyter Lab"

# Enter development environment
dev:
    nix develop

# Run simulation with analysis
sim:
    @nix develop --command python -c "from feels_sim.config import SimulationConfig; from feels_sim.core import FeelsSimulation; from feels_sim.metrics import analyze_results; sim = FeelsSimulation(SimulationConfig()); results = sim.run(hours=168); analysis = analyze_results(results); print(f'POMM deployments: {results.snapshots[-1].floor_state.pomm_deployments_count}'); print(f'Final floor price: \${results.snapshots[-1].floor_price_usd:.6f}'); print(f'Floor growth rate: {analysis[\"floor_growth_rate_annual\"]:.2%}'); print(f'Floor/market ratio: {analysis[\"avg_floor_to_market_ratio\"]:.2%}')"

# Run all tests
test *ARGS:
    #!/usr/bin/env bash
    if [[ "{{ARGS}}" == *"cov"* ]]; then
        @nix develop --command python -m pytest tests/ --cov=feels_sim --cov-report=term-missing
    else
        @nix develop --command python -m pytest tests/ -v
    fi

# Start Jupyter Lab
jupyter:
    @nix develop --command jupyter lab
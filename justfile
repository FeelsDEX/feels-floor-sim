# Feels Floor Price Simulation

# Show available commands
default:
    @echo "Available commands:"
    @echo "  dev       - Enter development environment"
    @echo "  sim       - Run basic simulation (168 hours)"
    @echo "  test      - Run test suite (24 hours)"
    @echo "  validate  - Run Phase 2 validation from implementation document"
    @echo "  analyze   - Run metrics analysis"
    @echo "  jupyter   - Start Jupyter Lab"
    @echo "  notebook  - Start Jupyter Notebook"

# Enter development environment
dev:
    nix develop

# Run basic simulation
sim:
    nix develop --command python -c "from feels_sim.config import SimulationConfig; from feels_sim.core import FeelsSimulation; sim = FeelsSimulation(SimulationConfig()); results = sim.run(hours=168); print('POMM deployments:', results.snapshots[-1].floor_state.pomm_deployments_count); print('Final floor price: $%.6f' % results.snapshots[-1].floor_price_usd)"

# Run test suite
test:
    nix develop --command python -c "from feels_sim.core import FeelsSimulation, SimulationConfig; sim = FeelsSimulation(SimulationConfig()); results = sim.run(hours=24); print('✓', len(results.snapshots), 'snapshots generated'); print('✓ Floor monotonic:', all(results.snapshots[i].floor_price_feelssol >= results.snapshots[i-1].floor_price_feelssol for i in range(1, len(results.snapshots))))"

# Run Phase 2 validation from implementation document
validate:
    nix develop --command python -c "from feels_sim.config import SimulationConfig; from feels_sim.core import FeelsSimulation; sim = FeelsSimulation(SimulationConfig(base_fee_bps=30)); results = sim.run(hours=168); deployments = [snap for snap in results.snapshots if snap.events.get('pomm_deployed', False)]; buffer_final = results.snapshots[-1].floor_state.buffer_balance; volume_total = sum(h['total_volume'] for h in results.hourly_aggregates); mintable_final = results.snapshots[-1].floor_state.mintable_feelssol; print(f'POMM deployments recorded: {len(deployments)}'); print(f'Final Buffer balance: {buffer_final:.2f}'); print(f'Mintable FeelsSOL: {mintable_final:.2f}'); print(f'Total trading volume: {volume_total:.0f}'); print(f'Final floor (USD): {results.snapshots[-1].floor_price_usd:.4f}'); assert all(results.snapshots[i].floor_price_usd >= results.snapshots[i-1].floor_price_usd for i in range(1, len(results.snapshots))); assert mintable_final > 0; print('✓ Monotonic floor verified')"

# Run metrics analysis test
analyze:
    nix develop --command python -c "from feels_sim.config import SimulationConfig; from feels_sim.core import FeelsSimulation; from feels_sim.metrics import analyze_results; sim = FeelsSimulation(SimulationConfig()); results = sim.run(hours=168); analysis = analyze_results(results); print(f'Floor growth rate: {analysis[\"floor_growth_rate_annual\"]:.2%}'); print(f'Average floor/market ratio: {analysis[\"avg_floor_to_market_ratio\"]:.2%}'); print(f'Total POMM deployments: {analysis[\"pomm_deployments\"]}'); print('✓ Analysis complete')"

# Start Jupyter Lab
jupyter:
    nix develop --command jupyter lab

# Start Jupyter Notebook
notebook:
    nix develop --command jupyter notebook
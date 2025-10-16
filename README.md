# Feels Floor Price Simulation

A simulation framework for analyzing Feels protocol floor price dynamics.

## Overview

The Feels Protocol implements a concentrated liquidity AMM where FeelsSOL serves as the universal routing token, backed 1:1 by yield-bearing JitoSOL. The protocol's key innovation is a Protocol-Owned Market Maker, which creates monotonically increasing price floors for tokens.

This simulation models the protocol's core economic dynamics:
- **Trading generates fees** → distributed to Buffer (~98.5%), protocol (~1%), creators (~0.5%)
- **Buffer automatically funds POMM** → deploys accumulated fees as floor liquidity
- **JitoSOL backing earns yield** → provides additional FeelsSOL minting capacity
- **POMM deploys liquidity** → advances floor price (monotonic increase)
- **Higher floors attract trading** → increases volume and fees

**Key Simulation Goal**: Analyze different fee allocation strategies to recommend optimal parameters for protocol launch, balancing floor advancement speed with protocol sustainability.

The framework enables analysis of trading activity, fee generation, JitoSOL yield accumulation, POMM deployment strategies, governance funding decisions, and liquidity provider economics.

## Quick Start

```bash
# Run a basic simulation
just sim

# Start Jupyter
just jupyter

# Run test suite
just test

# Enter development environment
just dev
```

### Jupyter Notebook Usage

Jupyter is the primary way to configure and run simulations. The Nix environment automatically registers the `feels-floor-sim` kernel. In the notebook:

```python
from feels_sim.config import SimulationConfig
from feels_sim.core import FeelsSimulation
from feels_sim.metrics import analyze_results, create_summary_plots

# Configure simulation parameters
config = SimulationConfig(
    base_fee_bps=30,            # 0.30% swap fee
    treasury_share_pct=1.0,     # 1% to protocol treasury (current default)
    creator_share_pct=0.5,      # 0.5% to creators (current default)
    # buffer_share_pct automatically calculated as remainder (98.5%)
    jitosol_yield_apy=0.07,     # 7% JitoSOL yield
    pomm_deployment_ratio=0.5,  # Deploy 50% of available funding
    sol_volatility_daily=0.05   # 5% daily SOL volatility
)

# Or use predefined fee scenarios for parameter exploration
config = SimulationConfig.create_fee_scenario("protocol_sustainable")

# Run simulation
sim = FeelsSimulation(config)
results = sim.run(hours=168)

# Analyze results
analysis = analyze_results(results)
create_summary_plots(results)
```

## Project Structure

```
feels-floor-sim/
├── feels_sim/           # Core simulation package
│   ├── config.py        # Configuration and parameters
│   ├── core.py          # Main simulation engine
│   ├── market.py        # Market environment and price evolution
│   └── metrics.py       # Analysis and reporting utilities
├── tests/               # Test suite
├── work/                # Implementation planning
└── docs/                # Protocol documentation
```

## License

Apache 2.0
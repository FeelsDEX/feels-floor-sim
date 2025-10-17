# Feels Floor Price Simulation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FeelsDEX/feels-floor-sim/HEAD?labpath=notebooks%2F01_baseline.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FeelsDEX/feels-floor-sim/blob/main/notebooks/01_baseline.ipynb)
[![NBViewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/FeelsDEX/feels-floor-sim/blob/main/notebooks/01_baseline.ipynb)

A simulation framework for analyzing Feels protocol floor price dynamics.

## Overview

The Feels Protocol implements a concentrated liquidity AMM where FeelsSOL serves as the universal routing token, backed 1:1 by yield-bearing JitoSOL. The protocol's key innovation is a Protocol-Owned Market Maker, which creates monotonically increasing price floors for tokens.

This simulation models the protocol's core economic dynamics:
- **Trading generates fees** → distributed to Buffer (~85%), protocol (~10%), creators (~5%)
- **Buffer automatically funds POMM** → deploys accumulated fees as floor liquidity
- **JitoSOL backing earns yield** → provides additional FeelsSOL minting capacity (~7% APR)
- **POMM deploys liquidity** → advances floor price (monotonic increase)
- **JIT liquidity system** → provides early-stage market bootstrapping
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

# Run parameter sweeps
just sweep --sweep-type full --hours 168
```

### Parameter Sweep Analysis

For systematic parameter exploration and optimization:

```bash
# Run comprehensive parameter sweep (fee ranges + market conditions)
just sweep --sweep-type full --hours 168

# Run fee range sweep only
just sweep --sweep-type fee_range --scenarios current_default protocol_sustainable

# Run market condition sweep only  
just sweep --sweep-type market_conditions --hours 24

# Direct CLI usage
python -m feels_sim.cli sweep --help
python -m feels_sim.cli sweep --sweep-type fee_range --fee-range 20 30 40 50
```

Results are saved in `experiments/runs/` with timestamped files for analysis.

### Interactive Notebooks

**Run Online (No Installation Required):**
- **Binder**: Click the Binder badge above for a fully interactive environment
- **Google Colab**: Click the Colab badge for GPU-accelerated execution

**Local Development:**
Jupyter is the primary way to configure and run simulations. The Nix environment automatically registers the `feels-floor-sim` kernel. In the notebook:

```python
from feels_sim.config import SimulationConfig
from feels_sim.core import FeelsSimulation
from feels_sim.metrics import analyze_results, create_summary_plots

# Configure simulation parameters
config = SimulationConfig(
    base_fee_bps=30,            # 0.30% swap fee
    treasury_share_pct=10.0,    # 10% to protocol treasury
    creator_share_pct=5.0,      # 5% to creators
    # buffer_share_pct automatically calculated as remainder (~85%)
    jitosol_yield_apy=0.07,     # 7% JitoSOL yield
    pomm_deployment_ratio=0.5,  # Deploy 50% of available funding
    volatility_daily=0.05,      # 5% daily SOL volatility
    jit_enabled=True            # Enable JIT liquidity system
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
│   ├── cli.py           # Command-line interface for parameter sweeps
│   ├── config.py        # Configuration and parameters
│   ├── core.py          # Main simulation engine with AgentPy integration
│   ├── market.py        # Market environment and price evolution
│   ├── metrics.py       # Polars-based analysis and reporting
│   ├── plotting.py      # Enhanced seaborn visualization
│   ├── pricing.py       # JIT liquidity management
│   └── participants.py  # Participant behavior modeling
├── notebooks/           # Analysis notebooks
│   ├── 01_baseline.ipynb      # Baseline scenario analysis
│   └── 02_parameter_sweep.ipynb # Parameter optimization analysis
├── tests/               # Test suite
├── experiments/         # Results and configuration files
├── docs/                # Protocol and implementation documentation
└── flake.nix           # Nix development environment
```

## License

Apache 2.0
# Feels Simulation Implementation Guide

This document covers the technical implementation details, metrics collection, analysis framework, and optimization strategies for the Feels floor price simulation system.

---

## Key Metrics and Visualization

### Primary Floor Price Metrics

The core objective of the simulation is understanding floor price dynamics under different scenarios. Multiple metrics capture different aspects of floor price performance. These provide comprehensive insights.

**Floor Price Trajectory Analysis**:
- `floor_price_usd(t)`: Absolute floor price in USD over time
- `floor_price_growth_rate`: Annualized growth rate of floor price
- `floor_price_volatility`: Standard deviation of daily floor price changes
- `floor_advancement_frequency`: Number of POMM deployments per time period
- `floor_level_aggregates`: Daily, weekly, and yearly summaries of average floor price and incremental changes

**Relative Performance Metrics**:
- `floor_to_market_ratio(t)`: Floor price as percentage of market price
- `floor_protection_efficiency`: How well floor price protects during market downturns
- `floor_catch_up_speed`: Rate at which floor price approaches market price during growth

### Market Activity and Health Indicators

Understanding the broader market dynamics provides context for floor price performance and helps identify sustainable parameter configurations.

**Trading Volume Analysis**:
- `daily_volume_feelssol`: Total daily trading volume in FeelsSOL terms
- `volume_concentration`: Distribution of trading across different participant types
- `volume_volatility`: Standard deviation of daily volume changes
- `volume_trend_correlation`: Relationship between volume and SOL price trends

**Market Efficiency Metrics**:
- `bid_ask_spread`: Average spread between best bid and ask prices
- `price_impact_per_trade`: Average price movement per unit of trading volume
- `arbitrage_opportunity_frequency`: Rate of profitable arbitrage opportunities
- `market_depth`: Total liquidity available within specified price ranges

**Participant Behavior Indicators**:
- `participant_count_by_type`: Number of active participants in each category
- `trading_frequency_by_type`: Average trades per participant type per day
- `trade_size_distribution`: Statistical distribution of individual trade sizes
- `participant_retention_rate`: Persistence of trading activity over time

### System Performance and Sustainability

Long-term system health requires balancing floor price growth with market functionality and participant satisfaction.

**Fee Collection and Distribution**:
- `total_fees_collected`: Cumulative fee generation over time
- `fee_distribution_breakdown`: Actual allocation across Buffer (~85%), protocol (~10%), creator (~5%)
- `buffer_accumulation_rate`: Rate of fee accumulation in Buffer for POMM deployment
- `pomm_deployment_frequency`: How often Buffer triggers automatic POMM deployment
- `protocol_treasury_growth`: Accumulation rate of protocol fees
- `accrual_aggregates`: Daily/weekly/yearly totals for protocol, creator, and Buffer receipts

**POMM System Performance**:
- `pomm_deployment_frequency`: Rate of automatic floor advancements from Buffer thresholds
- `pomm_capital_efficiency`: Floor price advancement per FeelsSOL deployed
- `pomm_timing_effectiveness`: Relationship between trading volume and deployment timing
- `buffer_sustainability`: Rate of fee accumulation vs deployment consumption
- `funding_source_analysis`: Breakdown of POMM funding sources (Buffer fees, synthetic minting, optional supplements)

**Market Participant Economics**:
- `trading_volume_impact`: Effect of fee levels on trading volume
- `participant_retention`: How fee structure affects long-term participation
- `floor_attraction_effect`: How floor price levels influence trading activity
- `ecosystem_growth_rate`: Compound effect of floor advancement on total activity

### Risk and Stress Testing Metrics

The simulation must evaluate system performance under adverse conditions to ensure robust parameter selection.

**Downside Protection Analysis**:
- `maximum_drawdown`: Largest peak-to-trough decline in floor price
- `floor_price_stability`: Volatility of floor price during market stress
- `crisis_trading_volume`: Trading activity during major market downturns
- `system_resilience_score`: Composite measure of performance during stress events

**Parameter Sensitivity Analysis**:
- `fee_elasticity`: Percentage change in volume per percentage change in fees
- `distribution_impact`: Effect of fee split changes on floor price growth
- `threshold_sensitivity`: Impact of POMM threshold changes on advancement frequency
- `yield_rate_sensitivity`: Floor price growth sensitivity to backing yield changes

### Visualization Strategy

Effective visualization communicates complex simulation results to facilitate parameter optimization and system understanding.

**Time Series Dashboards**: Multi-panel displays showing floor price evolution alongside key drivers such as trading volume, fee collection, and market conditions. These dashboards enable identification of causal relationships and system dynamics.

**Parameter Sensitivity Heatmaps**: Two-dimensional visualizations showing how floor price outcomes vary across different combinations of protocol parameters. These maps reveal optimal parameter regions and highlight sensitive parameter interactions.

**Scenario Comparison Plots**: Side-by-side comparisons of floor price trajectories under different market environment assumptions. These comparisons help identify robust parameter configurations that perform well across diverse conditions.

**Distribution Analysis**: Histograms and box plots showing the distribution of outcomes across multiple simulation runs with different random seeds. These visualizations quantify uncertainty and risk in floor price projections.

---

## Implementation Strategy

### Project Structure Blueprint

A suggested layout keeps reusable simulation code separated from experiments and dashboards:

```text
feels-floor-sim/
├─ pyproject.toml              # tooling + dependency pinning
├─ feels_sim/                  # importable simulation package
│  ├─ __init__.py
│  ├─ config.py                # SimulationConfig + scenario loader
│  ├─ core.py                  # minute engine, state dataclasses, liquidity math, POMM logic
│  ├─ market.py                # SOL price + sentiment models
│  ├─ participants.py          # all participant behaviors in one place
│  └─ metrics.py               # metrics collector and reporting helpers
├─ scripts/
│  └─ run_simulation.py        # CLI entry point for single/batch runs
├─ tests/
│  └─ test_core.py             # core and metrics unit tests (expand only if necessary)
├─ experiments/
│  ├─ configs/                 # scenario + calibration JSON/YAML
│  └─ runs/                    # generated outputs (gitignored)
└─ notebooks/
   ├─ calibration.ipynb        # parameter fitting
   ├─ 01_baseline.ipynb        # exploratory analysis
   └─ 02_parameter_sweep.ipynb # batch result review
```

Guidelines:
- Keep simulation logic confined to the handful of modules above; avoid new files unless required.
- Use configuration files and notebooks for variability rather than duplicating code.
- Persist each simulation run’s metadata (scenario hash, git commit, seeds) alongside exports for reproducibility.

### Simulation Framework Architecture

The simulation implementation follows a modular design that enables efficient parameter exploration and result analysis.

**Core Engine Structure**:
```python
class FeelsSimulation:
    def __init__(self, config: SimulationConfig):
        self.market_state = MarketState(config.initial_conditions)
        self.floor_state = FloorFundingState()
        self.participants = ParticipantPool(config.participant_mix)
        self.environment = MarketEnvironment(config.external_conditions)
        
    def run_simulation(self, duration_hours: int) -> SimulationResults:
        for minute in range(duration_hours * 60):
            self.step_minute()
            if (minute + 1) % 60 == 0:
                self.complete_hour()
        return self.collect_results()
```

### Simulation Step Execution

Minute-level substeps feed into hourly aggregates, ensuring the 60-second POMM cooldown is respected while keeping analytics manageable.

```python
def step_minute(self):
    self.update_market_environment()
    self.process_participant_decisions()
    self.execute_swaps_against_curve()
    self.distribute_fees()
    self.update_floor_funding_flows()
    self.evaluate_pomm_deployments()

def complete_hour(self):
    self.rebalance_participants()
    self.record_metrics(interval="hour")
```

**Event Processing Details**:
- `update_market_environment()`: Updates external factors like SOL price, staking yield, and sentiment.
- `process_participant_decisions()`: Models trading behavior for each participant type based on current market conditions.
- `execute_swaps_against_curve()`: Consumes orders across tick-indexed liquidity buckets to compute execution price and slippage.
- `distribute_fees()`: Routes ~85% of fees into Buffer, ~10% to protocol treasury, and ~5% to creators.
- `update_floor_funding_flows()`: Consolidates Buffer growth with synthetic minting and updates deployment readiness.
- `evaluate_pomm_deployments()`: Determines whether sufficient capital and timing conditions are met to advance the floor.
- `rebalance_participants()`: Reassesses LP and trader positioning at hourly boundaries.
- `record_metrics()`: Stores key metrics for analysis and visualization.

### Tick-Level Liquidity Representation

Liquidity is stored as tick-indexed buckets (`liquidity_curve`). Swap execution walks the curve, removing or adding liquidity as positions rebalance. This enables realistic microstructure metrics:

```python
def quote_swap(amount_in: float, direction: SwapDirection, curve: Dict[int, float]):
    # Iterate over ticks, consume liquidity, and accumulate price impact
    ...
```

Snapshots of the curve at minute and hourly intervals feed KPIs such as `bid_ask_spread`, `price_impact_per_trade`, and `market_depth`.

### Floor Funding Pipeline Integration

`FloorFundingState` mirrors the conceptual funding pipeline. Key responsibilities:

```python
@dataclass
class FloorFundingState:
    treasury_balance: float
    creator_balance: float
    buffer_balance: float
    mintable_feelssol: float
```

- Route the Buffer share of swap fees into `buffer_balance` each minute.
- Accrue synthetic FeelsSOL minting into `mintable_feelssol` at the configured drift rate.
- Track treasury and creator balances so optional supplements can be attributed separately from automatic funding.

### Configuration & Scenario Management

- Define all tunable parameters inside `config.py` dataclasses (fee schedule, buffer share, mint rate, liquidity granularity, participant elasticities).
- Store scenario bundles as YAML/JSON under `experiments/configs/` with descriptive names (`base.yml`, `bear_high_yield.yml`). Version them in git so analyses reference immutable inputs.
- Expose overrides via environment variables/CLI flags so CI and dashboards can run the same scenario with minor tweaks.
- Each simulation invocation should emit a `run.json` (or sqlite record) capturing scenario hash, git commit, random seeds, elapsed runtime, and day/week/year aggregate metrics.

### Development Phases

1. **Foundations**: Implement core state containers, liquidity math, and deterministic swap execution. Add unit tests comparing against reference Uniswap v3 calculations.
2. **Funding Mechanics**: Build the floor funding pipeline (fee routing, synthetic minting, cooldown enforcement) using scripted scenarios to validate multiple deployments per hour.
3. **Participant Behaviors**: Port behavior models into `participants/`, parameterize them with calibration inputs, and add regression tests to keep aggregate volume/fee responses stable.
4. **Metrics & Reporting**: Finish `MetricsCollector` and aggregation utilities, ensuring day/week/year rollups match expectations on synthetic data. Wire notebooks/dashboards to the produced parquet or arrow files.
5. **Optimization & Analysis**: Implement parameter exploration tools for scenario analysis and result comparison.

### Testing Strategy

- **Unit Tests**: Liquidity math, funding pipeline invariants (no double counting), buffer accounting, and metric aggregations.
- **Property Tests**: Ensure floor price never decreases after deployment, deployed capital never exceeds allocations, fee splits always sum to 100%, and minting drift matches the configured rate.
- **Integration Tests**: Deterministic 24-hour and 7-day scenarios with snapshot comparisons of floor price, protocol accrual, and LP earnings trajectories.

**Modular Component Design**: Each major system component (market environment, participant behavior, POMM deployment, fee collection) is implemented as a separate module with well-defined interfaces. This modularity enables testing individual components and easy modification of specific behaviors.

### Data Collection and Analysis Pipeline

Comprehensive data collection enables detailed analysis of simulation results and identification of optimal parameter configurations.

**Metrics Collection Framework**:
```python
class MetricsCollector:
    def __init__(self):
        self.time_series = defaultdict(list)
        self.summary_stats = {}
        
    def record_timestep(self, timestamp: int, market_state: MarketState, 
                       floor_state: FloorFundingState, trading_snapshot: Dict[str, float]):
        # Record all relevant metrics for this timestep
        # Update rolling aggregates for daily, weekly, yearly windows
        
    def calculate_summary_statistics(self) -> Dict[str, float]:
        # Compute key performance indicators across entire simulation
```

**Result Aggregation**: Multiple simulation runs with different random seeds provide statistical confidence in results. The framework automatically aggregates results across runs and computes confidence intervals for key metrics.

**Visualization Integration**: The simulation framework includes built-in visualization capabilities using Polars for data manipulation and matplotlib/plotly for chart generation. This integration enables rapid iteration and result interpretation.

### Calibration & Validation Workflow

- **Historical Benchmarks**: Import swap, liquidity, and fee data from comparable Solana CLMMs to seed priors for trade size distributions and tick utilization.
- **Partner Research**: Track survey responses and preliminary commitments from early LPs/market makers to parameterize governance cadence, required yields, and acceptable fee splits.
- **Backtesting Harness**: Replay historical SOL price paths and Uniswap v3 fee accrual data, checking that simulated volume/fee correlations stay within calibration bands.
- **Continuous Monitoring**: Maintain a calibration notebook that flags divergence between simulated and observed metrics once mainnet data becomes available, triggering parameter review.

### Parameter Optimization Workflow

Systematic parameter optimization requires structured exploration of the parameter space with efficient search algorithms.

**Grid Search Implementation**: For initial exploration, the framework implements comprehensive grid search across specified parameter ranges. This approach ensures coverage of the full parameter space but may be computationally intensive for high-dimensional problems.

**Bayesian Optimization**: For fine-tuning optimal regions identified through grid search, the framework implements Bayesian optimization using Gaussian process models. This approach efficiently explores promising parameter regions with fewer simulation runs.

**Multi-Objective Optimization**: The simulation optimizes multiple objectives simultaneously (floor price growth, market efficiency, participant satisfaction) using Pareto frontier analysis. This approach identifies trade-offs between competing objectives and enables informed parameter selection.

**Sensitivity Analysis**: The framework includes tools for analyzing parameter sensitivity around optimal configurations. This analysis helps understand which parameters most significantly impact outcomes and guides robust parameter selection.

### Critical Implications for Parameter Optimization

**Fundamental Model Change**: The correct understanding that ~85% of fees automatically flow to Buffer for POMM deployment fundamentally alters the parameter optimization problem:

**Key Implications**:
1. **Fee-Driven Floor Growth**: Primary floor funding comes from automatic Buffer accumulation (~85% of swap fees)
2. **Trading Volume Coupling**: Floor advancement directly depends on trading activity and fee generation
3. **Fee Structure Optimization**: Fee levels affect both trading volume and floor advancement speed
4. **Feedback Loop Dynamics**: Higher floors attract more trading, creating compound growth

**Modified Optimization Objectives**:
1. **Trading Volume Maximization**: Optimize fee levels to maximize sustainable trading activity
2. **Buffer Allocation Balance**: Balance floor advancement speed with protocol treasury needs
3. **Ecosystem Growth**: Create conditions for positive feedback between trading and floor advancement
4. **Long-term Sustainability**: Ensure fee structure supports continued ecosystem growth

**Parameter Exploration Priorities**:
- **Fee-Volume Elasticity**: Impact of fee changes on trading volume and Buffer accumulation
- **Buffer Allocation Optimization**: Optimal split between Buffer, protocol, and creator shares
- **Feedback Loop Analysis**: How floor advancement affects trading behavior and volume
- **Growth Sustainability**: Long-term viability of fee-driven floor advancement model

This implementation guide provides the foundation for building a robust simulation framework that can systematically explore Feels protocol parameters with correct understanding of the fee-driven floor advancement mechanism.

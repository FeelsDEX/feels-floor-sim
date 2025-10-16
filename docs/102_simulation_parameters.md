# Feels Simulation Parameters and Scenarios

This document defines all configurable parameters and test scenarios for the Feels floor price simulation. These parameters control the external market environment, protocol configuration, and testing scenarios used to explore optimal protocol settings.

---

## Environment Variables

### External Market Factors

The simulation operates within a broader cryptocurrency market environment that influences trading behavior and system performance. These external factors provide the context within which the Feels market operates but are not directly controlled by protocol parameters.

**SOL/USD Price Environment**

These parameters define the broader cryptocurrency market conditions that affect trading behavior and risk appetite:

| Parameter | Description | Baseline Range |
|-----------|-------------|----------------|
| `sol_usd_price` | Current SOL price in USD | $50-200 |
| `sol_price_volatility` | Daily price change standard deviation | 3-8% |
| `sol_trend_direction` | Market trend bias (-1 = strong bearish, 1 = strong bullish) | -1 to 1 |
| `macro_market_sentiment` | Overall crypto market condition affecting risk appetite | Low/Medium/High |

**Staking Yield Parameters**

These parameters control the yield environment for the underlying assets backing FeelsSOL:

| Parameter | Description | Baseline Range |
|-----------|-------------|----------------|
| `base_sol_staking_yield` | Standard SOL staking APY | 6-8% |
| `jitosol_mev_premium` | Additional yield from MEV extraction | 1-2% |
| `jitosol_total_yield` | Combined JitoSOL yield (base + MEV premium) | 7-9% |

**FeelsSOL Backing Specification**: JitoSOL appreciates relative to SOL at approximately 7% APY due to staking rewards and MEV extraction. This appreciation creates the capacity to mint additional FeelsSOL while maintaining full backing. For simulation purposes, we use 7% APY as the rate at which additional FeelsSOL can be minted for protocol purposes (such as floor advancement) while maintaining the SOL price target and full JitoSOL backing.

**FeelsSOL Backing Model**

FeelsSOL is a synthetic asset targeting SOL price, fully backed by JitoSOL reserves:

| Parameter | Description | Value |
|-----------|-------------|-------|
| `feelssol_sol_price_target` | FeelsSOL targets SOL price | 1:1 (price target) |
| `jitosol_appreciation_rate` | JitoSOL appreciation vs SOL | 7% APY |
| `feelssol_minting_rate` | Rate of additional FeelsSOL minting from backing appreciation | 7% APY |
| `redemption_rate` | FeelsSOL redeemed for JitoSOL at current JitoSOL/SOL rate | Variable |
| `backing_compound_frequency` | How often backing appreciation compounds | Daily |

### Protocol Configuration Space

These represent the tunable parameters that form the core of our optimization study. Each simulation run explores different combinations of these values to identify optimal configurations.

**Fee Structure Parameters**

These parameters control the fee structure for swaps and trading activities:

| Parameter | Description | Range/Value |
|-----------|-------------|-------------|
| `base_fee_bps` | Base swap fee in basis points | 10-100 bps |
| `impact_fee_enabled` | Whether dynamic impact fees are active | false (currently) |
| `impact_floor_bps` | Minimum impact fee when enabled | 5-50 bps |
| `max_total_fee_bps` | User protection cap for total fees | 1000-5000 bps |

**Fee Distribution Parameters** (configurable for optimization)

These parameters determine how collected fees are allocated between different stakeholders:

| Parameter | Description | Current Default | Range | Adjustable |
|-----------|-------------|----------------|-------|------------|
| `buffer_share_pct` | Percentage to Buffer for automatic POMM deployment | ~98.5% | Remainder after protocol+creator | ✓ |
| `treasury_share_pct` | Percentage to protocol treasury | ~1.0% | 0-10% | ✓ |
| `creator_share_pct` | Percentage to token creator | ~0.5% | 0-5% | ✓ |
| `pomm_funding_source` | How floor advancement is funded | Automatic fee accumulation | N/A | Fixed |

**Fee Split Optimization Goal**: One of the primary goals of this simulation is to analyze different fee allocation strategies and recommend optimal parameters for the protocol. The current program defaults (1% protocol, 0.5% creator, 98.5% buffer) serve as the baseline, with systematic exploration of alternative splits to balance floor advancement speed, protocol sustainability, and ecosystem incentives.

**POMM Deployment Parameters**

These parameters control when and how Protocol-Owned Market Making positions are deployed:

| Parameter | Description | Range/Value |
|-----------|-------------|-------------|
| `pomm_threshold_tokens` | Buffer threshold for activation | 100 tokens |
| `pomm_cooldown_seconds` | Time between deployments | 60 seconds |
| `pomm_deployment_ratio` | Fraction of buffer used per deployment | 0.1-0.8 |
| `floor_buffer_ticks` | Distance below current price for floor placement | 10-100 ticks |

**Supplemental Funding Parameters (optional)**

These settings define optional policy-driven top-ups that can be layered on top of automatic Buffer and mint flows:

| Parameter | Description | Baseline Range |
|-----------|-------------|----------------|
| `treasury_allocation_frequency` | Interval between discretionary top-ups | 1-4 weeks |
| `treasury_allocation_pct` | Portion of treasury balance earmarked per top-up | 10-50% |
| `emergency_allocation_trigger` | Floor/market ratio threshold that triggers ad-hoc funding | 0.2-0.6 |
| `governance_execution_delay_hours` | Delay between decision and capital availability | 0-72 hours |

**Liquidity Representation Parameters**

| Parameter | Description | Range/Value |
|-----------|-------------|-------------|
| `tick_bucket_spacing` | Granularity of liquidity buckets used for swap execution | 1-10 ticks |
| `liquidity_refresh_interval_minutes` | Frequency of sampling full liquidity curve snapshots | 1-60 minutes |
| `max_simulated_liquidity_range` | Tick window tracked around the current price | ±500-2000 ticks |

*Implementation note*: Scenario presets and calibration parameter files live under `experiments/configs/`. The CLI (`scripts/run_simulation.py`) loads these YAML/JSON files and can override individual fields via command-line flags or environment variables to keep experimentation reproducible without adding new modules.

---

## Simulation Parameters and Scenarios

### Base Case Configuration

The simulation requires a well-defined base case that represents realistic market conditions and serves as a reference point for parameter exploration.

**Market Environment Baseline**

The baseline market conditions used as the reference point for all simulations:

| Parameter | Value | Description |
|-----------|-------|-------------|
| SOL/USD price | $100 | Moderate valuation level |
| Daily volatility | 5% | Typical crypto volatility |
| Market trend | Neutral (zero drift) | No directional bias |
| JitoSOL yield | 7% APY | Conservative estimate |

**Protocol Parameter Baseline**

The default protocol configuration representing current implementation state:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base fee | 30 basis points (0.30%) | Standard swap fee |
| Impact fees | Disabled | Current implementation state |
| Fee distribution | 98.5% Buffer, 1% protocol, 0.5% creator | Current program defaults |
| POMM funding | Automatic Buffer accumulation | Continuous deployment |
| Mint rate | 7% synthetic FeelsSOL yield | Backing-driven drift |
| POMM deployment ratio | 50% of buffer per deployment | Conservative deployment |

**Market Participant Baseline**

The assumed composition and activity levels of market participants:

| Metric | Value | Description |
|--------|-------|-------------|
| Participant mix | 60% retail, 25% algo, 10% LPs, 5% arbitrageurs | Representative distribution |
| Daily volume | 10,000-50,000 FeelsSOL | Expected trading activity |
| Average trade size | 100 FeelsSOL | Geometric mean across all participants |

### Parameter Exploration Space

The simulation systematically explores the parameter space to identify optimal configurations across different dimensions.

**Fee Structure Exploration**

The range of fee structures to be tested for optimization:

| Parameter | Range/Options | Purpose |
|-----------|---------------|----------|
| Base fee range | 10-100 basis points (0.10%-1.00%) | Find optimal fee level |

**Fee Distribution Alternatives**

Different fee allocation scenarios to test impact on system dynamics:

| Scenario | Buffer Share | Protocol Share | Creator Share | Description |
|----------|--------------|----------------|---------------|-------------|
| Current Default | 98.5% | 1.0% | 0.5% | Current program defaults |
| Protocol Sustainable | 85% | 10% | 5% | Increased protocol treasury funding |
| Creator Incentive | 90% | 5% | 5% | Enhanced creator rewards while maintaining protocol funding |
| Balanced Growth | 92% | 5% | 3% | Moderate increases to protocol and creator shares |
| Maximum Protocol | 80% | 15% | 5% | Maximum sustainable protocol funding |

*Implementation note*: Buffer allocation directly controls floor advancement rate, while the synthetic mint rate adds a predictable drift. Higher Buffer allocation leads to faster floor growth but reduces protocol treasury accumulation. Each scenario should be evaluated for long-term sustainability.

**POMM Funding Scenarios**

Different approaches to funding floor price advancement:

| Scenario | Description | Funding Source |
|----------|-------------|----------------|
| Fee + Mint (Primary) | Automatic Buffer accumulation plus 7% mint drift | Fee share + synthetic mint |
| Yield-enhanced | Variable JitoSOL/SOL spread scenarios | Backing asset yield |
| Treasury-boosted | Additional protocol treasury allocation | Protocol treasury surplus |
| External capital | One-off or scheduled third-party funding | Governance / partnerships |

### Supplemental Funding Cadence Scenarios

| Scenario | Allocation Frequency | Execution Delay | Trigger Logic |
|----------|----------------------|-----------------|----------------|
| Monthly baseline | Every 4 weeks | 24 hours | Fixed percentage of treasury growth |
| Weekly responsive | Every 7 days | 12 hours | Allocation scales with volatility and volume |
| Emergency circuit | Ad-hoc | 1 hour | Fires when floor/market ratio drops below threshold |

**POMM Configuration Testing**

Parameters for testing different POMM deployment strategies:

| Parameter | Range | Purpose |
|-----------|-------|----------|
| Funding allocation rate | Variable frequency | Test governance funding patterns |
| Deployment ratio | 20%-80% of allocated funding | Optimize capital efficiency |
| Cooldown period | 30-300 seconds | Balance responsiveness vs stability |
| Floor buffer distance | 10-100 ticks below current price | Optimize floor placement |
| Funding sustainability | Allocation vs consumption balance | Ensure long-term viability |

**Market Scenario Matrix**

Different market conditions to test protocol robustness:

| Scenario | Annual Growth | Volatility | Description |
|----------|---------------|------------|-------------|
| Bull market | +20% | 4% | Strong upward trend, low volatility |
| Bear market | -20% | 7% | Downward trend, higher volatility |
| Sideways market | 0% | 6% | No trend, moderate volatility |
| High volatility | 0% | 10% | No trend, very high volatility |

### Stress Testing Scenarios

Robust parameter optimization requires testing system performance under extreme conditions that may occur infrequently but have significant impact.

**Market Crash Simulation**

Extreme downside scenario to test floor price protection:

| Parameter | Value | Purpose |
|-----------|-------|----------|
| Price drop | 50% over 1 week | Test severe market stress |
| Daily volatility | 15% during crash | Model panic conditions |
| Volume increase | 3x normal levels | Model exit rush behavior |
| Test objective | Floor price protection effectiveness | Validate downside protection |

**Yield Rate Shock**

Yield environment changes that affect the underlying economics:

| Scenario | Yield Rate | Driver | Impact |
|----------|------------|--------|---------|
| Yield drop | 3% APY | Regulatory changes | Test floor funding sustainability |
| Yield spike | 12% APY | Favorable conditions | Test system response to windfalls |
| Test focus | Floor price growth expectations | Participant behavior changes | Economic model validation |

**Liquidity Crisis**

Liquidity shortage scenario to test system resilience:

| Parameter | Value | Purpose |
|-----------|-------|----------|
| LP participation drop | 80% reduction | Model severe liquidity shortage |
| Trading concentration | Narrow price ranges | Test concentrated activity impact |
| Test objective | POMM performance with reduced organic liquidity | Validate protocol backstop mechanisms |

**Parameter Gaming Scenarios**

Adversarial testing to validate manipulation resistance:

| Attack Vector | Description | Defense Test |
|---------------|-------------|---------------|
| Floor timing manipulation | Coordinated attempts to manipulate advancement timing | Rate limiting and TWAP effectiveness |
| POMM gaming | Strategic trading around deployment periods | Deployment algorithm robustness |
| System manipulation | Sophisticated coordination attacks | Overall system security validation |

### Calibration Data Inputs

| Source | Usage |
|--------|-------|
| Solana CLMM historical swaps/liquidity | Calibrate trade size distributions and tick utilization |
| Partner LP & token team surveys | Parameterize acceptable fee splits and governance cadence |
| SOL & JitoSOL price/yield history | Validate market environment and yield assumptions |
| Post-launch governance + treasury logs | Recalibrate funding throughput and execution delays |

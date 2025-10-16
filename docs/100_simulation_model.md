# Feels Floor Price Simulation Model

## Overview

This document outlines the theoretical foundation and mathematical models for simulating floor price dynamics in Feels markets. The simulation model enables systematic exploration of how protocol parameters affect floor price trajectories under various market conditions and participant behaviors. The primary goal is to optimize protocol parameters such as swap fees and fee splits. This optimization aims to maximize floor price growth while maintaining healthy market dynamics.

The simulation focuses on modeling a single Feels market over time. It captures the interaction between trading activity, liquidity provider behavior, governance decisions, and floor price advancement. By systematically varying protocol parameters and market conditions, we can identify optimal configurations. These configurations balance multiple objectives including floor price growth, market efficiency, and participant satisfaction.

## Key Modeling Considerations

### Critical Architecture Insights

Based on analysis of the Feels protocol implementation, several key insights fundamentally shape the simulation approach:

**Fee Distribution Reality**: Contrary to some documentation, ~98.5% of swap fees go directly to liquidity providers. They go through Uniswap V3-style position fee accrual, not to a protocol buffer. Only ~1% goes to protocol treasury and ~0.5% to token creators.

**POMM Funding Independence**: Floor price advancement through POMM deployment is not automatically funded by trading fees. It requires explicit governance allocation decisions, protocol treasury funding, or JitoSOL yield allocation.

**LP-Centric Economics**: Since LPs receive the vast majority of fees, protocol parameter changes primarily affect LP profitability and liquidity provision incentives. They do not directly impact floor funding.

### Simulation Design Implications

**Governance Modeling**: The simulation must model POMM funding as discrete governance decisions rather than continuous fee accumulation. This requires modeling different funding strategies (yield-based, treasury-based, governance-based). It also requires modeling their sustainability.

**LP Behavior Emphasis**: With LPs receiving ~98.5% of fees, their behavioral responses to parameter changes become the primary driver of market dynamics. Fee sensitivity, capital allocation, and retention modeling are critical.

**Floor Independence**: Floor price growth operates largely independently of trading activity. This makes it essential to model alternative funding sources and their impact on long-term sustainability.

**Multi-Objective Optimization**: The optimization problem balances LP profitability (to maintain liquidity), protocol sustainability (to fund operations), and floor advancement (to provide token value support). It does not simply maximize fee-driven floor growth.

---

## Core Simulation Model

### Discrete Event Architecture

The simulation runs on a multi-scale loop. Minute-level substeps align with the protocol’s 60-second POMM cooldown, while higher-level checkpoints (hourly, daily) handle slower governance and reporting flows. This structure preserves computational efficiency without aliasing POMM opportunities.

At each one-minute tick the engine performs:

1. **Market Environment Update**: Apply short-term price shocks and sentiment shifts
2. **Participant Decision Making**: Generate orders based on updated conditions
3. **Swap Execution**: Match orders against the current liquidity curve
4. **Fee Distribution**: Accrue LP fees, protocol share, and creator share
5. **POMM Evaluation**: Check cooldown timers, TWAP windows, and funding availability
6. **Floor Funding Flow Update**: Accumulate staking yield and process queued allocations

Every 60 minutes the simulation aggregates metrics, re-evaluates governance policies, and recalibrates participant portfolios. Additional daily or weekly passes can layer on top for low-frequency processes without disrupting the minute cadence.

### State Management

The simulation maintains comprehensive state information. This information evolves over time through participant interactions and protocol mechanisms.

**Market State**
```python
class MarketState:
    current_tick: int              # Current price tick
    sqrt_price: float              # Current sqrt price (Q64 format)
    total_liquidity: float         # Active liquidity within the trading range
    liquidity_curve: Dict[int, float]  # Tick-indexed liquidity buckets
    floor_tick: int                # Current floor price tick
    floor_price_usd: float         # Floor price in USD terms
```

**Protocol Operations State**
```python
class ProtocolState:
    treasury_balance: float        # Protocol treasury accumulation (~1% of fees)
    creator_balance: float         # Creator fee accumulation (~0.5% of fees)
    staking_yield_buffer: float    # Accrued JitoSOL yield pending allocation
    governance_queue: List[GovernanceAction]  # Scheduled funding decisions
    pomm_funding_allocated: float  # Governance-earmarked capital ready for deployment
    last_pomm_deployment: int      # Timestamp of last POMM action
    pomm_deployments_count: int    # Number of floor advancements
```

**Token Economics**
```python
class TokenState:
    circulating_supply: float      # Tokens in circulation
    total_supply: float            # Total token supply (1B)
    market_cap_usd: float          # Current market capitalization
    floor_market_cap_ratio: float  # Floor price as % of market price
```

### Event Processing Logic

Each simulation step processes events in a deterministic order to ensure consistent and reproducible results. The event processing pipeline handles concurrent actions appropriately and maintains system invariants such as value conservation and monotonic floor advancement.

**Trade Processing**: When participants decide to trade, the system calculates price impact, executes swaps across liquidity positions, collects fees, and updates market state. The concentrated liquidity model requires iterating through tick ranges to determine the exact execution path and final prices.

**POMM Deployment Logic**: At each time step, the system evaluates whether conditions are met for floor advancement. This includes checking allocated POMM funding, cooldown periods, and the monotonic advancement requirement. When deployment occurs, the system selects optimal tick ranges and deploys allocated FeelsSOL as permanent liquidity. *Critical: POMM funding comes from governance allocations, not automatic fee accumulation.*

---

## Floor Price Calculation

### Core Floor Price Formula

The fundamental floor price calculation follows the formula established in the Feels architecture:

$$\text{Floor Price (in FeelsSOL)} = \frac{\text{Total Allocated FeelsSOL Reserves}}{\text{Circulating Token Supply}}$$

**USD Conversion**:
$$\text{Floor Price (USD)} = \text{Floor Price (FeelsSOL)} \times \frac{\text{FeelsSOL}}{\text{JitoSOL}} \times \frac{\text{JitoSOL}}{\text{SOL}} \times \text{SOL/USD Price}$$

Given our simulation assumptions:
- FeelsSOL/JitoSOL = 1.0 (always)
- JitoSOL/SOL = 1.0 (per simulation requirements)

Therefore: $$\text{Floor Price (USD)} = \text{Floor Price (FeelsSOL)} \times \text{SOL/USD Price}$$

### Reserve Accumulation Dynamics

**Critical Update**: Floor price reserves do not automatically accumulate from trading fees. The implementation shows that ~98.5% of fees go directly to LPs through position fee accrual, not to a protocol buffer for floor advancement.

**Actual Reserve Sources**:
1. **Governance Allocation**: Dedicated protocol funding for POMM deployment
2. **Protocol Treasury**: Small portion (~1%) that could be allocated to floor support
3. **Yield Allocation**: JitoSOL staking yield could be directed to floor reserves
4. **External Funding**: Additional capital allocation decisions

**Yield-Based Growth (when allocated)**:
$$\frac{d\text{Reserves}}{dt}_{\text{yield}} = \text{Allocated Reserves} \times \frac{\text{JitoSOL Yield Rate}}{365.25 \times 24}$$

**Governance-Based Allocation**:
$$\frac{d\text{Reserves}}{dt}_{\text{governance}} = \text{Periodic allocation decisions (not fee-driven)}$$

**Total Reserve Growth**:
$$\frac{d\text{Reserves}}{dt}_{\text{total}} = \frac{d\text{Reserves}}{dt}_{\text{yield}} + \frac{d\text{Reserves}}{dt}_{\text{governance}}$$

**Simulation Implication**: Floor price growth depends on explicit funding decisions, not automatic fee accumulation. This fundamentally changes the economic model.

### Floor Funding Pipeline

To preserve economic realism, the simulation models the full flow of capital before it becomes deployable floor liquidity:

1. **Fee Inflows**: Protocol and creator shares accrue to `treasury_balance`. LP receipts stay untouched unless a scenario explicitly authorizes redistribution.
2. **Staking Yield Accrual**: JitoSOL yield compounds into `staking_yield_buffer` using the configured compounding frequency.
3. **Governance Allocation**: DAO policies pop actions from `governance_queue`, earmarking treasury or yield balances for POMM usage and recording the decision timestamp.
4. **Deployment Readiness**: Cleared allocations move into `pomm_funding_allocated`, where cooldown timers and liquidity requirements determine actual deployment.

The simulator records each stage separately, enabling analysis of governance latency, funding sufficiency, and the sustainability of recurring allocations.

```python
@dataclass
class GovernanceAction:
    activation_time: int
    allocation_amount: float
    source: FundingSource  # e.g., TREASURY, YIELD, EXTERNAL
    destination: str       # Target pool or market identifier
```

### POMM Deployment Impact

When POMM deployment occurs, a portion of allocated reserves converts to permanent floor liquidity. This creates a ratcheting effect where floor prices can only advance upward.

### Floor & Accrual Reporting Horizons

The primary analysis objective is to understand how the floor evolves over multiple horizons and how much value accrues to each participant class. The simulation captures:

- **Daily** snapshots: end-of-day floor price, protocol treasury change, LP fee accruals, and creator receipts.
- **Weekly** rollups: seven-day averages and net changes, highlighting medium-term trends in floor support and funding.
- **Annualized** summaries: extrapolated floor growth and cumulative accruals to evaluate long-run sustainability.

These aggregates are computed from the minute-level engine outputs so that the same underlying mechanics inform short- and long-term reporting.

**Deployment Calculation**:
```python
def calculate_pomm_deployment(allocated_funding, current_floor_tick, current_price_tick, floor_buffer_ticks):
    candidate_floor_tick = current_price_tick - floor_buffer_ticks
    
    if candidate_floor_tick > current_floor_tick and allocated_funding > pomm_threshold:
        deployment_amount = min(
            allocated_funding * pomm_deployment_ratio,
            calculate_liquidity_required(candidate_floor_tick)
        )
        return deployment_amount, candidate_floor_tick
    
    return 0, current_floor_tick
```

**Floor Price Update After Deployment**:
$$\text{New Floor Reserves} = \text{Old Floor Reserves} + \text{Deployment Amount}$$

$$\text{New Floor Price} = \frac{\text{New Floor Reserves}}{\text{Circulating Supply}}$$

### Time-Dependent Floor Price Trajectory

The floor price trajectory combines deterministic yield growth with governance-driven funding allocation:

$$\text{Floor Price}(t) = \frac{\text{Initial Reserves} \times e^{r \times t} + \int_0^t \text{governance\_allocations}(\tau) \times e^{r \times (t-\tau)} d\tau}{\text{Supply}(t)}$$

where $r$ is the yield rate.

**Key Change**: Fee accumulation is replaced by governance allocations, making floor price growth dependent on explicit funding decisions rather than automatic trading fee capture. This creates a more controlled but potentially less predictable growth pattern.

---

## Market Environment Simulation

### Price Movement Modeling

The broader market environment significantly influences trading behavior and system performance. The simulation models SOL/USD price movements using a geometric Brownian motion with drift, calibrated to historical cryptocurrency volatility patterns.

**Price Evolution Model**:
```python
def update_sol_price(current_price, dt, volatility, trend_bias):
    drift = trend_bias * 0.001  # Convert trend to daily drift
    random_shock = np.random.normal(0, volatility * np.sqrt(dt))
    
    log_return = drift * dt + random_shock
    new_price = current_price * np.exp(log_return)
    
    return new_price
```

**Market Sentiment Integration**: The model incorporates broader market sentiment that affects participant risk appetite and trading frequency. Bull markets increase trading volume and attract more speculative activity, while bear markets reduce participation and increase price sensitivity.

**Volatility Regimes**: The simulation can model different volatility regimes that affect both SOL price movements and participant behavior. High volatility periods increase trading opportunities but also risk aversion, creating complex dynamics in trading patterns.

### Correlation with Trading Activity

Market environment changes influence Feels market activity through several channels that the simulation models explicitly.

**Volume Correlation**: Rising SOL prices tend to increase trading volume in token markets as participants become more active and optimistic. The simulation models this relationship through a volume multiplier that scales with recent price performance.

**Risk Appetite Changes**: Market sentiment affects participant willingness to trade and hold positions. During uncertain periods, participants may reduce position sizes and trading frequency, directly impacting fee generation.

**Arbitrage Opportunities**: SOL price movements create arbitrage opportunities that sophisticated traders exploit, generating additional trading volume and fees. The simulation includes specialized arbitrage participant classes that respond to price discrepancies.

### External Shock Events

The simulation can incorporate discrete shock events that test system resilience and floor price protection during extreme market conditions.

**Market Crash Scenarios**: Sudden SOL price drops test whether floor prices provide meaningful protection. These events increase trading volume as participants rush to exit positions, potentially accelerating floor advancement through increased fee collection.

**Yield Rate Changes**: Modifications to JitoSOL staking yields affect the background growth rate of floor prices. The simulation can model yield rate changes to understand their impact on long-term floor price trajectories.

---

## Participant Behavior Modeling

### Participant Type Classification

The simulation models distinct participant types with different behavioral patterns, risk tolerances, and trading objectives. This heterogeneous approach captures the diversity of real market participants and their varied responses to market conditions.

**Retail Traders** represent the largest group by count but smaller individual trade sizes. They exhibit higher price sensitivity, tend to follow trends, and often trade based on sentiment rather than sophisticated analysis. Their behavior includes:
- Trade size distribution: Log-normal with mean 10-100 FeelsSOL
- Frequency: 1-5 trades per day during active periods
- Price sensitivity: High (will wait for better prices)
- Sentiment correlation: Strong positive correlation with recent price performance

**Institutional Traders** execute larger trades with more sophisticated strategies. They provide market efficiency through arbitrage and may adjust strategies based on protocol parameters:
- Trade size distribution: Log-normal with mean 100-10,000 FeelsSOL
- Frequency: 10-50 trades per day with irregular clustering
- Price sensitivity: Lower (willing to pay for immediacy)
- Strategy adaptation: Respond to fee changes and market structure

**Liquidity Providers** add capital to earn fees through position fee accrual (~98.5% of swap fees). They must consider impermanent loss and fee collection rates. Their participation directly affects market efficiency and trading spreads:
- Position size distribution: Based on capital allocation models and fee earning expectations
- Duration: Range from hours to months depending on returns and fee accrual rates
- Fee sensitivity: High sensitivity to actual fee rates (since they receive most fees)
- Risk management: Active rebalancing during volatile periods
- Collection behavior: Frequency of calling `collect_fees` to realize earned fees

**Arbitrageurs** exploit price discrepancies between Feels markets and external venues. Their activity increases trading volume and improves price efficiency:
- Detection threshold: Minimum profit margins accounting for fees
- Execution speed: Near-instantaneous when opportunities arise
- Trade sizing: Based on available arbitrage profit
- Market impact awareness: Sophisticated models of price impact costs

### Behavioral Response Functions

Each participant type responds to changing conditions through parameterized behavioral functions that capture realistic decision-making patterns.

**Trading Frequency Response**:
```python
def calculate_trading_frequency(base_frequency, sol_price_change, volatility, sentiment):
    volatility_multiplier = 1 + (volatility - baseline_volatility) * volatility_sensitivity
    sentiment_multiplier = 1 + sentiment * sentiment_sensitivity
    trend_multiplier = 1 + abs(sol_price_change) * trend_sensitivity
    
    return base_frequency * volatility_multiplier * sentiment_multiplier * trend_multiplier
```

**Trade Size Distribution**:
```python
def generate_trade_size(participant_type, market_conditions):
    base_size = participant_size_parameters[participant_type]['mean']
    variance = participant_size_parameters[participant_type]['variance']
    
    market_size_modifier = 1 + market_conditions['momentum'] * size_sensitivity
    
    return np.random.lognormal(
        np.log(base_size * market_size_modifier),
        variance
    )
```

**Fee Sensitivity Modeling**: Different participant types exhibit varying sensitivity to fee changes. Retail traders may be highly sensitive to percentage fees, while institutional traders focus more on absolute costs. The simulation captures these differences through elasticity parameters that determine how trading volume responds to fee changes.

### Preference Distribution Simulation

The simulation models the distribution of participant preferences across key dimensions that affect trading behavior and system dynamics.

**Risk Tolerance Distribution**: Participants exhibit varying risk tolerance that affects their willingness to trade during volatile periods and their sensitivity to slippage and fees. The simulation models this as a beta distribution that can be calibrated to different market conditions.

**Time Horizon Preferences**: Trading strategies vary significantly in time horizon, from high-frequency arbitrage to long-term position building. These preferences affect how participants respond to market movements and protocol changes.

**Price Sensitivity Curves**: Individual participants have different price sensitivity functions that determine their willingness to trade at various fee levels and market impact costs. The simulation aggregates these individual curves to predict overall market response to parameter changes.

**Learning and Adaptation**: Sophisticated participants adapt their strategies over time based on observed market conditions and protocol behavior. The simulation includes learning mechanisms that allow participant behavior to evolve, capturing the dynamic nature of real markets.

### Empirical Calibration Roadmap

Behavioral parameters and liquidity responses must be grounded in observable data. The pre-launch calibration plan includes:

- **Comparable Market Benchmarks**: Collect trade and liquidity distributions from live Solana CLMMs (Orca Whirlpools, Raydium Concentrated) to seed priors for order sizes, tick utilization, and LP churn.
- **Partner Surveys & Commitments**: Work with planned launch partners to gather acceptable fee split bands, expected capital allocation cadence, and governance response times.
- **Historical Backtests**: Replay historical SOL price paths and Uniswap v3 fee data through the engine, tuning elasticity coefficients until simulated volume and fee capture correlate with real-world series.
- **Live Data Feedback Loop**: Maintain a calibration notebook that compares simulated outputs to post-launch metrics, with versioned parameter sets and automated alerting when deviations exceed tolerance.

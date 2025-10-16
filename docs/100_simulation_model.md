# Feels Floor Price Simulation Model

## Overview

This document outlines the theoretical foundation and mathematical models for simulating floor price dynamics in Feels markets. The simulation model enables systematic exploration of how protocol parameters affect floor price trajectories under various market conditions and participant behaviors. The primary goal is to optimize protocol parameters such as swap fees and fee splits. This optimization aims to maximize floor price growth while maintaining healthy market dynamics.

The simulation focuses on modeling a single Feels market over time. It captures the interaction between trading activity, liquidity provider behavior, protocol policy choices (fee splits, supplements), and floor price advancement. By systematically varying protocol parameters and market conditions, we can identify optimal configurations. These configurations balance multiple objectives including floor price growth, market efficiency, and participant satisfaction.

## Key Modeling Considerations

### Critical Architecture Insights

Based on analysis of the Feels protocol implementation, several key insights fundamentally shape the simulation approach:

**Fee Distribution Reality**: Swap fees are split with ~85% going to the Buffer (τ) for automatic POMM deployment, ~10% to protocol treasury, and ~5% to token creators. This creates direct fee-driven floor advancement.

**Automatic POMM Funding**: Floor price advancement through POMM deployment is automatically funded by trading fees accumulating in the Buffer. When thresholds are met (100+ tokens), POMM positions deploy automatically.

**Fee-Driven Economics**: Since the majority of fees flow to the Buffer for floor advancement, trading activity directly drives floor price growth. Higher trading volume leads to faster floor advancement.

### Simulation Design Implications

**Automatic POMM Modeling**: The simulation must model POMM funding as automatic fee accumulation with threshold-based deployment. When Buffer reaches 100+ tokens and cooldown periods are satisfied, POMM deployment occurs automatically.

**Fee-Volume Feedback Loop**: Trading volume drives fee accumulation, which drives floor advancement, which attracts more participants and trading. This creates a positive feedback loop that must be modeled carefully.

**Floor-Trading Coupling**: Floor price growth operates directly from trading activity through the Buffer mechanism. Higher trading volume leads to faster Buffer accumulation and more frequent floor advancements.

**Multi-Objective Optimization**: The optimization problem balances trading volume incentives (through fee levels), protocol sustainability (through treasury allocation), and floor advancement speed (through Buffer allocation). All are interconnected through the fee structure.

---

## Core Simulation Model

### Discrete Event Architecture

The simulation runs on a multi-scale loop. Minute-level substeps align with the protocol’s 60-second POMM cooldown, while higher-level checkpoints (hourly, daily) handle slower policy adjustments and reporting flows. This structure preserves computational efficiency without aliasing POMM opportunities.

At each one-minute tick the engine performs:

1. **Market Environment Update**: Apply short-term price shocks and sentiment shifts
2. **Participant Decision Making**: Generate orders based on updated conditions
3. **Swap Execution**: Match orders against the current liquidity curve
4. **Fee Routing**: Send the Buffer share to `buffer_balance` and credit treasury/creator balances
5. **Funding Flow Update**: Accrue synthetic minting into `mintable_feelssol` and consolidate deployable capital
6. **POMM Evaluation**: Check cooldown timers, TWAP windows, and funding availability for automatic deployment

Every 60 minutes the simulation aggregates metrics, recalibrates participant portfolios, and processes any scenario-level policy hooks. Additional daily or weekly passes can layer on top for low-frequency processes without disrupting the minute cadence.

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
    treasury_balance: float        # Protocol treasury accumulation (~10% of fees)
    creator_balance: float         # Creator fee accumulation (~5% of fees)
    buffer_balance: float          # Fee share routed to Buffer for automatic POMM
    mintable_feelssol: float       # FeelsSOL available from yield-driven minting
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

**POMM Deployment Logic**: At each time step, the system evaluates whether conditions are met for floor advancement. This includes checking Buffer balances, cooldown periods, and the monotonic advancement requirement. When deployment occurs, the system selects optimal tick ranges and deploys accumulated Buffer and mintable FeelsSOL as permanent liquidity. *Critical: Buffer funding is automatic; discretionary supplements are optional.*

---

## Floor Price Calculation

### Core Floor Price Formula

The fundamental floor price calculation follows the formula established in the Feels architecture:

$$\text{Floor Price (in FeelsSOL)} = \frac{\text{Total Allocated FeelsSOL Reserves}}{\text{Circulating Token Supply}}$$

**USD Conversion**:
**FeelsSOL Backing Model**: FeelsSOL is a synthetic asset targeting SOL price, fully backed by JitoSOL reserves. As JitoSOL appreciates relative to SOL (~7% annually), additional FeelsSOL can be minted to maintain the SOL price target.

**USD Conversion**:
$$\text{Floor Price (USD)} = \text{Floor Price (FeelsSOL)} \times \text{FeelsSOL/SOL Price Target} \times \text{SOL/USD Price}$$

For simulation purposes:
- FeelsSOL targets SOL price (1:1 price target)
- JitoSOL backing appreciates at ~7% APR relative to SOL
- FeelsSOL can be redeemed for JitoSOL at current JitoSOL/SOL rate

Therefore: $$\text{Floor Price (USD)} = \text{Floor Price (FeelsSOL)} \times \text{SOL/USD Price}$$

### Reserve Accumulation Dynamics

**Critical Update**: Floor price reserves automatically accumulate from trading fees. The implementation shows that ~85% of fees go directly to the Buffer (τ) for automatic POMM deployment, creating fee-driven floor advancement.

**Reserve Sources**:
1. **Buffer Fee Accumulation**: ~85% of every swap fee moves directly into Buffer for deployment
2. **FeelsSOL Minting Capacity**: JitoSOL’s ~7% APR outperformance of SOL allows continual FeelsSOL minting without breaking the peg
3. **Treasury/Creator Shares**: The remaining fee shares (~10% protocol, ~5% creator) can optionally reinforce the floor or fund incentives

**Buffer Accumulation Rate**:
$$\frac{d\text{Buffer}}{dt} = \text{Trading Volume} \times \text{Average Fee Rate} \times \text{buffer\_share}$$

**FeelsSOL Minting Rate**:
$$\frac{d\text{Mintable FeelsSOL}}{dt} = \text{Outstanding FeelsSOL} \times \frac{0.07}{365.25 \times 24}$$

**Total Reserve Growth**:
$$\frac{d\text{Reserves}}{dt} = \frac{d\text{Buffer}}{dt} + \frac{d\text{Mintable FeelsSOL}}{dt} + \text{supplementary transfers}$$

**Simulation Implication**: Floor price growth follows trading volume in real time, while yield-driven minting adds a predictable upward drift.

### Floor Funding Pipeline

To preserve economic realism, the simulation models the full flow of capital before it becomes deployable floor liquidity:

1. **Fee Inflows**: Each minute, the buffer share of fees increases `buffer_balance`, while treasury and creator balances accumulate their fixed shares.
2. **Yield Drift**: The engine mints new FeelsSOL at the 7% APR drift rate, crediting `mintable_feelssol`.
3. **Automatic Deployment**: When cooldown, TWAP placement, and minimum buffer thresholds are met, the model converts Buffer + mintable supply into fresh POMM liquidity.
4. **Optional Supplements**: Scenario hooks allow treasury or external capital to top up the floor, but these are additive rather than required.

Tracking these components independently allows attribution of floor growth between trading-driven Buffer inflows, synthetic minting, and discretionary injections.

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

The floor price trajectory combines automatic fee-driven accumulation with backing appreciation:

$$\text{Floor Price}(t) = \frac{\text{Initial Reserves} + \int_0^t \text{fee\_accumulation}(\tau) d\tau + \text{Supply}(0) \times (e^{r \times t} - 1)}{\text{Supply}(t)}$$

where $r$ is the JitoSOL appreciation rate relative to SOL (~7% APR).

**Key Changes**: 
1. **Primary Growth Driver**: Automatic fee accumulation in Buffer (~85% of swap fees)
2. **Secondary Growth**: FeelsSOL minting capacity from JitoSOL backing appreciation
3. **Trading Coupling**: Floor growth rate directly proportional to trading volume and activity
4. **Feedback Loop**: Higher floors attract more trading, creating compound growth potential

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

**Liquidity Providers** add capital to maintain tight spreads. With the majority of fees funneled into the Buffer, LP incentives rely on the residual fee share plus any supplemental rewards. Their participation directly affects market efficiency and trading spreads:
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
- **Partner Surveys & Commitments**: Work with planned launch partners to gather acceptable fee split bands, supplemental funding appetite, and operational response times.
- **Historical Backtests**: Replay historical SOL price paths and Uniswap v3 fee data through the engine, tuning elasticity coefficients until simulated volume and fee capture correlate with real-world series.
- **Live Data Feedback Loop**: Maintain a calibration notebook that compares simulated outputs to post-launch metrics, with versioned parameter sets and automated alerting when deviations exceed tolerance.

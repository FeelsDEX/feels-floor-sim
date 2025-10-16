# Feels Protocol Architecture: Economic and Financial Model

## System Overview

The Feels Protocol implements a hub-and-spoke concentrated liquidity AMM on Solana. The system is designed around FeelsSOL as the universal routing token. Traditional AMMs fragment liquidity across multiple trading pairs. Feels consolidates all trading activity through a single hub token that is fully backed by yield-bearing JitoSOL.

The protocol's architecture centers on three integrated systems. First, a dual-component fee structure captures trading value. Second, a Protocol-Owned Market Making (POMM) system establishes monotonically increasing price floors. Third, a Just-In-Time (JIT) liquidity mechanism supports newly launched markets. These systems operate within a carefully designed economic model. The model balances liquidity provider incentives, protocol sustainability, and floor price advancement.

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Fee System](#fee-system)
3. [Floor Liquidity System (POMM)](#floor-liquidity-system-pomm)
4. [Just-In-Time (JIT) Liquidity System](#just-in-time-jit-liquidity-system)
5. [Fee Split and Value Flows](#fee-split-and-value-flows)
6. [Market Lifecycle and Integration](#market-lifecycle-and-integration)
7. [Economic Parameters](#economic-parameters)
8. [Mathematical Formulas](#mathematical-formulas)

---

## Core Architecture

### The Hub-and-Spoke Foundation

At its core, Feels operates on a strict hub-and-spoke model. FeelsSOL serves as the universal intermediary for all trades. Traditional AMMs create isolated liquidity pools between arbitrary token pairs. Every market in Feels is structured as a TokenX/FeelsSOL pair. This architectural constraint means that trading from TokenA to TokenB requires at most two hops: TokenA → FeelsSOL → TokenB.

FeelsSOL is a synthetic asset that targets the price of SOL while being fully backed by JitoSOL reserves. JitoSOL is a liquid staking token that continuously earns approximately 7-8% annual staking rewards. This creates a key dynamic: as JitoSOL appreciates relative to SOL (~7% annually), the protocol can mint additional FeelsSOL at a rate equivalent to this price divergence. Users can enter the system by depositing JitoSOL to mint FeelsSOL. They can exit by burning FeelsSOL to redeem JitoSOL at the current JitoSOL/SOL exchange rate.

Each market implements concentrated liquidity mechanics similar to Uniswap V3. Liquidity providers can specify price ranges for their capital. FeelsSOL is always token_0 in every pair. This means the system benefits from consistent liquidity aggregation rather than fragmentation. NFT-tokenized positions represent individual liquidity provider stakes. This enables precise fee accounting and position management.

### Dual-Layer Solvency Model

The protocol maintains solvency through a carefully designed two-layer architecture. At the pool level, each individual market maintains its own FeelsSOL escrow account. This account holds the tokens needed for swaps and liquidity operations. At the protocol level, a unified reserve of JitoSOL fully backs all FeelsSOL tokens across every market in the system.

This structure provides both isolation and systemic stability. If a particular token market experiences problems, other markets remain unaffected. Each pool manages its own FeelsSOL inventory. The protocol-level JitoSOL reserves ensure that FeelsSOL can always be redeemed for JitoSOL at the current market rate. As JitoSOL appreciates relative to SOL through staking yield, the protocol can mint additional FeelsSOL to maintain the SOL price target while preserving full backing.

The Buffer (τ - tau) serves as a thermodynamic reservoir within each market. It accumulates fees and manages the flow of value between different system components. This buffer concept is central to how Feels maintains value conservation. No value is created or destroyed within the system. Value is only transferred between participants and pools according to well-defined rules.

---

## Fee System

### Integrated Fee Structure

The fee system in Feels operates as the primary value capture mechanism. It drives the protocol's economic dynamics. Every swap incurs a total fee composed of two components. A base fee provides predictable revenue. An impact fee scales with price movement to account for the cost of liquidity provision.

**Base fees** represent a fixed percentage charged on every trade. They are currently set at 30 basis points (0.30%) by default. This fee is configurable per market through governance and can range from 0% to 10%. This allows for fine-tuning based on market characteristics and competitive dynamics. The base fee provides steady revenue that flows into the system regardless of price volatility.

**Impact fees** add a dynamic component that reflects the actual market impact of each trade. As currently implemented, impact fees are designed to start at a minimum of 10 basis points (0.10%). They increase based on the number of ticks crossed during swap execution. However, the current implementation has this feature disabled (returning 0% impact fees). This means only base fees are active until the impact fee mechanism is fully deployed.

### Fee Calculation and User Protection

The fee calculation process balances economic efficiency with user protection. Before executing a swap, the system estimates the total fees to provide users with clear expectations. During execution, the actual price movement determines the final impact fee component. This is then added to the base fee to calculate the total charge.

The system implements a fee cap mechanism to prevent excessive fees during periods of high volatility. By default, total fees cannot exceed 2,500 basis points (25%) per transaction. If a swap would trigger fees above this threshold, the transaction fails rather than proceeding with unexpected costs. This protection ensures that users maintain control over their trading costs even in extreme market conditions.

The post-execution fee calculation creates transparency and fairness. The system calculates the final fee based on the realized price movement rather than pre-trade estimates. Pre-trade estimates might not reflect actual market impact. This ensures that users pay for the liquidity they actually consume.

---

## Floor Liquidity System (POMM)

### Protocol-Owned Market Making Mechanics

The Protocol-Owned Market Making (POMM) system represents Feels' most innovative feature. It provides the ability to create and maintain monotonically increasing price floors for every token in the ecosystem. POMM achieves this by strategically deploying accumulated fees as liquidity positions that support token prices. This creates guaranteed exit liquidity that can only improve over time.

The floor price calculation follows a simple but powerful formula: 

$$\text{Floor Price} = \frac{\text{Pool's Allocated FeelsSOL Reserves}}{\text{Token Circulating Supply}}$$ As fees accumulate in the Buffer and get deployed as POMM liquidity, the numerator grows. The denominator remains constant (or may decrease if tokens are burned). This ensures that the floor price trends upward.

The "monotonic property" is fundamental to POMM's value proposition. Traditional market making allows liquidity to be withdrawn or repositioned downward. POMM liquidity is deployed strategically to ensure that once a floor is established at a certain level, it never retreats. This creates confidence for token holders and provides a foundation for sustainable tokenomics.

### Strategic Liquidity Deployment

POMM operates through sophisticated placement logic that balances market efficiency with manipulation resistance. POMM uses a 5-minute time-weighted average price (TWAP) to determine optimal placement points rather than placing liquidity at current spot prices (which could be manipulated). This approach ensures that POMM liquidity reflects genuine market conditions rather than temporary price distortions.

The system deploys POMM positions with adaptive width based on market characteristics. Each position spans approximately 20 times the market's tick spacing. This provides meaningful liquidity depth while maintaining capital efficiency. This width ensures that POMM provides real price support.

POMM operations are protected by multiple complementary security mechanisms:

- **Rate Limiting**: Prevents more than one POMM deployment per minute, reducing manipulation and abuse potential
- **Threshold Gating**: Ensures POMM only activates when sufficient Buffer fees have accumulated
- **TWAP Price Reference**: Uses 5-minute time-weighted average to resist price manipulation
- **Fee Accounting Model**: Operates on accumulated fees rather than direct vault access, preventing flash loan attacks
- **Adaptive Position Width**: Scales deployment size with market tick spacing for appropriate liquidity depth

### Floor Advancement Algorithm

The floor advancement process follows a systematic approach that balances responsiveness with stability. The system continuously calculates a "candidate floor" by subtracting a buffer distance from the current market price. When this candidate floor exceeds the existing floor price, conditions are met for potential advancement.

An increase in floor level requires several conditions to align:

1. **Monotonic Improvement**: The candidate floor must represent a genuine improvement over the current floor (never retreating)
2. **Sufficient Buffer**: Adequate fees must have accumulated in the Buffer to fund the new liquidity position
3. **Rate Limit Compliance**: The 60-second cooldown between POMM deployments must have expired
4. **Appropriate Market Phase**: The market must be in a phase that permits POMM activity
5. **Reserve Requirements**: Sufficient operational reserves must remain after deployment

When advancement occurs, the system consumes FeelsSOL from the Buffer to establish new liquidity positions at the higher floor level. The amount deployed depends on the desired liquidity depth and the available buffer balance. Safeguards ensure that minimum operational reserves remain available.

---

## Just-In-Time (JIT) Liquidity System

### Virtual Concentration for New Markets

The Just-In-Time liquidity system addresses the cold start problem that new token markets face. It provides immediate, protocol-owned liquidity around the current trading price. JIT v0.5 implements a "virtual concentration" mechanism that effectively multiplies the impact of deployed liquidity. This creates tight spreads without requiring massive capital deployment.

When active, JIT provides a 10x liquidity multiplier for any capital within one tick of the current market price. This virtual concentration simulates the effect of having concentrated liquidity providers actively managing positions around the current price. This ensures that small trades experience minimal slippage even in newly launched markets.

**JIT is disabled by default** in the current implementation. Markets initialize with `jit_enabled = false` and zero multipliers. This means this system only becomes active when explicitly enabled for specific markets. This represents a conservative approach to rolling out JIT functionality.

### Activation Conditions and Safety Mechanisms

When enabled, JIT operates under strict conditions designed to prevent abuse while maximizing utility. The system activates primarily during the BondingCurve and early SteadyState phases. During these phases, organic liquidity provider activity may be insufficient to support smooth trading. As markets mature and attract sufficient LP liquidity, JIT can be disabled to reduce protocol exposure.

Rate limiting mechanisms prevent excessive JIT usage that could destabilize markets or create arbitrage opportunities. Per-slot caps limit how much JIT liquidity can be consumed in any single Solana slot. Rolling consumption windows track usage across multiple slots to prevent sustained abuse.

Anti-gaming protections include several mechanisms. Circuit breakers trigger when unusual trading patterns are detected. Sophisticated monitoring tracks JIT consumption rates. The system can quickly disable JIT for any market showing signs of manipulation or instability.

### Economic Integration

JIT operates as a fee-earning component of the protocol. It captures a portion of swap fees when its liquidity is utilized. This compensation structure ensures that JIT provides positive returns to the protocol while supporting market efficiency. The risk-adjusted returns account for the temporary nature of JIT positions and the potential for adverse selection in volatile markets.

The fee structure for JIT integrates with the broader revenue distribution system. It contributes to Buffer accumulation while providing direct returns for the liquidity provision service. This creates alignment between JIT operations and the overall economic health of the protocol.

---

## Fee Split and Value Flows

### Revenue Distribution Architecture

The protocol implements a **two-layer fee distribution system** that combines traditional fee splitting with Uniswap V3-style position fee accrual:

#### Primary Fee Split (from total swap fees)

| Component | Current Default | Configurable Range | Purpose |
|-----------|----------------|-------------------|---------|
| Buffer (τ) | ~98.5% | Variable remainder | Accumulates for automatic POMM deployment and floor advancement |
| Protocol Treasury | ~1.0% | 0-10% | Ongoing development and operational expenses |
| Token Creators | ~0.5% | 0-5% | Incentive for launching projects on the platform |

**Note**: Current program defaults allocate 1% to protocol treasury and 0.5% to creators, with the buffer receiving the remainder (~98.5%). These percentages are configurable by the protocol admin and may be adjusted based on simulation results and governance decisions.

#### Automatic Fee-Driven Floor Advancement

**Critical**: The Buffer (τ) receives the majority of swap fees and automatically funds POMM deployment when thresholds are met. This creates automatic, fee-driven floor price advancement.

**How the System Works**:
1. **Fee Collection**: Each swap splits fees according to protocol-configured allocation
2. **Buffer Accumulation**: Currently ~98.5% of fees flow into the market's Buffer (τ)
3. **Automatic POMM Deployment**: When Buffer reaches threshold (100 tokens), POMM deployment becomes eligible
4. **Floor Advancement**: POMM positions are deployed automatically, advancing the floor price
5. **Monotonic Growth**: Floor prices can only increase, never decrease

**Fee-Driven Growth Formula**:

$$\text{Buffer Growth Rate} = \text{Trading Volume} \times \text{Average Fee Rate} \times \text{Buffer Share}$$

$$\text{Floor Advancement Rate} \propto \text{Buffer Accumulation Rate}$$

This system ensures that active trading directly translates to floor price advancement, creating strong incentives for ecosystem growth.

**Fee Split Optimization**: The current defaults (1% protocol, 0.5% creator, 98.5% buffer) can be adjusted by the protocol admin. This simulation framework enables analysis of different fee allocation strategies to optimize floor advancement speed, protocol sustainability, and ecosystem incentives.

### Buffer Operations and Value Flows

**Important**: The Buffer (τ) is the primary accumulator of swap fees and the automatic funding source for POMM deployment.

The Buffer (τ) serves as the **automatic floor advancement engine** that drives the protocol's core value proposition:

**Buffer Functions**:
- **Primary Fee Collection**: Accumulates majority of all swap fees (currently ~98.5%)
- **Automatic POMM Funding**: Deploys accumulated fees when thresholds are met
- **Floor Price Advancement**: Creates monotonically increasing price floors
- **Thermodynamic Reservoir**: Manages fee flow and deployment timing

**Value Flows Through Buffer**:
- **Inflows**: Configurable percentage of total swap fees from all trading activity (currently ~98.5%)
- **Outflows**: Automatic POMM deployments when thresholds and cooldowns are satisfied
- **Design Purpose**: Converts trading activity directly into floor price support

**Automatic POMM Deployment Model**: The implementation shows that POMM deployment is automatic and fee-driven. When the Buffer accumulates sufficient fees (100+ tokens) and cooldown periods are satisfied, POMM positions deploy automatically. This creates direct causation between trading volume and floor price advancement.

### Integration with FeelsSOL Yield

The protocol benefits from a dual value accumulation mechanism. The majority of swap fees (~85%) flow into market-specific Buffers that automatically fund floor advancement. Meanwhile, the underlying JitoSOL backing of FeelsSOL continues to appreciate through staking rewards at approximately 7-8% annually. This creates a unique dynamic: as the backing appreciates relative to SOL, the protocol can mint additional FeelsSOL while maintaining full backing. This provides supplementary capacity beyond the primary fee-driven advancement mechanism.

Staking yield can be allocated to individual pool floors through governance decisions. This provides an additional mechanism for floor price support that operates independently of trading activity. This creates a baseline growth rate for floor prices even in markets with minimal trading volume.

The combination of fee accumulation and yield appreciation creates multiple pathways for value creation. This ensures that the protocol can sustain and grow floor prices across diverse market conditions.

---

## Market Lifecycle and Integration

### Phase-Based System Evolution

Markets in Feels progress through a carefully designed lifecycle. This lifecycle optimizes economic behavior for each stage of development. The system recognizes seven distinct phases. Each has specific characteristics that affect fee behavior, liquidity provision, and floor advancement.

**Created and BondingCurve phases** represent the initial launch period where price discovery is paramount. During these phases, markets rely heavily on bonding curve liquidity and nascent POMM support. Fee behavior may exhibit higher volatility as trading patterns establish. Floor advancement occurs conservatively to avoid premature price setting.

**SteadyState operation** represents the target condition where markets have sufficient organic liquidity, established trading patterns, and regular POMM floor advancement. During this phase, the full economic model operates as designed. This includes predictable fee generation, systematic buffer accumulation, and routine floor price improvements.

**Graduated markets** represent mature ecosystems with substantial liquidity and established communities. These markets may operate with reduced protocol intervention, lower fees, and less frequent but larger floor advancements. This occurs as they transition toward community-driven governance.

### System Component Integration

The true power of Feels emerges from how its components integrate throughout the market lifecycle. In early phases, JIT liquidity (when enabled) provides immediate trading capability. Meanwhile, fees begin accumulating in the Buffer. As trading volume grows, the Buffer reaches deployment thresholds. This triggers POMM placements that establish and advance floor prices.

This creates a positive feedback loop. Better floor prices attract more participants. This increases trading volume and generates more fees. More fees enable more POMM deployment and further advance floor prices. The hub-and-spoke architecture amplifies this effect by concentrating all trading activity. This ensures that every transaction contributes to the economic system.

The integrated nature of these systems means that parameter tuning must consider cross-component effects. Adjusting fee splits affects Buffer accumulation rates. This influences POMM deployment frequency. This impacts floor advancement speed.

---

## Economic Parameters

### Core System Constants

The protocol operates with carefully chosen constants that balance economic efficiency with practical constraints:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Standard Token Supply | 1 billion tokens | Default supply for new markets |
| Token Decimals | 6 | Standard precision level |
| Minimum Launch Requirement | 250 million tokens (25%) | Ensures meaningful initial liquidity |
| Base Fee Range | 0% - 10% (0-1000 bps) | Configurable per market via governance |
| Default Base Fee | 0.30% (30 bps) | Balances competitiveness with revenue |
| Impact Fee Floor | 0.10% (10 bps) | Currently disabled in implementation |
| Maximum Total Fee Cap | 25% (2500 bps) | Protection against extreme volatility |
| Maximum Ticks Per Swap | 200 ticks | DoS protection and compute limits |
| Minimum Liquidity Requirement | 1000 units | Prevents dust positions |

### POMM and Buffer Parameters

POMM operations follow strict timing and threshold parameters designed to prevent manipulation while ensuring responsive floor advancement:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| POMM Deployment Cooldown | 60 seconds | Prevents rapid-fire deployment and manipulation |
| Buffer Activation Threshold | 100 tokens | Triggers POMM deployment eligibility |
| POMM Position Width Multiplier | 20x tick spacing | Ensures meaningful liquidity depth |
| TWAP Window | 5 minutes | Manipulation-resistant price reference |
| Maximum Buffer Deployment | 50% of balance | Maintains operational reserves |

### Safety and Performance Limits

System-wide limits protect against denial-of-service attacks and ensure stable operation. The maximum of 200 ticks crossed per swap prevents excessively complex transactions that could consume too much compute. Minimum liquidity requirements of 1000 units prevent dust positions that would complicate accounting.

These parameters represent the current implementation state. The modular architecture allows for parameter updates through governance. This does not require fundamental system changes.

---

## Mathematical Formulas

### Core Price and Liquidity Calculations

Feels inherits the mathematical foundation of concentrated liquidity from established protocols like Uniswap V3. The relationship between ticks and prices follows this formula:

$$\text{price} = 1.0001^{\text{tick}}$$

This provides fine-grained price granularity with approximately 0.01% spacing between adjacent ticks.

Converting between sqrt prices and ticks requires logarithmic calculations:

$$\text{tick} = \frac{\log(\text{sqrt\_price})}{\log(1.0001)}$$

These conversions are fundamental to position management and fee calculations throughout the system.

Liquidity positions translate to token amounts through geometric relationships that account for the price range. For a given liquidity amount $L$ and price range from lower to upper:

$$\text{amount}_0 = L \times (\sqrt{\text{price}_{\text{upper}}} - \sqrt{\text{price}_{\text{lower}}})$$

$$\text{amount}_1 = L \times \left(\frac{1}{\sqrt{\text{price}_{\text{lower}}}} - \frac{1}{\sqrt{\text{price}_{\text{upper}}}}\right)$$

### Fee and Floor Calculations

Fee calculations combine base and impact components:

$$\text{total\_fee\_bps} = \text{base\_fee\_bps} + \max(\text{impact\_floor\_bps}, \text{actual\_impact\_bps})$$

The actual fee amount then derives from:

$$\text{fee\_amount} = \frac{\text{swap\_amount} \times \text{total\_fee\_bps}}{10000}$$

Floor price calculations follow the fundamental formula:

$$\text{floor\_price} = \frac{\text{allocated\_feelssol\_reserves}}{\text{circulating\_token\_supply}}$$

Floor advancement occurs when:

$$\text{candidate\_floor} = \text{current\_price\_tick} - \text{buffer\_ticks} > \text{current\_floor\_tick}$$

Buffer dynamics follow the relationship:

$$\text{buffer\_growth\_rate} = \text{fee\_inflow\_rate} - \text{pomm\_deployment\_rate}$$

POMM deployment decisions consider both buffer availability and liquidity requirements:

$$\text{deployment\_amount} = \min(\text{buffer\_balance} \times 0.5, \text{required\_feelssol})$$

### Economic Model Integration

These mathematical relationships integrate to create the protocol's economic model. Fee generation drives buffer accumulation. This enables POMM deployment. This advances floor prices. This attracts more trading, creating a self-reinforcing cycle.

The mathematical precision of these calculations ensures that the economic model operates predictably and fairly. It provides clearly defined relationships between trading activity, fee generation, and floor price advancement.
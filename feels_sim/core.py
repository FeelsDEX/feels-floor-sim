"""Core simulation engine and state management."""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

from .config import SimulationConfig


# Tick-to-price conversion utilities (Uniswap V3 compatible)
def tick_to_price(tick: int) -> float:
    """Convert tick to price using 1.0001^tick formula."""
    return 1.0001 ** tick  # Each tick = 0.01% price difference


def price_to_tick(price: float) -> int:
    """Convert price to tick using log formula."""
    return round(math.log(price) / math.log(1.0001))  # Inverse of tick_to_price


def tick_to_sqrt_price_x64(tick: int) -> int:
    """Convert tick to sqrt price in Q64.64 format."""
    sqrt_price = math.sqrt(tick_to_price(tick))
    return int(sqrt_price * (2 ** 64))


@dataclass
class MarketState:
    """Current market state including price and liquidity."""
    current_tick: int = 0
    sqrt_price: float = 1.0
    total_liquidity: float = 0.0
    liquidity_curve: Dict[int, float] = None
    floor_tick: int = -1000  # Start well below current price
    floor_price_feelssol: float = 0.0  # Floor price in FeelsSOL terms (monotonic)
    floor_price_usd: float = 0.0  # Floor price in USD terms (can fluctuate with SOL)
    
    def __post_init__(self):
        if self.liquidity_curve is None:
            self.liquidity_curve = {}


@dataclass
class FloorFundingState:
    """State of floor funding pipeline."""
    treasury_balance: float = 0.0
    creator_balance: float = 0.0
    buffer_balance: float = 0.0
    mintable_feelssol: float = 0.0
    deployed_feelssol: float = 0.0
    buffer_routed_cumulative: float = 0.0
    mint_cumulative: float = 0.0
    last_pomm_deployment: int = -10_000
    pomm_deployments_count: int = 0


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state at a point in time."""
    timestamp: int
    sol_price_usd: float
    floor_price_feelssol: float
    floor_price_usd: float
    floor_state: FloorFundingState
    volume_feelssol: float = 0.0
    fees_collected: float = 0.0
    events: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = {"pomm_deployed": False}


@dataclass
class SimulationResults:
    """Results from a complete simulation run."""
    snapshots: List[SimulationSnapshot]
    hourly_aggregates: List[Dict[str, float]]
    config: SimulationConfig


class FeelsSimulation:
    """Main simulation engine."""
    
    def __init__(self, config: SimulationConfig):
        config.validate()
        self.config = config
        self.market_state = MarketState()
        self.floor_state = FloorFundingState(
            buffer_balance=config.initial_buffer_balance,
            deployed_feelssol=config.initial_deployed_feelssol
        )
        
        # Initialize market state
        self.sol_price_usd = config.initial_sol_price_usd
        self.market_state.current_tick = price_to_tick(1.0)  # Start at 1 FeelsSOL = 1 USD equivalent
        self.market_state.sqrt_price = math.sqrt(1.0)
        
        # Initialize floor
        self._update_floor_price()
        
        # Simulation state
        self.current_minute = 0
        self.snapshots: List[SimulationSnapshot] = []
        self.hourly_aggregates: List[Dict[str, float]] = []
        
        # Random number generator for deterministic behavior
        self.rng = np.random.RandomState(42)
    
    def _update_floor_price(self):
        """Update floor price based on allocated reserves."""
        floor_reserves = self.floor_state.deployed_feelssol
        floor_price_feelssol = 0.0
        if self.config.circulating_supply > 0:
            floor_price_feelssol = floor_reserves / self.config.circulating_supply
        
        # Ensure monotonic floor in FeelsSOL terms
        self.market_state.floor_price_feelssol = max(
            floor_price_feelssol, self.market_state.floor_price_feelssol
        )
        # Convert to USD using current SOL price target
        self.market_state.floor_price_usd = self.market_state.floor_price_feelssol * self.sol_price_usd
    
    def step_minute(self):
        """Execute one minute of simulation."""
        # Update market environment (SOL price evolution)
        self._update_sol_price()
        
        # Generate placeholder trading activity
        volume, fees = self._generate_trading_activity()
        
        # Route fees into Buffer / treasury / creator balances
        self._route_fees(fees)
        
        # Accrue synthetic minting capacity from JitoSOL appreciation
        self._accrue_minting()
        
        # Evaluate POMM deployment
        pomm_deployed = self._evaluate_pomm_deployment()
        
        # Update floor price
        self._update_floor_price()
        
        # Record snapshot
        snapshot = SimulationSnapshot(
            timestamp=self.current_minute,
            sol_price_usd=self.sol_price_usd,
            floor_price_feelssol=self.market_state.floor_price_feelssol,
            floor_price_usd=self.market_state.floor_price_usd,
            floor_state=FloorFundingState(
                treasury_balance=self.floor_state.treasury_balance,
                creator_balance=self.floor_state.creator_balance,
                buffer_balance=self.floor_state.buffer_balance,
                mintable_feelssol=self.floor_state.mintable_feelssol,
                deployed_feelssol=self.floor_state.deployed_feelssol,
                buffer_routed_cumulative=self.floor_state.buffer_routed_cumulative,
                mint_cumulative=self.floor_state.mint_cumulative,
                last_pomm_deployment=self.floor_state.last_pomm_deployment,
                pomm_deployments_count=self.floor_state.pomm_deployments_count
            ),
            volume_feelssol=volume,
            fees_collected=fees,
            events={"pomm_deployed": pomm_deployed}
        )
        self.snapshots.append(snapshot)
        
        self.current_minute += 1
    
    def _update_sol_price(self):
        """Update SOL price using geometric Brownian motion."""
        dt = 1.0 / (365.25 * 24 * 60)  # One minute in years
        drift = self.config.sol_trend_bias * 0.001  # Convert trend to small drift
        volatility = self.config.sol_volatility_daily / math.sqrt(365.25)  # Scale daily vol to minute
        
        random_shock = self.rng.normal(0, volatility * math.sqrt(dt))  # Brownian motion
        log_return = drift * dt + random_shock  # Geometric Brownian motion formula
        
        self.sol_price_usd *= math.exp(log_return)  # Apply to current price
    
    def _accrue_minting(self):
        """Accrue synthetic FeelsSOL minting from JitoSOL appreciation."""
        dt = 1.0 / (365.25 * 24 * 60)  # One minute in years
        mint_growth = self.config.total_supply * self.config.jitosol_yield_apy * dt
        self.floor_state.mintable_feelssol += mint_growth
        self.floor_state.mint_cumulative += mint_growth

    def _generate_trading_activity(self) -> Tuple[float, float]:
        """Generate placeholder trading volume and fees."""
        # Simple volume model: base volume with some randomness
        base_volume = 1000.0  # Base daily volume in FeelsSOL
        daily_volume = base_volume * (1 + self.rng.normal(0, 0.1))  # 10% random variation
        minute_volume = daily_volume / (24 * 60)  # Convert to per-minute
        
        # Calculate fees from volume
        fee_rate = self.config.base_fee_bps / 10000.0  # Convert bps to decimal
        fees = minute_volume * fee_rate
        
        return minute_volume, fees
    
    def _route_fees(self, total_fees: float):
        """Route fees into Buffer, treasury, and creator balances."""
        buffer_fees = total_fees * (self.config.buffer_share_pct / 100.0)
        treasury_fees = total_fees * (self.config.treasury_share_pct / 100.0)
        creator_fees = total_fees * (self.config.creator_share_pct / 100.0)

        self.floor_state.buffer_balance += buffer_fees
        self.floor_state.buffer_routed_cumulative += buffer_fees
        self.floor_state.treasury_balance += treasury_fees
        self.floor_state.creator_balance += creator_fees

    def _evaluate_pomm_deployment(self) -> bool:
        """Evaluate whether POMM should deploy and advance the floor."""
        # Enforce cooldown (convert seconds to minutes)
        minutes_since_last = self.current_minute - self.floor_state.last_pomm_deployment
        cooldown_minutes = self.config.pomm_cooldown_seconds / 60
        if minutes_since_last < cooldown_minutes:
            return False

        deployable_capital = self.floor_state.buffer_balance + self.floor_state.mintable_feelssol
        if deployable_capital < self.config.pomm_threshold_tokens:
            return False

        deployment_amount = deployable_capital * self.config.pomm_deployment_ratio

        # Consume Buffer first, then synthetic minting if needed
        buffer_used = min(self.floor_state.buffer_balance, deployment_amount)
        self.floor_state.buffer_balance -= buffer_used

        remaining = deployment_amount - buffer_used
        mint_used = min(self.floor_state.mintable_feelssol, remaining)
        self.floor_state.mintable_feelssol -= mint_used

        total_deployed = buffer_used + mint_used
        if total_deployed <= 0:
            return False

        self.floor_state.deployed_feelssol += total_deployed
        self.floor_state.last_pomm_deployment = self.current_minute
        self.floor_state.pomm_deployments_count += 1
        return True
    
    def complete_hour(self):
        """Complete hourly aggregation and processing."""
        if len(self.snapshots) == 0:
            return
        
        # Get last hour of snapshots
        hour_start = max(0, len(self.snapshots) - 60)
        hour_snapshots = self.snapshots[hour_start:]
        
        # Calculate hourly aggregates
        hourly_data = {
            "hour": len(self.hourly_aggregates),
            "avg_sol_price": np.mean([s.sol_price_usd for s in hour_snapshots]),
            "avg_floor_price": np.mean([s.floor_price_usd for s in hour_snapshots]),
            "total_volume": sum(s.volume_feelssol for s in hour_snapshots),
            "total_fees": sum(s.fees_collected for s in hour_snapshots),
            "pomm_deployments": sum(1 for s in hour_snapshots if s.events["pomm_deployed"]),
            "buffer_balance": hour_snapshots[-1].floor_state.buffer_balance,
            "mintable_feelssol": hour_snapshots[-1].floor_state.mintable_feelssol,
            "treasury_balance": hour_snapshots[-1].floor_state.treasury_balance,
            "buffer_routed_cumulative": hour_snapshots[-1].floor_state.buffer_routed_cumulative,
            "mint_cumulative": hour_snapshots[-1].floor_state.mint_cumulative,
            "deployed_feelssol": hour_snapshots[-1].floor_state.deployed_feelssol
        }
        
        self.hourly_aggregates.append(hourly_data)
    
    def run(self, hours: int) -> SimulationResults:
        """Run simulation for specified number of hours."""
        total_minutes = hours * 60
        
        for minute in range(total_minutes):
            self.step_minute()
            
            # Complete hourly processing
            if (minute + 1) % 60 == 0:
                self.complete_hour()
        
        return SimulationResults(
            snapshots=self.snapshots,
            hourly_aggregates=self.hourly_aggregates,
            config=self.config
        )

"""Core data structures for Feels simulation state management."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .config import SimulationConfig


@dataclass
class MarketState:
    """Current market state including price and liquidity.
    
    Tracks the instantaneous state of the AMM including current price tick,
    liquidity distribution, and floor prices in both FeelsSOL and USD terms.
    Central to price discovery and POMM placement calculations.
    """
    current_tick: int = 0                      # Current price in tick space (0.01% per tick)
    sqrt_price: float = 1.0                    # Square root of price for AMM calculations
    total_liquidity: float = 0.0               # Total active liquidity for trading
    floor_tick: int = -1000                    # Current floor tick (starts below market)
    floor_price_feelssol: float = 0.0          # Floor price in FeelsSOL (monotonic increase)
    floor_price_usd: float = 0.0               # Floor price in USD (varies with SOL price)
    


@dataclass
class FloorFundingState:
    """State of floor funding pipeline.
    
    Tracks all funding sources for floor price advancement including fee-funded
    Buffer, protocol treasury, creator rewards, and synthetic FeelsSOL minting.
    Central to POMM deployment decisions and floor price calculations.
    """
    treasury_balance: float = 0.0               # Protocol treasury from fee routing
    creator_balance: float = 0.0                # Token creator rewards from fee routing
    buffer_balance: float = 0.0                 # Buffer for POMM deployments (from fees)
    mintable_feelssol: float = 0.0              # Synthetic FeelsSOL from JitoSOL yield
    deployed_feelssol: float = 0.0              # Total FeelsSOL deployed to floor positions
    buffer_routed_cumulative: float = 0.0       # Cumulative fees routed to Buffer
    mint_cumulative: float = 0.0                # Cumulative synthetic FeelsSOL minted
    lp_fee_cumulative: float = 0.0              # Cumulative fees paid to LPs
    initial_buffer_balance: float = 0.0         # Reference point for JIT health checks
    last_pomm_deployment: int = -10_000         # Timestamp of last POMM deployment (minute)
    pomm_deployments_count: int = 0             # Total number of POMM deployments


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state at a point in time.
    
    Captures all relevant state at a single simulation minute for metrics
    collection and analysis. Used to track price evolution, trading activity,
    funding accumulation, and significant events like POMM deployments.
    """
    timestamp: int                              # Simulation minute (0-based)
    sol_price_usd: float                        # SOL market price in USD
    floor_price_feelssol: float                 # Floor price in FeelsSOL terms
    floor_price_usd: float                      # Floor price in USD (floor * SOL price)
    floor_state: FloorFundingState              # Complete funding state snapshot
    volume_feelssol: float = 0.0                # Trading volume this minute
    fees_collected: float = 0.0                 # Swap fees collected this minute
    jit_virtual_liquidity: float = 0.0         # Virtual liquidity deployed by JIT
    jit_absorbed_volume: float = 0.0           # Volume absorbed by contrarian JIT placement
    jit_active: bool = False                    # Whether JIT was active this minute
    jit_side: Optional[str] = None              # Direction of JIT placement ('ask' or 'bid')
    jit_range: Optional[Tuple[int, int]] = None # Tick range covered by the virtual position
    lp_fees_distributed: float = 0.0            # Fees credited to LPs this minute
    events: Dict[str, bool] = None              # Significant events (POMM deployments)
    participant_volumes: Dict[str, float] = None  # Volume by participant type
    price_path: Tuple[int, int] = (0, 0)          # Tick range traversed this minute
    
    def __post_init__(self):
        if self.events is None:
            self.events = {"pomm_deployed": False}  # Initialize event tracking
        if self.participant_volumes is None:
            self.participant_volumes = {}


@dataclass
class SimulationResults:
    """Results from a complete simulation run.
    
    Contains all simulation data including minute-by-minute snapshots,
    hourly aggregates for analysis, and the configuration used.
    Primary output for metrics analysis and visualization.
    """
    snapshots: List[SimulationSnapshot]         # Minute-by-minute state snapshots
    hourly_aggregates: List[Dict[str, float]]   # Hourly summary statistics
    config: SimulationConfig                    # Configuration used for this run

"""Core simulation engine using AgentPy framework for the Feels Protocol.

This module implements the main simulation model using AgentPy's advanced features
including proper experiment framework, data collection, and agent-based modeling
capabilities.
"""

import math
import copy
from typing import Any, Dict, List, Optional, Tuple

import agentpy as ap
import numpy as np

from .config import SimulationConfig
from .participants import (
    ParticipantConfig,
    ParticipantRegistry,
    aggregate_participant_metrics,
    create_participant_registry,
)
from .pricing import price_to_tick, tick_to_price
from .state import FloorFundingState, SimulationSnapshot


class PriceHistory:
    """Rolling price history for TWAP and volatility calculations."""

    def __init__(self, max_minutes: int = 720) -> None:
        self.max_minutes = max_minutes
        self.records: List[Tuple[int, int]] = []

    def add(self, minute: int, tick: int) -> None:
        self.records.append((minute, tick))
        cutoff = minute - self.max_minutes
        while self.records and self.records[0][0] < cutoff:
            self.records.pop(0)

    def get_twap(self, minute: int, window_seconds: int, min_duration_seconds: int) -> Optional[int]:
        if not self.records:
            return None
        window_minutes = max(1, window_seconds // 60)
        min_duration_minutes = max(1, min_duration_seconds // 60)
        cutoff = minute - window_minutes
        recent = [tick for ts, tick in self.records if ts >= cutoff]
        if len(recent) < 2:
            return None
        duration = max(recent) - min(recent)  # Simple time span check
        if duration < min_duration_minutes:
            return None
        return int(round(np.mean(recent)))

    def get_volatility(self, minute: int, window: int = 120) -> float:
        if len(self.records) < 2:
            return 0.0
        cutoff = minute - window
        ticks = [tick for ts, tick in self.records if ts >= cutoff]
        if len(ticks) < 2:
            return 0.0
        diffs = np.diff(ticks)
        if len(diffs) == 0:
            return 0.0
        return float(np.std(diffs) / max(1, np.mean(np.abs(ticks))))


class JitController:
    """JIT liquidity controller for market bootstrapping."""

    def __init__(self, config: SimulationConfig) -> None:
        self.enabled = config.jit_enabled
        self.base_cap_bps = config.jit_base_cap_bps
        self.per_slot_cap_bps = config.jit_per_slot_cap_bps
        self.volume_boost_factor = config.jit_volume_boost_factor
        self.max_duration_minutes = int(config.jit_max_duration_hours * 60)
        self.buffer_health_threshold = config.jit_buffer_health_threshold
        self.current_slot: int = -1
        self.slot_used: float = 0.0
        self.initial_buffer: Optional[float] = None
        self.total_volume_boost: float = 0.0
        self.active_minutes: int = 0

    def apply(self, minute: int, base_volume: float, buffer_balance: float) -> Tuple[float, float, bool]:
        """Apply JIT liquidity boost to base trading volume with safety checks.
        
        Returns: (adjusted_volume, boost_amount, was_active)
        """
        # Basic safety checks: JIT enabled, positive volume and buffer
        if not self.enabled or base_volume <= 0 or buffer_balance <= 0:
            return base_volume, 0.0, False
        # Auto-disable after maximum duration (market maturity)
        if self.max_duration_minutes and minute >= self.max_duration_minutes:
            return base_volume, 0.0, False

        # Reset per-minute budget tracking
        if self.current_slot != minute:
            self.current_slot = minute
            self.slot_used = 0.0

        # Set initial buffer reference for health calculations
        if self.initial_buffer is None and buffer_balance > 0:
            self.initial_buffer = buffer_balance

        # Disable if buffer health is too low (preserve capital)
        if self.initial_buffer and buffer_balance < self.initial_buffer * self.buffer_health_threshold:
            return base_volume, 0.0, False

        # Calculate JIT budget caps based on buffer balance
        per_trade = buffer_balance * (self.base_cap_bps / 10_000.0)  # Per-swap cap
        per_slot = buffer_balance * (self.per_slot_cap_bps / 10_000.0)  # Per-minute cap
        remaining = max(0.0, per_slot - self.slot_used)  # Remaining minute budget
        allowance = max(0.0, min(per_trade, remaining))  # Final allowance

        # No budget available
        if allowance <= 0:
            return base_volume, 0.0, False

        # Calculate age-based decay factor (strongest when market is young)
        age_factor = 1.0 - (minute / max(1, self.max_duration_minutes)) if self.max_duration_minutes else 1.0
        age_factor = max(0.1, age_factor)  # Minimum 10% effectiveness

        # Calculate potential volume boost with age decay
        potential_extra = base_volume * self.volume_boost_factor * age_factor
        jit_extra = min(potential_extra, allowance)  # Apply budget constraint
        if jit_extra <= 0:
            return base_volume, 0.0, False

        # Update usage tracking and metrics
        self.slot_used += jit_extra  # Track slot usage
        self.total_volume_boost += jit_extra  # Cumulative boost
        self.active_minutes += 1  # Active minute count
        return base_volume + jit_extra, jit_extra, True


class FeelsMarketModel(ap.Model):
    """Agent-based model for Feels Protocol using full AgentPy framework.
    
    This model leverages AgentPy's advanced features including:
    - Proper experiment framework integration
    - Built-in data collection and reporting
    - Advanced random number generation
    - Structured parameter handling
    """

    def setup(self) -> None:
        """Initialize the model using AgentPy's setup lifecycle."""
        # Extract configuration from AgentPy parameters
        self.simulation_config: SimulationConfig = self.p.get('simulation_config')
        if self.simulation_config is None:
            # Reconstruct config from individual parameters for experiment compatibility
            self.simulation_config = SimulationConfig()
            for key, value in self.p.items():
                if hasattr(self.simulation_config, key):
                    setattr(self.simulation_config, key, value)

        # Core simulation state
        self.minute: int = 0
        self.sol_price_usd: float = self.p.get('initial_sol_price', 100.0)
        self.prev_sol_price_usd: float = self.sol_price_usd
        self.current_tick: int = price_to_tick(1.0)
        self.sqrt_price: float = math.sqrt(tick_to_price(self.current_tick))
        
        # Floor price tracking with monotonic advancement
        initial_deployed = self.p.get('initial_deployed_feelssol', 1000.0)
        circulating_supply = self.p.get('circulating_supply', 1000000.0)
        self.max_floor_price_feelssol: float = initial_deployed / max(1.0, circulating_supply)
        self.max_floor_price_usd: float = self.max_floor_price_feelssol * self.sol_price_usd

        # Market context for participant decision making
        self.market_ctx: Dict[str, Any] = {
            "minute": self.minute,
            "tick": self.current_tick,
            "price_change": 0.0,
            "fee_rate": self.p.get('base_fee_rate', 0.003),
            "volatility": self.p.get('volatility_daily', 0.05),
            "price_discrepancy": 0.0,
        }

        # Price history for TWAP and volatility calculations
        self.price_history = PriceHistory()
        self.price_history.add(self.minute, self.current_tick)

        # Floor funding state
        self.floor_state = FloorFundingState(
            buffer_balance=self.p.get('initial_buffer_balance', 0.0),
            deployed_feelssol=initial_deployed,
            initial_buffer_balance=self.p.get('initial_buffer_balance', 0.0),
        )

        # Simulation data collection
        self.snapshots: List[SimulationSnapshot] = []
        self.hourly_aggregates: List[Dict[str, Any]] = []
        self.market_age_minutes: int = 0

        # JIT liquidity controller
        self.jit_controller = JitController(self.simulation_config)

        # Participant system (optional)
        self.participants: Optional[ParticipantRegistry] = None
        if self.p.get('enable_participants', False):
            self.participant_config = self.p.get('participant_config', ParticipantConfig())
            self.participants = create_participant_registry(self, self.participant_config)

        # Record initial state using AgentPy's data collection
        self.record('initial_setup_complete', True)

    def step(self) -> None:
        """Execute one simulation step using AgentPy's step lifecycle."""
        # Update market environment
        self._update_sol_price()
        
        # Update market context for participants
        self.market_ctx.update({
            "minute": self.minute,
            "tick": self.current_tick,
            "price_change": (self.sol_price_usd - self.prev_sol_price_usd) / max(self.prev_sol_price_usd, 1e-9),
            "price_discrepancy": tick_to_price(self.current_tick) - (self.sol_price_usd / self.p.get('initial_sol_price', 100.0)),
        })

        # Update participant liquidity positions
        if self.participants:
            self._update_lp_state()

        # Generate trading activity
        buy_volume, sell_volume, per_type = self._collect_trading_activity()
        total_volume = buy_volume + sell_volume

        # Apply JIT liquidity boost
        adjusted_volume, jit_boost, jit_active = self.jit_controller.apply(
            self.minute, total_volume, self.floor_state.buffer_balance
        )

        # Scale volumes if JIT is active
        if total_volume > 0 and adjusted_volume > 0:
            scale = adjusted_volume / total_volume
            buy_volume *= scale
            sell_volume *= scale
            for key in list(per_type.keys()):
                per_type[key] *= scale
        total_volume = buy_volume + sell_volume

        # Calculate fees and apply price impact
        fee_rate = self.p.get('base_fee_rate', 0.003)
        total_fees = total_volume * fee_rate
        start_tick, end_tick = self._apply_trade_impact(buy_volume, sell_volume)

        # Route fees to stakeholders
        self._route_fees(total_fees, start_tick, end_tick)

        # Accrue synthetic FeelsSOL minting
        self._accrue_minting()

        # Evaluate POMM deployment
        pomm_deployed = self._evaluate_pomm()

        # Update price history
        self.price_history.add(self.minute, self.current_tick)

        # Calculate current floor price with monotonic advancement
        floor_price_feelssol = self.floor_state.deployed_feelssol / max(1.0, self.p.get('circulating_supply', 1000000.0))
        self.max_floor_price_feelssol = max(self.max_floor_price_feelssol, floor_price_feelssol)
        floor_price_usd = self.max_floor_price_feelssol * self.sol_price_usd
        self.max_floor_price_usd = max(self.max_floor_price_usd, floor_price_usd)

        # Create snapshot
        snapshot = SimulationSnapshot(
            timestamp=self.minute,
            sol_price_usd=self.sol_price_usd,
            floor_price_feelssol=self.max_floor_price_feelssol,
            floor_price_usd=self.max_floor_price_usd,
            floor_state=copy.deepcopy(self.floor_state),
            volume_feelssol=total_volume,
            fees_collected=total_fees,
            jit_volume_boost=jit_boost,
            jit_active=jit_active,
            events={"pomm_deployed": pomm_deployed},
            participant_volumes=per_type,
            price_path=(start_tick, end_tick),
        )
        self.snapshots.append(snapshot)

        # Record data using AgentPy's built-in data collection
        self.record('minute', self.minute)
        self.record('sol_price_usd', self.sol_price_usd)
        self.record('floor_price_usd', self.max_floor_price_usd)
        self.record('buffer_balance', self.floor_state.buffer_balance)
        self.record('total_volume', total_volume)
        self.record('total_fees', total_fees)
        self.record('pomm_deployed', pomm_deployed)

        # Advance time
        self.minute += 1
        self.market_age_minutes = self.minute

        # Process hourly aggregates
        if self.minute % 60 == 0:
            self._process_hourly_aggregate()

    def _update_sol_price(self) -> None:
        """Update SOL price using geometric Brownian motion."""
        dt = 1.0 / (365.25 * 24 * 60)
        drift = self.p.get('trend_bias', 0.0) * 0.001
        volatility = self.p.get('volatility_daily', 0.05) / math.sqrt(365.25)
        
        # Use AgentPy's random number generator (uses normalvariate instead of normal)
        random_shock = self.random.normalvariate(0, volatility * math.sqrt(dt))
        log_return = drift * dt + random_shock
        
        self.prev_sol_price_usd = self.sol_price_usd
        self.sol_price_usd *= math.exp(log_return)

    def _collect_trading_activity(self) -> Tuple[float, float, Dict[str, float]]:
        """Collect trading activity from participants or synthetic model."""
        if not self.participants:
            # Synthetic trading activity
            base_volume = 1000.0 / (24 * 60)
            return base_volume * 0.5, base_volume * 0.5, {"synthetic": base_volume}

        buy_volume = 0.0
        sell_volume = 0.0
        per_type: Dict[str, float] = {}

        for agent in self.participants.trader_agents:
            decision = agent.trade()
            net = decision["net"]
            gross = decision["abs"]
            if gross <= 0:
                continue

            agent_type = agent.participant_type
            per_type[agent_type] = per_type.get(agent_type, 0.0) + gross

            if net > 0:
                buy_volume += net
            else:
                sell_volume += -net

        return buy_volume, sell_volume, per_type

    def _apply_trade_impact(self, buy_volume: float, sell_volume: float) -> Tuple[int, int]:
        """Apply price impact from trading volume."""
        start_tick = self.current_tick
        total_volume = buy_volume + sell_volume
        net_volume = buy_volume - sell_volume
        
        if total_volume <= 0 or abs(net_volume) < 1e-9:
            return start_tick, start_tick

        # Calculate available liquidity
        liquidity = self._active_liquidity(start_tick)
        net_fraction = max(-0.95, min(0.95, net_volume / liquidity))
        price_ratio = max(1e-6, 1.0 + net_fraction)
        delta_tick = int(round(math.log(price_ratio) / math.log(1.0001)))
        
        end_tick = start_tick + delta_tick
        self.current_tick = end_tick
        self.sqrt_price = math.sqrt(max(1e-9, tick_to_price(end_tick)))
        
        if self.participants:
            self.participants.note_market_tick(end_tick)

        return start_tick, end_tick

    def _active_liquidity(self, tick: int) -> float:
        """Calculate active liquidity at given tick."""
        if not self.participants:
            return max(1.0, self.floor_state.deployed_feelssol + 1.0)
        liquidity = self.participants.active_liquidity(tick)
        return max(liquidity, 1.0)

    def _route_fees(self, total_fees: float, start_tick: int, end_tick: int) -> None:
        """Route fees to stakeholders according to fee split configuration."""
        if total_fees <= 0:
            return

        fee_split = self.p.get('fee_split', {
            'protocol_fee_rate_bps': 100, 'creator_fee_rate_bps': 50
        })
        
        # Calculate protocol and creator fees first (fixed percentages)
        protocol_amount = total_fees * (fee_split['protocol_fee_rate_bps'] / 10_000.0)
        creator_amount = total_fees * (fee_split['creator_fee_rate_bps'] / 10_000.0)
        
        # Buffer gets the remainder after protocol and creator fees are deducted
        buffer_amount = max(0.0, total_fees - protocol_amount - creator_amount)

        # Distribute LP fees (they earn through position accrual, not direct percentage)
        distributed_lp = 0.0
        if self.participants and buffer_amount > 0:
            # LPs can earn fees from their active liquidity during swaps
            # This is separate from the main fee split - they accrue fees based on their liquidity
            distributed_lp = self.participants.accrue_lp_fees(buffer_amount * 0.1, start_tick, end_tick)  # Small portion for LP accrual
            self.floor_state.lp_fee_cumulative += distributed_lp

        # Update balances
        self.floor_state.buffer_balance += buffer_amount
        self.floor_state.buffer_routed_cumulative += buffer_amount
        self.floor_state.treasury_balance += protocol_amount
        self.floor_state.creator_balance += creator_amount
        
        # Set initial buffer reference
        if self.floor_state.initial_buffer_balance <= 0 and self.floor_state.buffer_balance > 0:
            self.floor_state.initial_buffer_balance = self.floor_state.buffer_balance

    def _accrue_minting(self) -> None:
        """Accrue synthetic FeelsSOL minting from JitoSOL yield."""
        dt = 1.0 / (365.25 * 24 * 60)
        total_supply = self.p.get('total_supply', 1000000.0)
        yield_apy = self.p.get('jitosol_yield_apy', 0.07)
        mint_growth = total_supply * yield_apy * dt
        
        self.floor_state.mintable_feelssol += mint_growth
        self.floor_state.mint_cumulative += mint_growth

    def _evaluate_pomm(self) -> bool:
        """Evaluate and execute POMM deployment if conditions are met."""
        pomm_config = self.p.get('pomm_config', {
            'cooldown_seconds': 3600,
            'threshold': 100.0,
            'deployment_ratio': 0.5,
            'twap_window_seconds': 300,
            'min_twap_seconds': 60,
        })
        
        # Check cooldown
        elapsed = self.minute - self.floor_state.last_pomm_deployment
        if elapsed < pomm_config['cooldown_seconds'] / 60:
            return False
        
        # Check capital threshold
        deployable = self.floor_state.buffer_balance + self.floor_state.mintable_feelssol
        if deployable < pomm_config['threshold']:
            return False
        
        # Check TWAP requirement
        twap_tick = self.price_history.get_twap(
            self.minute, 
            pomm_config['twap_window_seconds'], 
            pomm_config['min_twap_seconds']
        )
        if twap_tick is None:
            return False
        
        # Execute deployment
        amount = deployable * pomm_config['deployment_ratio']
        buffer_used = min(self.floor_state.buffer_balance, amount)
        mint_used = min(self.floor_state.mintable_feelssol, amount - buffer_used)
        total_deployed = buffer_used + mint_used
        
        if total_deployed <= 0:
            return False
        
        self.floor_state.buffer_balance -= buffer_used
        self.floor_state.mintable_feelssol -= mint_used
        self.floor_state.deployed_feelssol += total_deployed
        self.floor_state.last_pomm_deployment = self.minute
        self.floor_state.pomm_deployments_count += 1
        
        return True

    def _update_lp_state(self) -> None:
        """Update liquidity provider state."""
        if self.participants:
            self.participants.liquidity_providers.step()

    def _process_hourly_aggregate(self) -> None:
        """Process hourly aggregate data."""
        if len(self.snapshots) < 60:
            return
            
        hour_snapshots = self.snapshots[-60:]
        
        # Calculate metrics
        metrics = aggregate_participant_metrics(
            self.participants, self.current_tick
        ) if self.participants else {}
        
        hourly = {
            "hour": self.minute // 60 - 1,
            "avg_sol_price": float(np.mean([s.sol_price_usd for s in hour_snapshots])),
            "avg_floor_price": float(np.mean([s.floor_price_usd for s in hour_snapshots])),
            "total_volume": float(np.sum([s.volume_feelssol for s in hour_snapshots])),
            "total_fees": float(np.sum([s.fees_collected for s in hour_snapshots])),
            "pomm_deployments": int(sum(1 for s in hour_snapshots if s.events.get("pomm_deployed"))),
            "buffer_balance": self.floor_state.buffer_balance,
            "mintable_feelssol": self.floor_state.mintable_feelssol,
            "treasury_balance": self.floor_state.treasury_balance,
            "buffer_routed_cumulative": self.floor_state.buffer_routed_cumulative,
            "mint_cumulative": self.floor_state.mint_cumulative,
            "deployed_feelssol": self.floor_state.deployed_feelssol,
            "jit_volume_boost": float(np.sum([s.jit_volume_boost for s in hour_snapshots])),
            "jit_active_minutes": int(sum(1 for s in hour_snapshots if s.jit_active)),
        }
        
        if metrics:
            hourly["participant_metrics"] = metrics
            
        # Aggregate participant volumes
        volume_totals: Dict[str, float] = {}
        for s in hour_snapshots:
            for k, v in s.participant_volumes.items():
                volume_totals[k] = volume_totals.get(k, 0.0) + v
        if volume_totals:
            hourly["participant_volumes"] = volume_totals
            
        self.hourly_aggregates.append(hourly)


# Backward compatibility wrapper
class FeelsSimulation:
    """High-level simulation interface providing backward compatibility.
    
    This class maintains the original API while internally using the AgentPy
    framework for improved performance and capabilities.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.config.validate()

    def run(self, hours: int):
        """Run simulation using AgentPy framework."""
        from .state import SimulationResults
        
        # Convert to AgentPy parameters
        params = self.config.to_agentpy_params()
        
        # Create and run model
        model = FeelsMarketModel(params)
        model.setup()
        
        # Run simulation steps
        for _ in range(hours * 60):
            model.step()
        
        # Store results for get_dataframe method
        self._last_results = SimulationResults(
            snapshots=model.snapshots,
            hourly_aggregates=model.hourly_aggregates,
            config=self.config,
        )
        
        return self._last_results

    def get_dataframe(self):
        """Get polars DataFrame of simulation data.
        
        Converts simulation snapshots to a structured polars DataFrame for analysis.
        Uses the metrics module function for efficient conversion.
        
        Returns:
            polars DataFrame containing all simulation snapshots and derived metrics
        """
        from .metrics import snapshots_to_dataframe
        
        # Run simulation first if no snapshots exist
        if not hasattr(self, '_last_results') or not self._last_results:
            # Need to run simulation to generate snapshots
            return None
            
        # Convert snapshots to DataFrame using metrics function
        return snapshots_to_dataframe(self._last_results.snapshots)
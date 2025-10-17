"""Configuration for Feels simulation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .participants import ParticipantConfig


@dataclass
class SimulationConfig:
    """Configuration bundle for a single simulation run.
    
    Controls all simulation parameters including market environment, fee routing,
    POMM deployment logic, JIT liquidity bootstrapping, and participant behavior.
    Designed to match on-chain Feels Protocol constraints and defaults.
    """

    # Market environment - external factors affecting trading
    initial_sol_price_usd: float = 100.0  # Starting SOL price for price evolution
    sol_volatility_daily: float = 0.05  # Daily price volatility (5% = moderate crypto volatility)
    sol_trend_bias: float = 0.0  # Market drift (-1=bearish, 0=neutral, +1=bullish)

    # Swap fee structure - revenue generation for protocol
    base_fee_bps: int = 30  # Base swap fee in basis points (30 = 0.30%)
    impact_fee_enabled: bool = False  # Dynamic impact fees (disabled for simplicity)

    # Fee routing - how swap fees are allocated between stakeholders
    # Protocol and creator fees are taken first, remainder goes to Buffer for POMM
    protocol_fee_rate_bps: int = 100  # Protocol treasury fee rate (100 bps = 1.0%)
    creator_fee_rate_bps: int = 50    # Token creator fee rate (50 bps = 0.5%)

    # Tick spacing - granularity of price levels (immutable per market on-chain)
    tick_spacing: int = 10  # Each tick = 0.01% price change, spacing affects liquidity distribution

    # POMM parameters - Protocol-Owned Market Making for floor price advancement
    pomm_threshold_tokens: float = 100.0  # Minimum Buffer balance before POMM deployment
    pomm_cooldown_seconds: int = 60       # Minimum time between POMM deployments (anti-spam)
    pomm_deployment_ratio: float = 0.5    # Fraction of available funds deployed each time
    pomm_width_multiplier: int = 20       # Position width = tick_spacing × multiplier
    pomm_min_width_ticks: int = 10        # Minimum position width (tight spread protection)
    pomm_max_width_ticks: int = 2000      # Maximum position width (capital efficiency)
    pomm_twap_window_seconds: int = 300   # 5-minute TWAP for manipulation-resistant placement
    pomm_min_twap_seconds: int = 60       # Minimum observation window before deployment

    # JIT parameters - Just-In-Time liquidity for market bootstrapping
    jit_enabled: bool = True                      # Enable JIT liquidity system
    jit_base_cap_bps: float = 300.0              # Per-swap budget: 3% of Buffer balance
    jit_per_slot_cap_bps: float = 500.0          # Per-minute budget: 5% of Buffer balance
    jit_max_multiplier: float = 10.0             # Peak effective liquidity multiplier at current price
    jit_concentration_width: int = 10            # Tick range for concentrated JIT liquidity
    jit_volume_boost_factor: float = 0.3         # Multiplier for JIT virtual liquidity sizing
    jit_max_duration_hours: float = 24.0         # Auto-disable JIT after 24 hours (market maturity)
    jit_buffer_health_threshold: float = 0.30    # Disable JIT when Buffer < 30% of initial

    # Simulation controls
    random_seed: int = 42  # Seed forwarded to agentpy for deterministic runs

    # JitoSOL backing assumptions - synthetic minting rate from staking yield
    jitosol_yield_apy: float = 0.07  # 7% APY appreciation vs SOL (staking + MEV rewards)

    # Token economics - supply and circulation parameters
    total_supply: float = 1_000_000_000    # Total FeelsSOL token supply (1B)
    circulating_supply: float = 1_000_000_000  # Initially all tokens circulating

    # Initial floor state - starting conditions for POMM system
    initial_deployed_feelssol: float = 5_000_000.0  # Initial floor liquidity deployed (sets ~$0.50 starting floor)
    initial_buffer_balance: float = 0.0          # Buffer starts empty, grows from fees

    # Participant behavior - controls synthetic trading activity
    enable_participant_behavior: bool = True     # Enable sophisticated participant modeling
    participant_config: ParticipantConfig = field(default_factory=ParticipantConfig)  # Participant parameters

    def __post_init__(self):
        # Ensure participant config exists (field default handles most cases)
        if self.participant_config is None:
            self.participant_config = ParticipantConfig()

    def validate(self) -> None:
        """Validate configuration against program invariants."""
        # Fee routing constraints - protocol and creator fees taken first, remainder to buffer
        assert 0 <= self.protocol_fee_rate_bps <= 10000, "Protocol fee rate must be between 0 and 10000 bps"
        assert 0 <= self.creator_fee_rate_bps <= 10000, "Creator fee rate must be between 0 and 10000 bps"
        assert (self.protocol_fee_rate_bps + self.creator_fee_rate_bps) <= 10000, "Combined protocol and creator fees cannot exceed 10000 bps"

        # Fee parameters
        assert 0 <= self.base_fee_bps <= 1000, "Base fee must be within 0-10%"

        # POMM parameter sanity checks
        assert self.tick_spacing > 0, "Tick spacing must be positive"
        assert self.pomm_min_width_ticks > 0
        assert self.pomm_min_width_ticks <= self.pomm_max_width_ticks
        assert self.pomm_deployment_ratio > 0 and self.pomm_deployment_ratio <= 1.0

        # JIT constraints
        if self.jit_enabled:
            assert self.jit_base_cap_bps >= 0
            assert self.jit_per_slot_cap_bps >= 0
            assert self.jit_max_multiplier >= 1.0
            assert self.jit_volume_boost_factor >= 0
            assert self.jit_buffer_health_threshold >= 0

        assert self.jitosol_yield_apy >= 0, "Yield cannot be negative"
        assert self.circulating_supply <= self.total_supply, "Circulating supply cannot exceed total supply"

    @classmethod
    def from_calibration_file(cls, file_path: str, overrides: Optional[dict] = None):
        """
        Load configuration (and participant parameters) from JSON.

        Structure:
        {
            "simulation_config": {...},
            "participant_config": {...}
        }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {file_path}")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        sim_config = data.get("simulation_config", {})
        participant_config_data = data.get("participant_config", {})

        overrides = overrides or {}
        sim_config.update(overrides.get("simulation_config", {}))
        participant_config_data.update(overrides.get("participant_config", {}))

        participant_config = ParticipantConfig(**participant_config_data)
        sim_config["participant_config"] = participant_config

        return cls(**sim_config)

    def to_agentpy_params(self) -> dict:
        """Convert configuration to agentpy-friendly parameter dictionary."""
        return {
            "base_fee_rate": self.base_fee_bps / 10_000.0,
            "tick_spacing": self.tick_spacing,
            "initial_sol_price": self.initial_sol_price_usd,
            "volatility_daily": self.sol_volatility_daily,
            "trend_bias": self.sol_trend_bias,
            "jitosol_yield_apy": self.jitosol_yield_apy,
            "total_supply": self.total_supply,
            "circulating_supply": self.circulating_supply,
            "enable_participants": self.enable_participant_behavior,
            "participant_config": self.participant_config,
            "jit_config": {
                "enabled": self.jit_enabled,
                "base_cap_bps": self.jit_base_cap_bps,
                "per_slot_cap_bps": self.jit_per_slot_cap_bps,
                "volume_boost_factor": self.jit_volume_boost_factor,
                "max_duration_hours": self.jit_max_duration_hours,
                "buffer_health_threshold": self.jit_buffer_health_threshold,
            },
            "pomm_config": {
                "threshold": self.pomm_threshold_tokens,
                "cooldown_seconds": self.pomm_cooldown_seconds,
                "deployment_ratio": self.pomm_deployment_ratio,
                "width_multiplier": self.pomm_width_multiplier,
                "min_width_ticks": self.pomm_min_width_ticks,
                "max_width_ticks": self.pomm_max_width_ticks,
                "twap_window_seconds": self.pomm_twap_window_seconds,
                "min_twap_seconds": self.pomm_min_twap_seconds,
            },
            "fee_split": {
                "protocol_fee_rate_bps": self.protocol_fee_rate_bps,
                "creator_fee_rate_bps": self.creator_fee_rate_bps,
            },
            "initial_buffer_balance": self.initial_buffer_balance,
            "initial_deployed_feelssol": self.initial_deployed_feelssol,
            "simulation_config": self,
        }

    @classmethod
    def create_fee_scenario(cls, scenario: str, **kwargs):
        """Convenience helper for common fee routing experiments."""
        fee_scenarios = {
            "default": {"protocol_fee_rate_bps": 100, "creator_fee_rate_bps": 50},  # 1.0% protocol, 0.5% creator
            "protocol_sustainable": {"protocol_fee_rate_bps": 150, "creator_fee_rate_bps": 50},  # 1.5% protocol, 0.5% creator
            "creator_focused": {"protocol_fee_rate_bps": 100, "creator_fee_rate_bps": 200},  # 1.0% protocol, 2.0% creator
            "creator_incentive": {"protocol_fee_rate_bps": 100, "creator_fee_rate_bps": 200},  # 1.0% protocol, 2.0% creator
            "balanced_growth": {"protocol_fee_rate_bps": 125, "creator_fee_rate_bps": 75},  # 1.25% protocol, 0.75% creator
            "minimum_protocol": {"protocol_fee_rate_bps": 50, "creator_fee_rate_bps": 25},   # 0.5% protocol, 0.25% creator
            "maximum_protocol": {"protocol_fee_rate_bps": 200, "creator_fee_rate_bps": 50},  # 2.0% protocol, 0.5% creator
        }

        if scenario not in fee_scenarios:
            available = ", ".join(sorted(fee_scenarios.keys()))
            raise ValueError(f"Unknown fee scenario '{scenario}'. Available: {available}")

        config_params = fee_scenarios[scenario].copy()
        config_params.update(kwargs)

        return cls(**config_params)

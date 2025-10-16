"""Configuration for Feels simulation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .participants import ParticipantConfig
@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    
    # Market environment
    initial_sol_price_usd: float = 100.0
    sol_volatility_daily: float = 0.05  # 5% daily volatility
    sol_trend_bias: float = 0.0  # No directional bias
    
    # Protocol parameters
    base_fee_bps: int = 30  # 0.30% base fee
    impact_fee_enabled: bool = False  # Currently disabled
    
    # Fee distribution (percentages) - matches current program defaults
    treasury_share_pct: float = 1.0    # Protocol treasury (max 10% per program constraints)
    creator_share_pct: float = 0.5     # Token creator rewards (max 5% per program constraints)
    # buffer_share_pct is calculated as remainder: 100 - treasury - creator = 98.5%
    
    # POMM parameters
    pomm_threshold_tokens: float = 100.0
    pomm_cooldown_seconds: int = 60
    pomm_deployment_ratio: float = 0.5
    floor_buffer_ticks: int = 50
    
    # JitoSOL yield
    jitosol_yield_apy: float = 0.07  # 7% APY
    
    # Token economics
    total_supply: float = 1_000_000_000  # 1B tokens
    circulating_supply: float = 1_000_000_000
    
    # Initial conditions
    initial_deployed_feelssol: float = 1000.0  # Initial FeelsSOL deployed as floor liquidity
    initial_buffer_balance: float = 0.0
    
    # Participant behavior configuration
    enable_participant_behavior: bool = True
    participant_config: ParticipantConfig = None
    
    def __post_init__(self):
        if self.participant_config is None:
            self.participant_config = ParticipantConfig()

    @property
    def buffer_share_pct(self) -> float:
        """Calculate buffer share as remainder after protocol and creator fees."""
        return 100.0 - self.treasury_share_pct - self.creator_share_pct
    
    def validate(self) -> None:
        """Validate configuration parameters against protocol constraints."""
        # Protocol constraints from program code
        assert 0 <= self.treasury_share_pct <= 10.0, "Treasury share must be 0-10% (protocol constraint)"
        assert 0 <= self.creator_share_pct <= 5.0, "Creator share must be 0-5% (protocol constraint)"
        assert (self.treasury_share_pct + self.creator_share_pct) <= 10.0, "Combined treasury + creator must be ≤10%"
        
        # Buffer must be positive (calculated as remainder)
        assert self.buffer_share_pct > 0, f"Buffer share must be positive, got {self.buffer_share_pct:.1f}%"
        
        # Other validations
        assert 0 <= self.base_fee_bps <= 1000, "Base fee must be within 0-10%"
        assert 0 < self.pomm_deployment_ratio <= 1.0, "Deployment ratio must be (0, 1]"
        assert self.jitosol_yield_apy >= 0, "Yield cannot be negative"
        assert self.circulating_supply <= self.total_supply, "Circulating supply cannot exceed total supply"
    
    @classmethod
    def from_calibration_file(cls, file_path: str, overrides: Optional[dict] = None):
        """Load configuration from calibration JSON file with optional overrides."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {file_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extract configurations
        sim_config = data.get('simulation_config', {})
        participant_config_data = data.get('participant_config', {})
        
        # Handle legacy buffer_share_pct parameter - remove it if present
        if 'buffer_share_pct' in sim_config:
            sim_config.pop('buffer_share_pct')  # Remove legacy parameter
        
        # Apply overrides
        if overrides:
            sim_config.update(overrides.get('simulation_config', {}))
            participant_config_data.update(overrides.get('participant_config', {}))
        
        # Create participant config
        participant_config = ParticipantConfig(**participant_config_data)
        sim_config['participant_config'] = participant_config
        
        return cls(**sim_config)
    
    @classmethod
    def create_fee_scenario(cls, scenario: str, **kwargs):
        """Create configuration with predefined fee split scenarios for parameter exploration."""
        fee_scenarios = {
            "current_default": {"treasury_share_pct": 1.0, "creator_share_pct": 0.5},      # 98.5% buffer
            "protocol_sustainable": {"treasury_share_pct": 7.0, "creator_share_pct": 3.0}, # 90% buffer (sustainable protocol funding)
            "creator_incentive": {"treasury_share_pct": 5.0, "creator_share_pct": 5.0},    # 90% buffer
            "balanced_growth": {"treasury_share_pct": 5.0, "creator_share_pct": 3.0},      # 92% buffer
            "maximum_protocol": {"treasury_share_pct": 8.0, "creator_share_pct": 2.0},     # 90% buffer (near max sustainable)
            "invalid_high_protocol": {"treasury_share_pct": 15.0, "creator_share_pct": 5.0},  # 80% buffer (fails validation)
        }
        
        if scenario not in fee_scenarios:
            available = ", ".join(fee_scenarios.keys())
            raise ValueError(f"Unknown fee scenario '{scenario}'. Available: {available}")
        
        # Start with scenario parameters
        config_params = fee_scenarios[scenario].copy()
        # Override with any additional kwargs
        config_params.update(kwargs)
        
        return cls(**config_params)

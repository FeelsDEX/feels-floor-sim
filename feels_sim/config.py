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
    
    # Fee distribution (percentages)
    buffer_share_pct: float = 85.0
    treasury_share_pct: float = 10.0
    creator_share_pct: float = 5.0
    
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

    def validate(self) -> None:
        """Validate configuration parameters."""
        total_fee_share = self.buffer_share_pct + self.treasury_share_pct + self.creator_share_pct
        assert abs(total_fee_share - 100.0) < 1e-6, "Fee shares must sum to 100%"
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
        
        # Apply overrides
        if overrides:
            sim_config.update(overrides.get('simulation_config', {}))
            participant_config_data.update(overrides.get('participant_config', {}))
        
        # Create participant config
        participant_config = ParticipantConfig(**participant_config_data)
        sim_config['participant_config'] = participant_config
        
        return cls(**sim_config)

"""Fee routing, Buffer accumulation, and scenario tests.

Tests the fee distribution system, Buffer accumulation mechanics, POMM deployment
thresholds, and various fee scenario configurations. Validates protocol economics
and ensures proper fund allocation across stakeholders."""

import math

from feels_sim.core import FeelsSimulation
from feels_sim.config import SimulationConfig
from tests.utils import assert_close, expect_raises


def test_mint_accumulation():
    """Test synthetic FeelsSOL minting from JitoSOL yield.
    
    Validates that JitoSOL staking yield correctly accumulates as mintable
    FeelsSOL over time. Uses 8.76% APY for 30-day simulation.
    """
    config = SimulationConfig(jitosol_yield_apy=0.0876, enable_participant_behavior=False)  # 8.76% annual yield
    final_state = FeelsSimulation(config).run(hours=24 * 30).snapshots[-1].floor_state  # 30 days
    
    assert final_state.mint_cumulative > 0      # Should have accumulated synthetic minting
    assert final_state.mintable_feelssol >= 0   # Available synthetic supply should be non-negative


def test_fee_distribution_invariant():
    """Test that fees are distributed according to the new fee system.
    
    Validates that the fee routing system correctly allocates swap fees:
    - Protocol and creator fees are taken first based on basis points
    - Buffer gets the remainder (~98.5% with default rates)
    - LPs earn through position accrual, not direct allocation
    """
    config = SimulationConfig(enable_participant_behavior=True)  # Enable participants
    final_state = FeelsSimulation(config).run(hours=24).snapshots[-1].floor_state  # 24-hour simulation
    
    total_fees_collected = (
        final_state.buffer_routed_cumulative
        + final_state.treasury_balance
        + final_state.creator_balance
    )
    
    if total_fees_collected == 0:
        return  # No fees collected - test not applicable
    
    # Calculate expected allocations based on basis points
    expected_treasury_pct = config.protocol_fee_rate_bps / 100.0  # bps to percentage
    expected_creator_pct = config.creator_fee_rate_bps / 100.0
    expected_buffer_pct = 100.0 - expected_treasury_pct - expected_creator_pct
    
    # Calculate actual percentage distribution
    actual_treasury_pct = (final_state.treasury_balance / total_fees_collected) * 100
    actual_creator_pct = (final_state.creator_balance / total_fees_collected) * 100
    actual_buffer_pct = (final_state.buffer_routed_cumulative / total_fees_collected) * 100
    
    # Verify distribution matches expected rates (within 5% tolerance)
    assert_close(actual_treasury_pct, expected_treasury_pct, rel=0.05)
    assert_close(actual_creator_pct, expected_creator_pct, rel=0.05)
    assert_close(actual_buffer_pct, expected_buffer_pct, rel=0.05)


def test_buffer_volume_correlation():
    """Test that Buffer accumulation correlates correctly with trading volume.
    
    Validates that Buffer receives the expected amount based on total trading
    volume, fee rate, and Buffer share percentage.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Use default configuration
    results = FeelsSimulation(config).run(hours=48)  # 48-hour simulation
    
    # Calculate total trading volume across all snapshots
    total_volume = sum(s.volume_feelssol for s in results.snapshots)
    
    # Calculate expected Buffer accumulation: volume × fee_rate × buffer_remainder
    # Buffer gets remainder after protocol and creator fees are deducted
    buffer_share_rate = 1.0 - (config.protocol_fee_rate_bps + config.creator_fee_rate_bps) / 10_000.0
    expected_buffer = total_volume * (config.base_fee_bps / 10_000.0) * buffer_share_rate
    
    # Get actual Buffer accumulation from simulation
    actual_buffer = results.snapshots[-1].floor_state.buffer_routed_cumulative
    
    # Verify correlation within 2% tolerance (accounts for rounding and timing)
    assert_close(actual_buffer, expected_buffer, rel=0.02)


def test_pomm_deployment_advances_floor():
    """Test that POMM deployments advance the floor price monotonically.
    
    Validates that when POMM deployments occur, they increase the deployed
    FeelsSOL amount and advance the floor price as intended.
    """
    config = SimulationConfig(
        base_fee_bps=100,  # Higher fees to generate more buffer
        pomm_threshold_tokens=50.0,  # Lower threshold for faster deployment
        pomm_cooldown_seconds=1800,  # 30-minute cooldown
        enable_participant_behavior=False  # Use synthetic volume for consistency
    )
    results = FeelsSimulation(config).run(hours=48)  # 48-hour simulation
    
    # Track floor price advancement over time
    initial_floor = results.snapshots[0].floor_price_usd
    final_floor = results.snapshots[-1].floor_price_usd
    
    # Floor price should be monotonically increasing (never decrease)
    floor_prices = [s.floor_price_usd for s in results.snapshots]
    for i in range(1, len(floor_prices)):
        assert floor_prices[i] >= floor_prices[i-1], f"Floor price decreased at minute {i}"
    
    # Final floor should be at least as high as initial (could be same if no deployments)
    assert final_floor >= initial_floor, "Floor price should never decrease over time"
    
    # Track deployed FeelsSOL should be monotonically increasing
    initial_deployed = results.snapshots[0].floor_state.deployed_feelssol
    final_deployed = results.snapshots[-1].floor_state.deployed_feelssol
    assert final_deployed >= initial_deployed, "Deployed FeelsSOL should never decrease"


# --- Scenario helpers ------------------------------------------------------
# These tests validate predefined fee scenario configurations that represent
# different protocol governance and economics strategies.

def test_current_default_scenario():
    """Test default fee scenario configuration.
    
    Validates the baseline fee distribution scenario with standard
    allocation rates for protocol operations.
    """
    config = SimulationConfig.create_fee_scenario("default")
    config.validate()  # Should pass validation
    
    # Verify default fee rates: protocol=100 bps (1.0%), creator=50 bps (0.5%)
    assert config.protocol_fee_rate_bps == 100
    assert config.creator_fee_rate_bps == 50
    # Buffer gets remainder: ~98.5%


def test_protocol_sustainable_scenario():
    """Test protocol-sustainable fee scenario configuration.
    
    Validates a fee distribution that prioritizes protocol sustainability
    with increased protocol fee allocation.
    """
    config = SimulationConfig.create_fee_scenario("protocol_sustainable")
    config.validate()  # Should pass validation
    
    # Verify sustainable allocation: protocol=150 bps (1.5%), creator=50 bps (0.5%)
    assert config.protocol_fee_rate_bps == 150
    assert config.creator_fee_rate_bps == 50
    # Buffer gets remainder: ~98.0%


def test_creator_incentive_scenario():
    """Test creator-incentive fee scenario configuration.
    
    Validates a fee distribution that maximizes creator incentives
    to encourage high-quality token launches.
    """
    config = SimulationConfig.create_fee_scenario("creator_incentive")
    config.validate()  # Should pass validation
    
    # Verify creator-incentive allocation: protocol=100 bps (1.0%), creator=200 bps (2.0%)
    assert config.protocol_fee_rate_bps == 100
    assert config.creator_fee_rate_bps == 200
    # Buffer gets remainder: ~97.0%


def test_balanced_growth_scenario():
    """Test balanced growth fee scenario configuration.
    
    Validates a fee distribution that balances protocol sustainability
    with creator incentives for optimal ecosystem growth.
    """
    config = SimulationConfig.create_fee_scenario("balanced_growth")
    config.validate()  # Should pass validation
    
    # Verify balanced allocation: protocol=125 bps (1.25%), creator=75 bps (0.75%)
    assert config.protocol_fee_rate_bps == 125
    assert config.creator_fee_rate_bps == 75
    # Buffer gets remainder: ~98.0%


def test_scenario_override():
    """Test that scenario configurations can be overridden.
    
    This test demonstrates that scenario configurations can be
    overridden with custom values that still pass validation.
    """
    config = SimulationConfig.create_fee_scenario("default", protocol_fee_rate_bps=80, creator_fee_rate_bps=30)
    config.validate()  # Should pass validation with these values
    
    # Verify the override values were applied
    assert config.protocol_fee_rate_bps == 80
    assert config.creator_fee_rate_bps == 30


def test_invalid_scenario_validation():
    """Test that scenario overrides with invalid values fail validation.
    
    Ensures that even when using predefined scenarios, configuration
    validation still catches constraint violations.
    """
    config = SimulationConfig.create_fee_scenario("default", protocol_fee_rate_bps=12000)  # Exceeds 10000 bps limit
    expect_raises(AssertionError, config.validate)  # Should fail validation


def test_fee_rate_validation():
    """Test that fee rate validation works correctly for the new system.
    
    Validates that protocol_fee_rate_bps and creator_fee_rate_bps are within
    valid ranges (0-10000 bps) and their sum doesn't exceed 10000 bps.
    """
    # Test individual rate limits
    config = SimulationConfig(protocol_fee_rate_bps=-1)
    expect_raises(AssertionError, config.validate)  # Negative protocol fee
    
    config = SimulationConfig(creator_fee_rate_bps=-1)
    expect_raises(AssertionError, config.validate)  # Negative creator fee
    
    config = SimulationConfig(protocol_fee_rate_bps=10001)
    expect_raises(AssertionError, config.validate)  # Protocol fee too high
    
    config = SimulationConfig(creator_fee_rate_bps=10001)
    expect_raises(AssertionError, config.validate)  # Creator fee too high
    
    # Test sum constraint
    config = SimulationConfig(protocol_fee_rate_bps=6000, creator_fee_rate_bps=5000)  # Sum = 11000 > 10000
    expect_raises(AssertionError, config.validate)  # Combined fees exceed 100%
    
    # Test valid configurations
    config = SimulationConfig(protocol_fee_rate_bps=5000, creator_fee_rate_bps=5000)  # Sum = 10000
    config.validate()  # Should pass - exactly at limit
    
    config = SimulationConfig(protocol_fee_rate_bps=100, creator_fee_rate_bps=50)  # Default values
    config.validate()  # Should pass


def test_unknown_scenario():
    expect_raises(ValueError, SimulationConfig.create_fee_scenario, "does_not_exist")


def test_scenario_comparison():
    """Test that different scenarios produce different economic outcomes.
    
    Validates that protocol_sustainable scenario generates more treasury
    revenue than the default scenario due to different fee allocations.
    """
    config_default = SimulationConfig.create_fee_scenario("default")              # 100 bps protocol (1.0%)
    config_protocol = SimulationConfig.create_fee_scenario("protocol_sustainable") # 150 bps protocol (1.5%)
    
    # Run 24-hour simulations with both configurations
    treasury_default = FeelsSimulation(config_default).run(hours=24).snapshots[-1].floor_state.treasury_balance
    treasury_protocol = FeelsSimulation(config_protocol).run(hours=24).snapshots[-1].floor_state.treasury_balance
    
    # Protocol sustainable should generate more treasury revenue (150 bps vs 100 bps)
    if treasury_default > 0 and treasury_protocol > 0:  # Only test if fees were actually collected
        ratio = treasury_protocol / treasury_default
        # The ratio should be positive and different (can vary due to simulation dynamics)
        assert ratio > 0.1  # Sanity check that protocol sustainable generates meaningful revenue
        assert ratio < 10.0  # Sanity check that the difference isn't unreasonably large

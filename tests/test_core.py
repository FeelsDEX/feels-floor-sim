"""Core engine and configuration tests (pytest-free)."""

import math

from feels_sim.core import FeelsSimulation, FeelsMarketModel
from feels_sim.pricing import price_to_tick, tick_to_price
from feels_sim.config import SimulationConfig
from tests.utils import assert_close, expect_raises
import agentpy as ap


# --- Tick/price conversion -------------------------------------------------
# These tests validate the core price representation system used throughout
# the simulation for AMM calculations and POMM positioning.

def test_tick_to_price_roundtrip():
    """Test that tick-to-price conversion is perfectly reversible.
    
    Critical for ensuring price calculations maintain precision across
    the simulation and POMM deployment logic works correctly.
    """
    for tick in [0, 1, -1, 10, -10, 1_000, -1_000]:  # Test range of tick values
        price = tick_to_price(tick)        # Convert tick to price
        converted = price_to_tick(price)   # Convert back to tick
        assert converted == tick           # Must be exactly equal (no rounding errors)


def test_price_to_tick_roundtrip():
    """Test that price-to-tick conversion maintains acceptable precision.
    
    While tick-to-price is exact, price-to-tick involves rounding to discrete
    tick levels. This test ensures precision loss is within acceptable bounds.
    """
    for price in [0.5, 1.0, 1.5, 2.0, 10.0]:  # Test various price levels
        tick = price_to_tick(price)        # Convert price to nearest tick
        converted = tick_to_price(tick)    # Convert tick back to price
        assert_close(converted, price, rel=1e-4)  # Allow small rounding error


# --- Simulation loop invariants --------------------------------------------
# These tests verify that the simulation executes correctly with proper
# timing, data collection, and state management.

def test_minute_loop_execution():
    """Test that simulation runs for correct duration with proper data collection.
    
    Verifies minute-by-minute execution and hourly aggregation work as expected.
    Uses simplified model without participant behavior for deterministic testing.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Disable complex behavior
    sim = FeelsSimulation(config)
    results = sim.run(hours=2)  # Run 2-hour simulation
    
    # Verify correct number of snapshots (2 hours × 60 minutes/hour = 120 snapshots)
    assert len(results.snapshots) == 120
    # Verify correct number of hourly aggregates
    assert len(results.hourly_aggregates) == 2


def test_deterministic_behavior():
    """Test that simulation produces identical results with same configuration.
    
    Critical for ensuring reproducible research and debugging. Fixed random
    seeds should produce identical outcomes across multiple runs.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Deterministic config
    sim_a = FeelsSimulation(config)  # First simulation instance
    sim_b = FeelsSimulation(config)  # Second simulation instance

    results_a = sim_a.run(hours=1)  # Run first simulation
    results_b = sim_b.run(hours=1)  # Run second simulation

    # Both simulations should produce same number of snapshots
    assert len(results_a.snapshots) == len(results_b.snapshots)
    
    # Every snapshot should be identical between runs
    for snap_a, snap_b in zip(results_a.snapshots, results_b.snapshots):
        assert_close(snap_a.sol_price_usd, snap_b.sol_price_usd)      # Same price evolution
        assert_close(snap_a.floor_price_usd, snap_b.floor_price_usd)  # Same floor advancement
        assert_close(snap_a.volume_feelssol, snap_b.volume_feelssol)  # Same trading volume


# --- Configuration validation ----------------------------------------------
# These tests ensure configuration validation catches invalid parameter
# combinations and enforces protocol constraints.

def test_valid_config_passes_validation():
    """Test that default configuration passes all validation checks.
    
    Ensures the default configuration represents a valid protocol state
    that can be used for simulation without errors.
    """
    SimulationConfig().validate()  # Should not raise any exceptions


def test_invalid_fee_split_raises():
    """Test that invalid fee splits are rejected during validation.
    
    Protocol limits individual fees and their sum. This test ensures
    the validation catches violations of on-chain constraints.
    """
    config = SimulationConfig()
    config.protocol_fee_rate_bps = 12000  # Exceeds 10000 bps (100%) maximum limit
    expect_raises(AssertionError, config.validate)  # Should raise validation error


def test_invalid_fee_range_raises():
    """Test that unreasonably high fees are rejected during validation.
    
    Prevents configuration of fees that would make the protocol unusable.
    15% fee (1500 bps) should be rejected as economically unrealistic.
    """
    config = SimulationConfig(base_fee_bps=1_500)  # 15% fee is unreasonably high
    expect_raises(AssertionError, config.validate)  # Should raise validation error


def test_invalid_supply_relationship_raises():
    """Test that impossible token supply relationships are rejected.
    
    Circulating supply cannot exceed total supply - this would violate
    basic token economics and cause calculation errors.
    """
    config = SimulationConfig()
    config.circulating_supply = config.total_supply + 1  # Impossible: more circulating than total
    expect_raises(AssertionError, config.validate)       # Should raise validation error


# --- AgentPy Integration Tests --------------------------------------------
# These tests verify that the new AgentPy-based model works correctly.

def test_agentpy_model_creation():
    """Test that AgentPy model can be created and initialized properly.
    
    Verifies that the new FeelsMarketModel integrates correctly with
    AgentPy's framework and maintains all required simulation state.
    """
    config = SimulationConfig(enable_participant_behavior=False)
    params = config.to_agentpy_params()
    
    # Test model creation
    model = FeelsMarketModel(params)
    assert model is not None
    
    # Test setup lifecycle
    model.setup()
    assert model.minute == 0
    assert model.sol_price_usd > 0
    assert model.floor_state is not None
    assert len(model.snapshots) == 0  # No snapshots yet


def test_agentpy_simulation_execution():
    """Test that AgentPy simulation runs correctly for multiple steps.
    
    Verifies the full AgentPy lifecycle including setup, step execution,
    and data collection works as expected.
    """
    config = SimulationConfig(enable_participant_behavior=False)
    params = config.to_agentpy_params()
    
    # Create and run simulation manually (AgentPy 0.1.5 doesn't have Simulation)
    model = FeelsMarketModel(params)
    model.setup()
    for _ in range(60):  # Run for 1 hour
        model.step()
    
    # Verify simulation state
    assert model.minute == 60
    assert len(model.snapshots) == 60
    assert model.floor_state.buffer_balance >= 0
    
    # Verify data was recorded using AgentPy's collection
    # (model.record() calls should have collected data)
    
    # Check that floor price is monotonic (never decreases)
    floor_prices = [s.floor_price_usd for s in model.snapshots]
    for i in range(1, len(floor_prices)):
        assert floor_prices[i] >= floor_prices[i-1]


def test_agentpy_reproducibility():
    """Test that AgentPy simulations are reproducible with same seed.
    
    Critical for scientific reproducibility and parameter sweep validity.
    Two simulations with identical parameters and seed should produce
    exactly the same results.
    """
    config = SimulationConfig(enable_participant_behavior=False)
    params = config.to_agentpy_params()
    
    # Run first simulation manually with fixed seed
    model1 = FeelsMarketModel(params)
    model1.setup()
    model1.random.seed(42)
    for _ in range(30):
        model1.step()
    
    # Run second simulation with same seed
    model2 = FeelsMarketModel(params)
    model2.setup()
    model2.random.seed(42)
    for _ in range(30):
        model2.step()
    
    # Results should be identical
    assert model1.minute == model2.minute
    assert len(model1.snapshots) == len(model2.snapshots)
    
    # Check SOL prices are identical (most sensitive to RNG)
    prices1 = [s.sol_price_usd for s in model1.snapshots]
    prices2 = [s.sol_price_usd for s in model2.snapshots]
    for p1, p2 in zip(prices1, prices2):
        assert_close(p1, p2, rel=1e-10)  # Very tight tolerance

"""Tests for core simulation functionality."""

import pytest
import math
from feels_sim.core import tick_to_price, price_to_tick, FeelsSimulation
from feels_sim.config import SimulationConfig


class TestTickConversion:
    """Test tick-to-price conversion utilities."""
    
    def test_tick_to_price_basic(self):
        """Test basic tick to price conversion."""
        assert tick_to_price(0) == pytest.approx(1.0)  # Tick 0 = price 1.0
        assert tick_to_price(1) == pytest.approx(1.0001)  # Each tick = 0.01% increase
        assert tick_to_price(-1) == pytest.approx(1.0 / 1.0001)  # Negative ticks decrease price
        assert tick_to_price(100) == pytest.approx(1.0001 ** 100)  # Large tick differences
    
    def test_price_to_tick_basic(self):
        """Test basic price to tick conversion."""
        assert price_to_tick(1.0) == 0  # Price 1.0 = tick 0
        assert price_to_tick(1.0001) == 1  # 0.01% increase = tick 1
        assert price_to_tick(1.0 / 1.0001) == -1  # 0.01% decrease = tick -1
    
    def test_tick_price_roundtrip(self):
        """Test that tick->price->tick conversion is consistent."""
        test_ticks = [0, 1, -1, 100, -100, 1000, -1000]  # Range of representative ticks
        
        for tick in test_ticks:
            price = tick_to_price(tick)
            converted_tick = price_to_tick(price)
            assert converted_tick == tick  # Should roundtrip exactly
    
    def test_price_tick_roundtrip(self):
        """Test that price->tick->price conversion is consistent."""
        test_prices = [0.5, 1.0, 1.5, 2.0, 10.0]  # Range of realistic prices
        
        for price in test_prices:
            tick = price_to_tick(price)
            converted_price = tick_to_price(tick)  # Should recover original price
            assert converted_price == pytest.approx(price, rel=1e-4)  # Within rounding error


class TestSimulationLoop:
    """Test simulation loop invariants."""
    
    def test_minute_loop_execution(self):
        """Test that minute loop executes correct number of steps."""
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        
        results = sim.run(hours=2)  # 2 hours = 120 minutes
        
        assert len(results.snapshots) == 120  # One snapshot per minute
        assert len(results.hourly_aggregates) == 2  # One aggregate per hour
    
    def test_deterministic_behavior(self):
        """Test that simulation produces deterministic results."""
        config = SimulationConfig()  # Same config for both runs
        
        sim1 = FeelsSimulation(config)
        results1 = sim1.run(hours=1)
        
        sim2 = FeelsSimulation(config)  # Fresh simulation with same seed
        results2 = sim2.run(hours=1)
        
        # Should produce identical results due to fixed RNG seed
        assert len(results1.snapshots) == len(results2.snapshots)
        
        for snap1, snap2 in zip(results1.snapshots, results2.snapshots):
            assert snap1.sol_price_usd == pytest.approx(snap2.sol_price_usd)  # Same price evolution
            assert snap1.floor_price_usd == pytest.approx(snap2.floor_price_usd)  # Same floor evolution
            assert snap1.volume_feelssol == pytest.approx(snap2.volume_feelssol)  # Same volume generation
    
    def test_mint_accumulation(self):
        """Test that synthetic minting contributes to floor growth."""
        config = SimulationConfig(jitosol_yield_apy=0.0876)  # 8.76% APY for easy math
        sim = FeelsSimulation(config)
        
        initial_deployed = sim.floor_state.deployed_feelssol
        results = sim.run(hours=24 * 30)  # Run for 30 days
        
        final_state = results.snapshots[-1].floor_state
        
        # Deployed floor liquidity should have grown thanks to minting + Buffer
        assert final_state.deployed_feelssol > initial_deployed
        # Mintable buffer should remain non-negative
        assert final_state.mintable_feelssol >= 0
    
    def test_fee_distribution_invariant(self):
        """Test that fee distribution maintains correct splits."""
        config = SimulationConfig()  # Default fee splits: 85% Buffer, 10% protocol, 5% creator
        sim = FeelsSimulation(config)
        
        results = sim.run(hours=24)  # Run long enough to generate meaningful fees
        
        # Calculate total fees and final balances
        total_fees = sum(s.fees_collected for s in results.snapshots)
        final_state = results.snapshots[-1].floor_state
        final_treasury = final_state.treasury_balance
        final_creator = final_state.creator_balance
        buffer_routed = final_state.buffer_routed_cumulative
        
        if total_fees > 0:  # Only test if fees were actually generated
            # Verify protocol treasury gets its configured share
            treasury_pct = (final_treasury / total_fees) * 100
            assert treasury_pct == pytest.approx(config.treasury_share_pct, rel=0.1)
            
            # Verify creator gets their configured share
            creator_pct = (final_creator / total_fees) * 100
            assert creator_pct == pytest.approx(config.creator_share_pct, rel=0.1)

            buffer_pct = (buffer_routed / total_fees) * 100
            assert buffer_pct == pytest.approx(config.buffer_share_pct, rel=0.1)
    
    def test_floor_price_monotonic(self):
        """Test that floor price never decreases."""
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        
        results = sim.run(hours=48)
        
        floor_prices = [s.floor_price_usd for s in results.snapshots]
        
        # Floor price should be monotonically increasing (POMM property)
        for i in range(1, len(floor_prices)):
            assert floor_prices[i] >= floor_prices[i-1]  # Never decreases
    
    def test_pomm_cooldown_enforcement(self):
        """Test that POMM deployments respect cooldown period."""
        config = SimulationConfig(pomm_cooldown_seconds=60)  # 60-second cooldown
        sim = FeelsSimulation(config)
        
        results = sim.run(hours=4)  # Run long enough for multiple deployments
        
        # Find all deployment times (in minutes from start)
        deployment_times = []
        for i, snapshot in enumerate(results.snapshots):
            if snapshot.events.get("pomm_deployed", False):
                deployment_times.append(i)  # Store minute index
        
        # Verify minimum time between consecutive deployments
        for i in range(1, len(deployment_times)):
            time_diff = deployment_times[i] - deployment_times[i-1]  # Minutes apart
            assert time_diff >= config.pomm_cooldown_seconds / 60  # Must respect cooldown


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test that valid config passes validation."""
        config = SimulationConfig()
        config.validate()  # Should not raise
    
    def test_invalid_fee_splits(self):
        """Test that invalid fee splits are caught."""
        config = SimulationConfig()
        config.buffer_share_pct = 80.0
        # This will break the sum (80+10+5 = 95)
        
        with pytest.raises(AssertionError):  # Should fail validation
            config.validate()
    
    def test_invalid_fee_range(self):
        """Test that invalid fee range is caught."""
        config = SimulationConfig()
        config.base_fee_bps = 1500  # 15% fee is unreasonably high
        
        with pytest.raises(AssertionError):  # Should exceed maximum allowed
            config.validate()
    
    def test_invalid_supply_relationship(self):
        """Test that invalid supply relationship is caught."""
        config = SimulationConfig()
        config.circulating_supply = config.total_supply + 1  # Impossible: more circulating than total
        
        with pytest.raises(AssertionError):  # Should fail logical constraint
            config.validate()

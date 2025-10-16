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


class TestPhase2Implementation:
    """Test Phase 2: Fee Distribution & Funding Pipeline."""
    
    def test_fee_routing_stability(self):
        """Test that fee splits remain stable regardless of base fee."""
        fees_to_test = [10, 30, 50, 100]  # Different base fee levels
        
        for base_fee in fees_to_test:
            config = SimulationConfig(base_fee_bps=base_fee)
            sim = FeelsSimulation(config)
            results = sim.run(hours=24)
            
            total_fees = sum(s.fees_collected for s in results.snapshots)
            final_state = results.snapshots[-1].floor_state
            
            if total_fees > 0:
                buffer_pct = (final_state.buffer_routed_cumulative / total_fees) * 100
                treasury_pct = (final_state.treasury_balance / total_fees) * 100
                creator_pct = (final_state.creator_balance / total_fees) * 100
                
                assert buffer_pct == pytest.approx(85.0, rel=0.01)
                assert treasury_pct == pytest.approx(10.0, rel=0.01) 
                assert creator_pct == pytest.approx(5.0, rel=0.01)
    
    def test_buffer_volume_correlation(self):
        """Test that Buffer growth tracks trading volume."""
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=48)
        
        total_volume = sum(s.volume_feelssol for s in results.snapshots)
        buffer_accumulated = results.snapshots[-1].floor_state.buffer_routed_cumulative
        expected_buffer = total_volume * (config.base_fee_bps / 10000.0) * (config.buffer_share_pct / 100.0)
        
        assert buffer_accumulated == pytest.approx(expected_buffer, rel=0.01)
    
    def test_pomm_deployment_thresholds(self):
        """Test automatic POMM deployment when Buffer thresholds are met."""
        config = SimulationConfig(pomm_threshold_tokens=50.0)  # Lower threshold for testing
        sim = FeelsSimulation(config)
        results = sim.run(hours=72)
        
        # Should have deployments when buffer + mintable > threshold
        deployments = [s for s in results.snapshots if s.events.get("pomm_deployed", False)]
        assert len(deployments) > 0, "Should have POMM deployments when thresholds are met"
        
        # Verify cooldown enforcement
        if len(deployments) > 1:
            deployment_times = [s.timestamp for s in deployments]
            for i in range(1, len(deployment_times)):
                time_diff = deployment_times[i] - deployment_times[i-1]
                assert time_diff >= config.pomm_cooldown_seconds / 60
    
    def test_synthetic_minting_drift(self):
        """Test that minting matches configured 7% APR drift."""
        config = SimulationConfig(jitosol_yield_apy=0.07)
        sim = FeelsSimulation(config)
        results = sim.run(hours=24 * 30)  # 30 days
        
        final_mint = results.snapshots[-1].floor_state.mint_cumulative
        days_elapsed = len(results.snapshots) / (24 * 60)
        expected_mint = config.total_supply * config.jitosol_yield_apy * (days_elapsed / 365.25)
        
        assert final_mint == pytest.approx(expected_mint, rel=0.01)


class TestPhase3Implementation:
    """Test Phase 3: Participant Behavior & Calibration Inputs."""
    
    def test_participant_behavior_integration(self):
        """Test that participant behavior affects trading volume."""
        # Test with participant behavior enabled
        config_with_participants = SimulationConfig(enable_participant_behavior=True)
        sim_with = FeelsSimulation(config_with_participants)
        results_with = sim_with.run(hours=12)
        
        # Test with participant behavior disabled
        config_without_participants = SimulationConfig(enable_participant_behavior=False)
        sim_without = FeelsSimulation(config_without_participants)
        results_without = sim_without.run(hours=12)
        
        volume_with = sum(s.volume_feelssol for s in results_with.snapshots)
        volume_without = sum(s.volume_feelssol for s in results_without.snapshots)
        
        # Volumes should be different when participant behavior is enabled
        assert volume_with != volume_without
        # Participant behavior should generally increase volume
        assert volume_with > volume_without * 0.5  # At least some volume difference
    
    def test_fee_elasticity_behavior(self):
        """Test that participants respond to fee changes."""
        from feels_sim.participants import ParticipantConfig
        
        # Create config with high fee sensitivity
        sensitive_config = ParticipantConfig(
            retail_fee_sensitivity=3.0,
            algo_fee_sensitivity=1.5
        )
        
        # Test high vs low fees
        high_fee_config = SimulationConfig(base_fee_bps=60, participant_config=sensitive_config)
        low_fee_config = SimulationConfig(base_fee_bps=20, participant_config=sensitive_config)
        
        high_fee_sim = FeelsSimulation(high_fee_config)
        low_fee_sim = FeelsSimulation(low_fee_config)
        
        high_results = high_fee_sim.run(hours=24)
        low_results = low_fee_sim.run(hours=24)
        
        high_volume = sum(s.volume_feelssol for s in high_results.snapshots)
        low_volume = sum(s.volume_feelssol for s in low_results.snapshots)
        
        # Lower fees should lead to higher volume (fee elasticity)
        assert low_volume > high_volume, f"Expected low fee volume ({low_volume}) > high fee volume ({high_volume})"
    
    def test_calibration_file_loading(self):
        """Test loading configuration from calibration files."""
        # This test assumes the baseline config file exists
        try:
            config = SimulationConfig.from_calibration_file('experiments/configs/params_baseline.json')
            assert config.enable_participant_behavior == True
            assert config.participant_config is not None
            assert config.participant_config.retail_count > 0
            assert config.participant_config.algo_count > 0
            
            # Test with overrides
            overrides = {
                'simulation_config': {'base_fee_bps': 40},
                'participant_config': {'retail_count': 50, 'algo_count': 8}
            }
            config_override = SimulationConfig.from_calibration_file(
                'experiments/configs/params_baseline.json', 
                overrides=overrides
            )
            assert config_override.base_fee_bps == 40
            assert config_override.participant_config.retail_count == 50
            assert config_override.participant_config.algo_count == 8
            
        except FileNotFoundError:
            # Skip test if calibration file doesn't exist
            pytest.skip("Calibration file not found")
    
    def test_participant_metrics_collection(self):
        """Test that participant metrics are collected in hourly aggregates."""
        config = SimulationConfig(enable_participant_behavior=True)
        sim = FeelsSimulation(config)
        results = sim.run(hours=3)
        
        # Should have participant metrics in hourly aggregates
        assert len(results.hourly_aggregates) >= 2
        for hour_data in results.hourly_aggregates:
            assert 'participant_metrics' in hour_data
            assert 'lp_positions' in hour_data
            assert 'lp_fees_earned' in hour_data
            
            metrics = hour_data['participant_metrics']
            assert 'total_participants' in metrics
            assert metrics['total_participants'] > 0


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

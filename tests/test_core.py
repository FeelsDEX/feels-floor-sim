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
        config = SimulationConfig()  # Default fee splits: 98.5% Buffer, 1% protocol, 0.5% creator
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
                
                assert buffer_pct == pytest.approx(98.5, rel=0.01)
                assert treasury_pct == pytest.approx(1.0, rel=0.01) 
                assert creator_pct == pytest.approx(0.5, rel=0.01)
    
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
        config.treasury_share_pct = 15.0  # Exceeds 10% protocol constraint
        
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


class TestFeeScenarios:
    """Test fee split scenario functionality."""
    
    def test_current_default_scenario(self):
        """Test current default fee scenario."""
        config = SimulationConfig.create_fee_scenario("current_default")
        config.validate()  # Should not raise
        
        assert config.treasury_share_pct == 1.0
        assert config.creator_share_pct == 0.5
        assert config.buffer_share_pct == 98.5
    
    def test_protocol_sustainable_scenario(self):
        """Test protocol sustainable fee scenario."""
        config = SimulationConfig.create_fee_scenario("protocol_sustainable")
        config.validate()  # Should not raise
        
        assert config.treasury_share_pct == 7.0
        assert config.creator_share_pct == 3.0
        assert config.buffer_share_pct == 90.0
    
    def test_creator_incentive_scenario(self):
        """Test creator incentive fee scenario."""
        config = SimulationConfig.create_fee_scenario("creator_incentive")
        config.validate()  # Should not raise
        
        assert config.treasury_share_pct == 5.0
        assert config.creator_share_pct == 5.0
        assert config.buffer_share_pct == 90.0
    
    def test_balanced_growth_scenario(self):
        """Test balanced growth fee scenario."""
        config = SimulationConfig.create_fee_scenario("balanced_growth")
        config.validate()  # Should not raise
        
        assert config.treasury_share_pct == 5.0
        assert config.creator_share_pct == 3.0
        assert config.buffer_share_pct == 92.0
    
    def test_maximum_protocol_scenario(self):
        """Test maximum protocol scenario."""
        config = SimulationConfig.create_fee_scenario("maximum_protocol")
        config.validate()  # Should not raise
        
        assert config.treasury_share_pct == 8.0
        assert config.creator_share_pct == 2.0
        assert config.buffer_share_pct == 90.0
    
    def test_invalid_scenario_fails_validation(self):
        """Test that invalid scenario fails validation."""
        config = SimulationConfig.create_fee_scenario("invalid_high_protocol")
        
        with pytest.raises(AssertionError):  # Should fail due to 15% treasury > 10% limit
            config.validate()
    
    def test_unknown_scenario_raises_error(self):
        """Test that unknown scenarios raise ValueError."""
        with pytest.raises(ValueError, match="Unknown fee scenario"):
            SimulationConfig.create_fee_scenario("nonexistent_scenario")
    
    def test_scenario_with_overrides(self):
        """Test scenario creation with parameter overrides."""
        config = SimulationConfig.create_fee_scenario(
            "current_default", 
            base_fee_bps=50,
            sol_volatility_daily=0.08
        )
        config.validate()  # Should not raise
        
        # Should have scenario defaults
        assert config.treasury_share_pct == 1.0
        assert config.creator_share_pct == 0.5
        
        # Should have overridden values
        assert config.base_fee_bps == 50
        assert config.sol_volatility_daily == 0.08
    
    def test_different_scenarios_produce_different_results(self):
        """Test that different fee scenarios produce measurably different outcomes."""
        # Test current default vs protocol sustainable
        config_default = SimulationConfig.create_fee_scenario("current_default")
        config_sustainable = SimulationConfig.create_fee_scenario("protocol_sustainable")
        
        sim_default = FeelsSimulation(config_default)
        sim_sustainable = FeelsSimulation(config_sustainable)
        
        results_default = sim_default.run(hours=24)
        results_sustainable = sim_sustainable.run(hours=24)
        
        # Should have different treasury balances due to different fee splits
        treasury_default = results_default.snapshots[-1].floor_state.treasury_balance
        treasury_sustainable = results_sustainable.snapshots[-1].floor_state.treasury_balance
        
        # Sustainable scenario should accumulate much more treasury
        assert treasury_sustainable > treasury_default * 5  # 10% vs 1% = 10x difference expected


class TestPhase4Implementation:
    """Test Phase 4: Metrics, Aggregates & Reporting."""
    
    def test_metrics_collector_basic_functionality(self):
        """Test basic MetricsCollector functionality."""
        from feels_sim.metrics import MetricsCollector
        
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=6)
        
        collector = MetricsCollector()
        for snapshot in results.snapshots:
            collector.add_snapshot(snapshot)
        
        # Test basic metric calculations
        floor_growth = collector.calculate_floor_growth_rate()
        floor_ratio = collector.calculate_floor_to_market_ratio()
        pomm_count = collector.calculate_pomm_deployment_count()
        total_volume = collector.calculate_total_volume()
        total_fees = collector.calculate_total_fees()
        
        assert isinstance(floor_growth, float)
        assert isinstance(floor_ratio, float)
        assert isinstance(pomm_count, int)
        assert isinstance(total_volume, float)
        assert isinstance(total_fees, float)
        
        assert floor_ratio >= 0.0
        assert pomm_count >= 0
        assert total_volume >= 0.0
        assert total_fees >= 0.0
    
    def test_hourly_aggregates_derivation(self):
        """Test derivation of hourly aggregates from minute snapshots."""
        from feels_sim.metrics import MetricsCollector
        
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=3)  # 3 hours = 180 minutes
        
        collector = MetricsCollector()
        for snapshot in results.snapshots:
            collector.add_snapshot(snapshot)
        
        hourly_data = collector.derive_hourly_aggregates()
        
        assert len(hourly_data) == 3  # 3 hours of data
        
        for hour_data in hourly_data:
            assert 'hour' in hour_data
            assert 'volume_feelssol' in hour_data
            assert 'fees_collected' in hour_data
            assert 'buffer_routed' in hour_data
            assert 'mint_amount' in hour_data
            assert 'pomm_deployments' in hour_data
            assert 'avg_sol_price' in hour_data
            assert 'avg_floor_price' in hour_data
            assert 'floor_delta' in hour_data
            
            # Values should be non-negative
            assert hour_data['volume_feelssol'] >= 0
            assert hour_data['fees_collected'] >= 0
            assert hour_data['buffer_routed'] >= 0
            assert hour_data['mint_amount'] >= 0
            assert hour_data['pomm_deployments'] >= 0
    
    def test_daily_weekly_aggregates(self):
        """Test derivation of daily and weekly aggregates."""
        from feels_sim.metrics import MetricsCollector
        
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=168)  # 1 week = 168 hours
        
        collector = MetricsCollector()
        for snapshot in results.snapshots:
            collector.add_snapshot(snapshot)
        
        daily_data = collector.derive_daily_aggregates()
        weekly_data = collector.derive_weekly_aggregates()
        
        assert len(daily_data) == 7  # 7 days
        assert len(weekly_data) == 1  # 1 week
        
        # Verify daily aggregation sums to weekly
        daily_volume_sum = sum(d['volume_feelssol'] for d in daily_data)
        weekly_volume = weekly_data[0]['volume_feelssol']
        
        assert daily_volume_sum == pytest.approx(weekly_volume, rel=0.01)
    
    def test_advanced_metrics_calculation(self):
        """Test advanced metrics like LP yield and protocol efficiency."""
        from feels_sim.metrics import MetricsCollector
        
        config = SimulationConfig(enable_participant_behavior=True)
        sim = FeelsSimulation(config)
        results = sim.run(hours=24)
        
        collector = MetricsCollector()
        for snapshot in results.snapshots:
            collector.add_snapshot(snapshot)
        
        # Add hourly aggregates for LP yield calculation
        for aggregate in results.hourly_aggregates:
            collector.add_hourly_aggregate(aggregate)
        
        lp_yield = collector.calculate_lp_yield_apy()
        efficiency = collector.calculate_protocol_efficiency()
        utilization = collector.calculate_buffer_utilization()
        
        assert isinstance(lp_yield, float)
        assert isinstance(efficiency, float)
        assert isinstance(utilization, float)
        
        assert lp_yield >= 0.0
        assert utilization >= 0.0 and utilization <= 1.0
    
    def test_analysis_results_integration(self):
        """Test the analyze_results function with all new metrics."""
        from feels_sim.metrics import analyze_results
        
        config = SimulationConfig(enable_participant_behavior=True)
        sim = FeelsSimulation(config)
        results = sim.run(hours=12)
        
        analysis = analyze_results(results)
        
        # Verify all expected metrics are present
        expected_keys = [
            'floor_growth_rate_annual', 'avg_floor_to_market_ratio',
            'pomm_deployments', 'total_volume', 'total_fees',
            'lp_yield_apy', 'protocol_efficiency', 'buffer_utilization',
            'final_treasury_balance', 'final_buffer_balance',
            'simulation_hours', 'initial_floor_price', 'final_floor_price'
        ]
        
        for key in expected_keys:
            assert key in analysis, f"Missing key: {key}"
            assert isinstance(analysis[key], (int, float)), f"Invalid type for {key}"
    
    def test_floor_ratio_statistics(self):
        """Test floor/market ratio statistical analysis."""
        from feels_sim.metrics import calculate_floor_floor_ratio_stats
        
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=6)
        
        stats = calculate_floor_floor_ratio_stats(results)
        
        expected_stats = ['mean_floor_ratio', 'median_floor_ratio', 'std_floor_ratio',
                         'min_floor_ratio', 'max_floor_ratio', 'p25_floor_ratio', 'p75_floor_ratio']
        
        for stat in expected_stats:
            assert stat in stats
            assert isinstance(stats[stat], float)
            assert stats[stat] >= 0.0
        
        # Logical checks
        assert stats['min_floor_ratio'] <= stats['median_floor_ratio'] <= stats['max_floor_ratio']
        assert stats['p25_floor_ratio'] <= stats['median_floor_ratio'] <= stats['p75_floor_ratio']
    
    def test_pomm_efficiency_metrics(self):
        """Test POMM deployment efficiency analysis."""
        from feels_sim.metrics import calculate_pomm_efficiency_metrics
        
        config = SimulationConfig(pomm_threshold_tokens=25.0)  # Lower threshold for testing
        sim = FeelsSimulation(config)
        results = sim.run(hours=24)
        
        pomm_metrics = calculate_pomm_efficiency_metrics(results)
        
        expected_metrics = ['pomm_count', 'avg_deployment_size', 'median_deployment_size',
                           'deployment_frequency', 'total_deployed_amount']
        
        for metric in expected_metrics:
            assert metric in pomm_metrics
            assert isinstance(pomm_metrics[metric], (int, float))
            assert pomm_metrics[metric] >= 0
    
    def test_volume_elasticity_calculation(self):
        """Test fee elasticity calculation between different fee scenarios."""
        from feels_sim.metrics import calculate_volume_elasticity
        
        # Test the elasticity calculation function with mock data
        # Create simple mock results for testing the calculation logic
        from feels_sim.core import SimulationSnapshot, SimulationResults, FloorFundingState
        
        # Mock low fee results (higher volume)
        low_snapshots = []
        for i in range(60):  # 1 hour of data
            snapshot = SimulationSnapshot(
                timestamp=i,
                sol_price_usd=100.0,
                floor_price_feelssol=0.95,
                floor_price_usd=95.0,
                floor_state=FloorFundingState(),
                volume_feelssol=1000.0,  # Higher volume
                fees_collected=3.0,
                events={'pomm_deployed': False}
            )
            low_snapshots.append(snapshot)
        
        # Mock high fee results (lower volume)
        high_snapshots = []
        for i in range(60):  # 1 hour of data
            snapshot = SimulationSnapshot(
                timestamp=i,
                sol_price_usd=100.0,
                floor_price_feelssol=0.95,
                floor_price_usd=95.0,
                floor_state=FloorFundingState(),
                volume_feelssol=800.0,  # Lower volume
                fees_collected=6.4,
                events={'pomm_deployed': False}
            )
            high_snapshots.append(snapshot)
        
        # Create mock results
        low_results = SimulationResults(low_snapshots, [], SimulationConfig())
        high_results = SimulationResults(high_snapshots, [], SimulationConfig())
        
        elasticity = calculate_volume_elasticity(low_results, high_results)
        
        expected_keys = ['low_fee_volume', 'high_fee_volume', 'volume_change_pct', 'elasticity_observed']
        
        for key in expected_keys:
            assert key in elasticity
        
        # With mock data, should observe elasticity
        assert elasticity['elasticity_observed'] == True
        assert elasticity['low_fee_volume'] > elasticity['high_fee_volume']
        assert elasticity['low_fee_volume'] == 60000.0  # 60 * 1000
        assert elasticity['high_fee_volume'] == 48000.0  # 60 * 800
        assert elasticity['volume_change_pct'] == 0.25  # (60000-48000)/48000
    
    def test_metrics_export_functionality(self):
        """Test metrics export to file."""
        from feels_sim.metrics import MetricsCollector
        import tempfile
        import json
        import os
        
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=4)
        
        collector = MetricsCollector()
        for snapshot in results.snapshots:
            collector.add_snapshot(snapshot)
        
        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            collector.export_metrics(export_path, format='json')
            
            # Verify file was created and contains expected structure
            assert os.path.exists(export_path)
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert 'summary' in data
            assert 'aggregates' in data
            assert 'hourly' in data['aggregates']
            assert 'daily' in data['aggregates']
            assert 'weekly' in data['aggregates']
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_visualization_functions(self):
        """Test that visualization functions run without errors."""
        from feels_sim.metrics import create_summary_plots, create_detailed_analysis_plots
        import tempfile
        import os
        
        config = SimulationConfig()
        sim = FeelsSimulation(config)
        results = sim.run(hours=6)
        
        # Test summary plots
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            plot_path = f.name
        
        try:
            # Should not raise exceptions
            create_summary_plots(results, save_path=plot_path)
            create_detailed_analysis_plots(results, save_path=plot_path)
            
        except ImportError:
            # Skip if matplotlib not available
            pytest.skip("matplotlib not available")
        finally:
            for path in [plot_path, plot_path.replace('.png', '_detailed.png')]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_summary_report_generation(self):
        """Test markdown summary report generation."""
        from feels_sim.metrics import generate_summary_report
        import tempfile
        import os
        
        config = SimulationConfig(enable_participant_behavior=True)
        sim = FeelsSimulation(config)
        results = sim.run(hours=8)
        
        # Test report generation
        report_content = generate_summary_report(results)
        
        assert isinstance(report_content, str)
        assert len(report_content) > 0
        assert "# Feels Simulation Summary Report" in report_content
        assert "## Simulation Overview" in report_content
        assert "## Key Metrics" in report_content
        
        # Test file export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            report_path = f.name
        
        try:
            generate_summary_report(results, file_path=report_path)
            assert os.path.exists(report_path)
            
            with open(report_path, 'r') as f:
                file_content = f.read()
            
            assert file_content == report_content
            
        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)

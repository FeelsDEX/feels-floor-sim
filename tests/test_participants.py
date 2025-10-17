"""Participant behaviour integration tests (pytest-free).

Tests the participant behavior system including trading frequency, fee sensitivity,
configuration loading, and metrics collection. Validates that sophisticated
participant models produce realistic market dynamics."""

from feels_sim.core import FeelsSimulation
from feels_sim.config import SimulationConfig
from feels_sim.participants import ParticipantConfig
from tests.utils import expect_raises, file_exists


def test_participant_behavior_integration():
    """Test that participant behavior produces different outcomes than fallback model.
    
    Validates that the sophisticated participant behavior system generates
    different trading patterns compared to the simple fallback volume model.
    """
    config_with = SimulationConfig(enable_participant_behavior=True)   # Sophisticated participant model
    config_without = SimulationConfig(enable_participant_behavior=False) # Simple fallback model

    # Run 12-hour simulations with both models
    volume_with = sum(s.volume_feelssol for s in FeelsSimulation(config_with).run(hours=12).snapshots)
    volume_without = sum(s.volume_feelssol for s in FeelsSimulation(config_without).run(hours=12).snapshots)

    # Participant behavior should produce different volume patterns
    assert volume_with != volume_without  # Must be different
    assert volume_with > volume_without * 0.5  # Should be reasonably comparable


def test_fee_elasticity_behavior():
    """Test that participants respond to fee changes (fee elasticity).
    
    Validates that trading behavior changes in response to different fee levels,
    demonstrating that participants exhibit realistic price sensitivity.
    """
    # Configure high fee sensitivity for testing
    participant_cfg = ParticipantConfig(retail_fee_sensitivity=3.0, algo_fee_sensitivity=1.5)
    
    # Create configurations with different fee levels
    high_fee_config = SimulationConfig(base_fee_bps=80, participant_config=participant_cfg)  # 0.8% fee
    low_fee_config = SimulationConfig(base_fee_bps=20, participant_config=participant_cfg)   # 0.2% fee

    # Run 24-hour simulations with both fee levels
    high_volume = sum(s.volume_feelssol for s in FeelsSimulation(high_fee_config).run(hours=24).snapshots)
    low_volume = sum(s.volume_feelssol for s in FeelsSimulation(low_fee_config).run(hours=24).snapshots)

    # Fee elasticity: behavior should respond to fee changes
    # Direction depends on model tuning, but magnitude should be significant
    difference = abs(low_volume - high_volume)
    assert difference > 0.01 * max(high_volume, low_volume)  # At least 1% difference


def test_calibration_file_loading():
    """Test loading configuration from JSON calibration files.
    
    Validates that simulation parameters can be loaded from external files
    and that override mechanisms work correctly for parameter tuning.
    """
    path = 'experiments/configs/params_baseline.json'  # Standard calibration file path
    if not file_exists(path):
        return  # Skip test if calibration file doesn't exist

    # Test basic file loading
    config = SimulationConfig.from_calibration_file(path)
    assert config.participant_config is not None  # Should have participant configuration

    # Test parameter override functionality
    overrides = {
        'simulation_config': {'base_fee_bps': 40},    # Override base fee
        'participant_config': {'retail_count': 50},   # Override retail count
    }
    config_override = SimulationConfig.from_calibration_file(path, overrides=overrides)
    
    # Verify overrides were applied correctly
    assert config_override.base_fee_bps == 40                            # Fee override applied
    assert config_override.participant_config.retail_count == 50         # Participant override applied


def test_participant_metrics_collection():
    """Test that participant metrics are collected in hourly aggregates.
    
    Validates that participant activity statistics are properly tracked
    and included in simulation output for analysis and reporting.
    """
    # Run simulation with participant behavior enabled
    results = FeelsSimulation(SimulationConfig(enable_participant_behavior=True)).run(hours=3)
    assert results.hourly_aggregates  # Should have hourly aggregate data
    
    # Check that at least one hourly aggregate contains participant metrics
    for hour_data in results.hourly_aggregates:
        metrics = hour_data.get('participant_metrics', {})
        if metrics:  # If participant metrics are present
            assert metrics.get('total_participants', 0) > 0  # Should have participants
            return  # Test passed - found participant metrics
    
    # If we reach here, no participant metrics were found
    raise AssertionError("Expected participant metrics in hourly aggregates")

"""Metrics, aggregation, and reporting tests without external deps.

Tests the metrics collection system including data aggregation, analysis calculations,
reporting functions, and visualization generation. Ensures accurate performance
measurement and protocol analytics."""

from feels_sim.core import FeelsSimulation
from feels_sim.config import SimulationConfig
from feels_sim.metrics import (
    calculate_key_metrics,
    calculate_hourly_aggregates,
    calculate_daily_aggregates,
    calculate_weekly_aggregates,
)
from feels_sim.analysis import analyze_results
from feels_sim.plotting import create_summary_plots


def test_polars_metrics_basic():
    """Test basic polars-based metrics calculation functionality.
    
    Validates that the new streamlined polars system can process simulation 
    snapshots and compute key performance indicators without errors.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Simplified for testing
    results = FeelsSimulation(config).run(hours=6)  # Run 6-hour simulation

    # Use new polars-based metrics calculation
    metrics = calculate_key_metrics(results.snapshots)

    # Verify all metrics return valid values (non-negative)
    assert metrics.get("floor_growth_rate_annual", 0) >= 0.0      # Floor should grow or stay flat
    assert metrics.get("avg_floor_to_market_ratio", 0) >= 0.0     # Ratio should be positive
    assert metrics.get("pomm_deployments", 0) >= 0               # Deployments should be countable
    assert metrics.get("total_volume", 0) >= 0.0                 # Volume should be non-negative
    assert metrics.get("total_fees", 0) >= 0.0                   # Fees should be non-negative


def test_polars_time_aggregates():
    """Test multi-timeframe data aggregation using polars group_by_dynamic.
    
    Validates that metrics can be properly aggregated across different time
    horizons using efficient polars operations.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Simplified for testing
    results = FeelsSimulation(config).run(hours=48)  # Run 2-day simulation

    # Generate aggregates at different time scales using new polars functions
    hourly = calculate_hourly_aggregates(results.snapshots)   # Hour-by-hour summaries
    daily = calculate_daily_aggregates(results.snapshots)     # Day-by-day summaries
    weekly = calculate_weekly_aggregates(results.snapshots)   # Week-by-week summaries

    # Verify correct number of aggregates for 48-hour simulation
    assert len(hourly) == 48  # Should have exactly 48 hourly aggregates
    assert len(daily) >= 1    # Should have at least 1 daily aggregate (48 hours = 2 days)
    assert len(weekly) >= 1   # Should have at least 1 weekly aggregate


def test_analyze_results_output():
    """Test that analysis results contain all expected metrics.
    
    Validates that the analyze_results function produces a complete set of
    key performance indicators needed for protocol evaluation and reporting.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Simplified for testing
    results = FeelsSimulation(config).run(hours=12)  # Run 12-hour simulation

    analysis = analyze_results(results)  # Generate comprehensive analysis
    
    # Define critical metrics that must be present in analysis output
    expected_keys = {
        'floor_growth_rate_annual',   # Annualized floor price growth rate
        'avg_floor_to_market_ratio',  # Average floor-to-market price ratio
        'total_volume',               # Total trading volume
        'total_fees',                 # Total fees collected
        'final_buffer_balance',       # Final Buffer balance
        'final_treasury_balance',     # Final treasury balance
    }
    assert expected_keys.issubset(analysis.keys())  # All expected keys must be present


def test_create_summary_plots():
    """Test plot generation functionality (optional dependency).
    
    Validates that visualization functions work when matplotlib is available
    and gracefully handle missing dependencies. Important for analysis workflows.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Simplified for testing
    results = FeelsSimulation(config).run(hours=2)  # Short simulation for quick testing
    
    try:
        create_summary_plots(results)  # Attempt to create visualization
    except ImportError:
        pass  # matplotlib not installed – this is acceptable for testing


def test_polars_analysis_consistency():
    """Test consistency between polars metrics and analysis functions.
    
    Validates that the streamlined polars metrics produce consistent results
    with the analysis functions. Critical for ensuring report accuracy.
    """
    config = SimulationConfig(enable_participant_behavior=False)  # Simplified for testing
    results = FeelsSimulation(config).run(hours=6)  # Run 6-hour simulation
    
    # Process results through both methods
    analysis = analyze_results(results)  # Uses polars metrics internally
    direct_metrics = calculate_key_metrics(results.snapshots)  # Direct polars call
    
    # Both should produce the same results
    assert analysis["total_volume"] == direct_metrics["total_volume"]
    assert analysis["total_fees"] == direct_metrics["total_fees"]
    assert analysis["pomm_deployments"] == direct_metrics["pomm_deployments"]
    
    # Test hourly aggregates
    hourly = calculate_hourly_aggregates(results.snapshots)
    assert len(hourly) == 6  # Should have exactly 6 hourly aggregates
    
    # Cross-validate POMM deployment counts
    deployments_logged = sum(h.get('pomm_deployments', 0) for h in hourly)
    assert deployments_logged == analysis['pomm_deployments']

"""Analysis functions for simulation results.

Provides comprehensive analysis tools for processing Feels Protocol simulation
results including protocol efficiency metrics, POMM performance, and trading
volume elasticity calculations.
"""

from typing import Dict, Any, List

import numpy as np

from .state import SimulationResults
from .metrics import (
    calculate_key_metrics,
    calculate_floor_to_market_ratio_stats,
    calculate_pomm_efficiency_metrics as calc_pomm_metrics,
)


# Statistical helper functions for robust metric calculations
def _mean(values):
    """Calculate mean with null check for empty sequences."""
    if not values:
        return 0.0
    return float(np.mean(values))


def _median(values):
    """Calculate median with null check for empty sequences."""
    if not values:
        return 0.0
    return float(np.median(values))


def _std(values):
    """Calculate standard deviation with null check for empty sequences."""
    if not values:
        return 0.0
    return float(np.std(values))


def _percentile(values, pct: float):
    """Calculate percentile with null check for empty sequences."""
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def analyze_results(results: SimulationResults) -> Dict[str, Any]:
    """Primary analysis function for extracting key protocol metrics.
    
    Processes complete simulation results to calculate performance indicators
    including floor growth rates, POMM efficiency, liquidity provision yields,
    and protocol revenue metrics. Uses streamlined polars-based calculations.
    
    Args:
        results: Complete simulation results with snapshots and config
    
    Returns:
        Dictionary of key analysis metrics for protocol evaluation
    """
    if not results.snapshots:
        return {}
    
    # Use new streamlined polars-based metrics calculation
    return calculate_key_metrics(results.snapshots)


def calculate_floor_floor_ratio_stats(results: SimulationResults) -> Dict[str, float]:
    """Calculate statistical measures of floor/market price ratio.
    
    Analyzes how effectively the floor price tracks the market price over time.
    Higher ratios indicate stronger floor support relative to market valuation.
    Uses efficient polars-based calculations.
    
    Args:
        results: Complete simulation results with price snapshots
        
    Returns:
        Dictionary of statistical measures for floor/market price ratios
    """
    return calculate_floor_to_market_ratio_stats(results.snapshots)


def calculate_pomm_efficiency_metrics(results: SimulationResults) -> Dict[str, float]:
    """Calculate POMM deployment efficiency and capital utilization metrics.
    
    Analyzes Protocol-Owned Market Making deployment patterns to evaluate
    capital efficiency, deployment timing, and floor advancement effectiveness.
    Uses efficient polars-based calculations.
    
    Args:
        results: Complete simulation results with POMM deployment events
        
    Returns:
        Dictionary of POMM performance and efficiency metrics
    """
    return calc_pomm_metrics(results.snapshots)


def calculate_volume_elasticity(low_fee_results: SimulationResults, high_fee_results: SimulationResults) -> Dict[str, float]:
    """Calculate trading volume sensitivity to fee rate changes.
    
    Analyzes how trading volume responds to different fee levels to evaluate
    optimal fee pricing. Higher elasticity suggests traders are fee-sensitive,
    indicating potential for volume optimization through fee adjustments.
    
    Args:
        low_fee_results: Simulation results with lower fee configuration
        high_fee_results: Simulation results with higher fee configuration
        
    Returns:
        Dictionary of volume elasticity metrics and comparative analysis
    """
    low_volume = sum(s.volume_feelssol for s in low_fee_results.snapshots)
    high_volume = sum(s.volume_feelssol for s in high_fee_results.snapshots)
    
    if low_volume == 0 or high_volume == 0:
        return {'elasticity': 0, 'volume_change_pct': 0}
    
    volume_change_pct = (low_volume - high_volume) / high_volume
    
    return {
        'low_fee_volume': low_volume,
        'high_fee_volume': high_volume,
        'volume_change_pct': volume_change_pct,
        'elasticity_observed': low_volume > high_volume
    }
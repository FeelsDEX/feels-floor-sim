"""Polars-based metrics calculation system.

Uses group_by_dynamic and declarative expressions for efficient time-based aggregations.
"""

from typing import List, Dict, Any, Optional
import polars as pl
from .state import SimulationSnapshot, SimulationResults


def snapshots_to_dataframe(snapshots: List[SimulationSnapshot]) -> pl.DataFrame:
    """Convert list of SimulationSnapshot objects to a polars DataFrame.
    
    Creates a structured DataFrame with all simulation data for efficient
    time-series analysis and aggregation using polars expressions.
    """
    if not snapshots:
        return pl.DataFrame()
    
    # Extract all data from snapshots into lists for DataFrame construction
    data = {
        # Core simulation timing and prices
        "timestamp": [s.timestamp for s in snapshots],
        "sol_price_usd": [s.sol_price_usd for s in snapshots],
        "floor_price_feelssol": [s.floor_price_feelssol for s in snapshots],
        "floor_price_usd": [s.floor_price_usd for s in snapshots],
        
        # Trading activity metrics
        "volume_feelssol": [s.volume_feelssol for s in snapshots],
        "fees_collected": [s.fees_collected for s in snapshots],
        "jit_volume_boost": [s.jit_volume_boost for s in snapshots],
        "jit_active": [s.jit_active for s in snapshots],
        "pomm_deployed": [s.events.get("pomm_deployed", False) for s in snapshots],
        
        # Floor funding state tracking
        "treasury_balance": [s.floor_state.treasury_balance for s in snapshots],
        "creator_balance": [s.floor_state.creator_balance for s in snapshots],
        "buffer_balance": [s.floor_state.buffer_balance for s in snapshots],
        "mintable_feelssol": [s.floor_state.mintable_feelssol for s in snapshots],
        "deployed_feelssol": [s.floor_state.deployed_feelssol for s in snapshots],
        "buffer_routed_cumulative": [s.floor_state.buffer_routed_cumulative for s in snapshots],
        "mint_cumulative": [s.floor_state.mint_cumulative for s in snapshots],
        "lp_fee_cumulative": [s.floor_state.lp_fee_cumulative for s in snapshots],
        
        # Price movement tracking for impact analysis
        "start_tick": [s.price_path[0] for s in snapshots],
        "end_tick": [s.price_path[1] for s in snapshots],
    }
    
    # Add participant volume data dynamically based on what's present
    participant_types = set()  # Discover all participant types in data
    for snapshot in snapshots:
        participant_types.update(snapshot.participant_volumes.keys())
    
    # Create volume columns for each participant type found
    for ptype in participant_types:
        data[f"{ptype}_volume"] = [
            s.participant_volumes.get(ptype, 0.0) for s in snapshots
        ]
    
    # Create DataFrame with proper time indexing
    df = pl.DataFrame(data)
    
    # Add derived columns for time-based analysis
    df = df.with_columns([
        # Convert timestamp to datetime-like for time operations
        (pl.col("timestamp") * pl.duration(minutes=1)).alias("datetime"),
        
        # Add hour and day columns for grouping
        (pl.col("timestamp") // 60).alias("hour"),
        (pl.col("timestamp") // (60 * 24)).alias("day"),
        
        # Price change calculations
        (pl.col("sol_price_usd").pct_change()).alias("sol_price_change"),
        (pl.col("floor_price_usd").pct_change()).alias("floor_price_change"),
        
        # Floor to market ratio
        (pl.col("floor_price_usd") / pl.col("sol_price_usd")).alias("floor_to_market_ratio"),
        
        # Tick movements
        (pl.col("end_tick") - pl.col("start_tick")).alias("tick_change"),
    ])
    
    return df


def calculate_twap_and_volatility(df: pl.DataFrame, window_minutes: int = 60) -> pl.DataFrame:
    """Add TWAP and volatility calculations using polars rolling windows.
    
    Replaces the manual PriceHistory class with efficient polars operations.
    """
    if df.is_empty():
        return df
    
    # Add rolling calculations
    df_with_rolling = df.with_columns([
        # TWAP (Time-Weighted Average Price) using rolling mean of end_tick
        pl.col("end_tick").rolling_mean(window_size=window_minutes).alias("twap_tick"),
        
        # Price volatility using rolling standard deviation of tick changes
        pl.col("tick_change").abs().rolling_std(window_size=min(window_minutes, 30)).alias("tick_volatility"),
        
        # Rolling volume averages
        pl.col("volume_feelssol").rolling_mean(window_size=window_minutes).alias("avg_volume"),
        
        # Rolling fee collection
        pl.col("fees_collected").rolling_sum(window_size=window_minutes).alias("hourly_fees"),
    ])
    
    return df_with_rolling


def calculate_key_metrics(snapshots: List[SimulationSnapshot]) -> Dict[str, Any]:
    """Calculate all key simulation metrics using polars operations.
    
    Args:
        snapshots: List of simulation snapshots to analyze
        
    Returns:
        Dictionary containing all key performance metrics
    """
    if not snapshots:
        return {}
    
    # Convert snapshots to polars DataFrame for efficient processing
    df = snapshots_to_dataframe(snapshots)
    
    if df.is_empty():
        return {}
    
    # Calculate basic aggregates using polars expressions
    metrics = {}
    
    # Floor growth rate calculation
    initial_floor = df.select("floor_price_usd").item(0, 0)
    final_floor = df.select("floor_price_usd").item(-1, 0)
    
    if initial_floor > 0 and len(snapshots) > 1:
        minutes_elapsed = len(snapshots)
        years_elapsed = minutes_elapsed / (365.25 * 24 * 60)
        
        if years_elapsed > 0:
            growth_ratio = final_floor / initial_floor
            if years_elapsed < 0.1 or growth_ratio > 1000:
                metrics["floor_growth_rate_annual"] = growth_ratio - 1.0
            else:
                metrics["floor_growth_rate_annual"] = min(growth_ratio ** (1 / years_elapsed) - 1, 10.0)
        else:
            metrics["floor_growth_rate_annual"] = 0.0
    else:
        metrics["floor_growth_rate_annual"] = 0.0
    
    # Aggregate metrics using polars expressions
    aggregates = df.select([
        # Volume and fees
        pl.col("volume_feelssol").sum().alias("total_volume"),
        pl.col("fees_collected").sum().alias("total_fees"),
        
        # POMM metrics
        pl.col("pomm_deployed").sum().alias("pomm_deployments"),
        
        # Floor to market ratio
        (pl.col("floor_price_usd") / pl.col("sol_price_usd").filter(pl.col("sol_price_usd") > 0)).mean().alias("avg_floor_to_market_ratio"),
        
        # JIT metrics
        pl.col("jit_volume_boost").sum().alias("jit_total_volume_boost"),
        pl.col("jit_active").sum().alias("jit_active_minutes"),
        
        # Buffer utilization
        ((pl.col("deployed_feelssol") / (pl.col("buffer_balance") + pl.col("deployed_feelssol")))
         .filter((pl.col("buffer_balance") + pl.col("deployed_feelssol")) > 0)).mean().alias("buffer_utilization"),
    ]).to_dicts()[0]
    
    # Add aggregated metrics to results
    metrics.update(aggregates)
    
    # Final state metrics
    final_snapshot = snapshots[-1]
    metrics.update({
        "final_treasury_balance": final_snapshot.floor_state.treasury_balance,
        "final_buffer_balance": final_snapshot.floor_state.buffer_balance,
        "final_mintable_feelssol": final_snapshot.floor_state.mintable_feelssol,
        "final_deployed_feelssol": final_snapshot.floor_state.deployed_feelssol,
        "buffer_routed_cumulative": final_snapshot.floor_state.buffer_routed_cumulative,
        "mint_cumulative": final_snapshot.floor_state.mint_cumulative,
        "simulation_hours": len(snapshots) / 60,
        "initial_floor_price": snapshots[0].floor_price_usd,
        "final_floor_price": final_snapshot.floor_price_usd,
        "initial_sol_price": snapshots[0].sol_price_usd,
        "final_sol_price": final_snapshot.sol_price_usd,
    })
    
    # Calculate protocol efficiency
    if metrics["total_fees"] > 0:
        floor_advancement = final_floor - initial_floor
        metrics["protocol_efficiency"] = floor_advancement / metrics["total_fees"]
    else:
        metrics["protocol_efficiency"] = 0.0
    
    # Calculate LP yield APY (simplified - would need hourly data for accurate calculation)
    metrics["lp_yield_apy"] = 0.0  # Placeholder - requires more complex calculation
    
    return metrics


def calculate_time_aggregates(snapshots: List[SimulationSnapshot], 
                            timeframe: str = "1h") -> List[Dict[str, Any]]:
    """Calculate time-based aggregates using polars group_by_dynamic.
    
    Uses polars' powerful time-based grouping for efficient aggregation.
    
    Args:
        snapshots: List of simulation snapshots
        timeframe: Time period for aggregation ("1h", "1d", "1w")
        
    Returns:
        List of aggregated metrics for each time period
    """
    if not snapshots:
        return []
    
    df = snapshots_to_dataframe(snapshots)
    
    if df.is_empty():
        return []
    
    # Convert timestamp to proper datetime for group_by_dynamic
    # Use a simpler approach that works with current polars version
    df = df.with_columns([
        (pl.datetime(2024, 1, 1) + pl.col("timestamp") * pl.duration(minutes=1)).alias("datetime")
    ])
    
    # Define aggregation expressions
    agg_exprs = [
        # Sum additive metrics
        pl.col("volume_feelssol").sum().alias("volume_feelssol"),
        pl.col("fees_collected").sum().alias("fees_collected"),
        pl.col("pomm_deployed").sum().alias("pomm_deployments"),
        pl.col("jit_volume_boost").sum().alias("jit_volume_boost"),
        pl.col("jit_active").sum().alias("jit_active_minutes"),
        
        # Average point-in-time metrics
        pl.col("sol_price_usd").mean().alias("avg_sol_price"),
        pl.col("floor_price_usd").mean().alias("avg_floor_price"),
        
        # End-of-period balances
        pl.col("buffer_balance").last().alias("final_buffer_balance"),
        pl.col("treasury_balance").last().alias("final_treasury_balance"),
        pl.col("deployed_feelssol").last().alias("final_deployed_feelssol"),
        
        # Period changes
        (pl.col("floor_price_usd").last() - pl.col("floor_price_usd").first()).alias("floor_delta"),
        (pl.col("buffer_routed_cumulative").last() - pl.col("buffer_routed_cumulative").first()).alias("buffer_routed"),
        (pl.col("mint_cumulative").last() - pl.col("mint_cumulative").first()).alias("mint_amount"),
    ]
    
    # Add participant volume aggregations dynamically
    participant_cols = [col for col in df.columns if col.endswith("_volume") and col != "volume_feelssol"]
    for col in participant_cols:
        agg_exprs.append(pl.col(col).sum().alias(col))
    
    # Use group_by_dynamic for time-based aggregation
    result_df = (df
                .sort("datetime")
                .group_by_dynamic("datetime", every=timeframe, closed="left")
                .agg(agg_exprs)
                .sort("datetime"))
    
    return result_df.to_dicts()


def calculate_hourly_aggregates(snapshots: List[SimulationSnapshot]) -> List[Dict[str, Any]]:
    """Calculate hourly aggregates using group_by_dynamic."""
    return calculate_time_aggregates(snapshots, "1h")


def calculate_daily_aggregates(snapshots: List[SimulationSnapshot]) -> List[Dict[str, Any]]:
    """Calculate daily aggregates using group_by_dynamic."""
    return calculate_time_aggregates(snapshots, "1d")


def calculate_weekly_aggregates(snapshots: List[SimulationSnapshot]) -> List[Dict[str, Any]]:
    """Calculate weekly aggregates using group_by_dynamic."""
    return calculate_time_aggregates(snapshots, "1w")


def calculate_floor_to_market_ratio_stats(snapshots: List[SimulationSnapshot]) -> Dict[str, float]:
    """Calculate statistical measures of floor/market price ratio using polars."""
    if not snapshots:
        return {}
    
    df = snapshots_to_dataframe(snapshots)
    
    if df.is_empty():
        return {}
    
    # Calculate ratio statistics using polars expressions
    stats = (df
            .filter(pl.col("sol_price_usd") > 0)
            .with_columns((pl.col("floor_price_usd") / pl.col("sol_price_usd")).alias("ratio"))
            .select([
                pl.col("ratio").mean().alias("mean_floor_ratio"),
                pl.col("ratio").median().alias("median_floor_ratio"),
                pl.col("ratio").std().alias("std_floor_ratio"),
                pl.col("ratio").min().alias("min_floor_ratio"),
                pl.col("ratio").max().alias("max_floor_ratio"),
                pl.col("ratio").quantile(0.25).alias("p25_floor_ratio"),
                pl.col("ratio").quantile(0.75).alias("p75_floor_ratio"),
            ])
            .to_dicts())
    
    return stats[0] if stats else {}


def calculate_pomm_efficiency_metrics(snapshots: List[SimulationSnapshot]) -> Dict[str, float]:
    """Calculate POMM deployment efficiency metrics using polars."""
    if not snapshots:
        return {}
    
    df = snapshots_to_dataframe(snapshots)
    
    if df.is_empty():
        return {"pomm_count": 0, "avg_deployment_size": 0, "deployment_frequency": 0}
    
    # Find POMM deployment events
    deployment_df = df.filter(pl.col("pomm_deployed") == True)
    
    if deployment_df.is_empty():
        return {"pomm_count": 0, "avg_deployment_size": 0, "deployment_frequency": 0}
    
    # Calculate deployment metrics
    pomm_count = len(deployment_df)
    
    # Estimate deployment sizes by looking at buffer balance changes
    deployment_sizes = []
    for i in range(len(deployment_df)):
        if i > 0:
            current_buffer = deployment_df.item(i, "buffer_balance")
            prev_buffer = deployment_df.item(i-1, "buffer_balance")
            if prev_buffer > current_buffer:
                deployment_sizes.append(prev_buffer - current_buffer)
    
    avg_deployment_size = sum(deployment_sizes) / len(deployment_sizes) if deployment_sizes else 0
    
    # Calculate deployment frequency (deployments per hour)
    hours_elapsed = len(df) / 60
    deployment_frequency = pomm_count / max(hours_elapsed, 0.1)
    
    return {
        "pomm_count": pomm_count,
        "avg_deployment_size": avg_deployment_size,
        "median_deployment_size": sorted(deployment_sizes)[len(deployment_sizes)//2] if deployment_sizes else 0,
        "deployment_frequency": deployment_frequency,
        "total_deployed_amount": sum(deployment_sizes),
    }


def export_metrics_to_file(snapshots: List[SimulationSnapshot], 
                          file_path: Optional[str] = None) -> None:
    """Export comprehensive metrics to JSON file."""
    import json
    import os
    from datetime import datetime
    
    # Calculate all metrics
    metrics_data = {
        'summary': calculate_key_metrics(snapshots),
        'aggregates': {
            'hourly': calculate_hourly_aggregates(snapshots),
            'daily': calculate_daily_aggregates(snapshots),
            'weekly': calculate_weekly_aggregates(snapshots),
        },
        'statistics': {
            'floor_ratio_stats': calculate_floor_to_market_ratio_stats(snapshots),
            'pomm_efficiency': calculate_pomm_efficiency_metrics(snapshots),
        }
    }
    
    if file_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Metrics exported to {file_path}")
    else:
        # Default to experiments/outputs/data/ directory
        output_dir = "experiments/outputs/data"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/polars_metrics_{timestamp}.json"
        
        with open(default_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Metrics exported to {default_path}")
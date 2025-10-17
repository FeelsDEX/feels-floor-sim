"""Pricing utilities for Feels simulation including tick conversions and JIT liquidity."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import polars as pl

from .config import SimulationConfig


# Tick-to-price conversion utilities (Uniswap V3 compatible)
# These functions map between human-readable prices and tick-based price levels
# used by the AMM. Each tick represents a 0.01% price change.

def tick_to_price(tick: int) -> float:
    """Convert tick to price using 1.0001^tick formula.
    
    Tick 0 = price 1.0, tick 100 = ~1.01x price increase.
    Used for converting internal tick calculations to USD prices.
    """
    return 1.0001 ** tick  # Each tick = 0.01% price difference


def price_to_tick(price: float) -> int:
    """Convert price to tick using log formula.
    
    Inverse of tick_to_price, rounds to nearest tick for discrete price levels.
    Used for placing POMM positions and determining price ranges.
    """
    return round(math.log(price) / math.log(1.0001))  # Inverse of tick_to_price


def tick_to_sqrt_price_x64(tick: int) -> int:
    """Convert tick to sqrt price in Q64.64 format.
    
    Used for low-level AMM calculations that require fixed-point arithmetic.
    Mirrors on-chain price representation in concentrated liquidity pools.
    """
    sqrt_price = math.sqrt(tick_to_price(tick))
    return int(sqrt_price * (2 ** 64))




@dataclass
class JitMetrics:
    """Aggregate metrics for JIT operation.
    
    Tracks cumulative JIT performance including total volume boost
    and duration of active periods for analysis and reporting.
    """
    total_volume_boost: float = 0.0  # Cumulative extra volume from JIT
    active_minutes: int = 0           # Total minutes JIT was active


# JitManager class removed - use JitController from core.py instead


# Polars-based TWAP and volatility utilities
def get_twap_from_dataframe(df: pl.DataFrame, current_minute: int, window_seconds: int, min_duration_seconds: int) -> Optional[int]:
    """Extract TWAP from polars DataFrame efficiently."""
    if df.is_empty():
        return None
    
    window_minutes = max(1, window_seconds // 60)
    min_duration_minutes = max(1, min_duration_seconds // 60)
    
    # Filter to recent data
    cutoff = current_minute - window_minutes
    recent = df.filter(pl.col("timestamp") >= cutoff)
    
    if len(recent) < 2:
        return None
    
    duration = recent.select(pl.col("timestamp").max() - pl.col("timestamp").min()).item(0, 0)
    if duration < min_duration_minutes:
        return None
    
    # Get TWAP from the most recent row
    twap = recent.select(pl.col("twap_tick")).tail(1).item(0, 0)
    return int(round(twap)) if twap is not None else None


def get_volatility_from_dataframe(df: pl.DataFrame, current_minute: int) -> float:
    """Extract volatility from polars DataFrame efficiently."""
    if df.is_empty():
        return 0.0
    
    # Get the most recent volatility measurement
    recent = df.filter(pl.col("timestamp") <= current_minute).tail(1)
    if recent.is_empty():
        return 0.0
    
    volatility = recent.select("tick_volatility").item(0, 0)
    return (volatility / 100.0) if volatility is not None else 0.0
"""Market environment and external factors."""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketEnvironment:
    """External market environment state."""
    sol_price_usd: float
    volatility: float
    trend_bias: float
    sentiment: float = 0.0  # -1 to 1 scale
    
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.05, trend_bias: float = 0.0):
        self.sol_price_usd = initial_price
        self.volatility = volatility
        self.trend_bias = trend_bias
        self.sentiment = 0.0


class PriceEvolution:
    """Handles SOL price evolution using geometric Brownian motion."""
    
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(42)
    
    def update_price(self, current_price: float, volatility: float, trend_bias: float, dt: float) -> float:
        """
        Update price using geometric Brownian motion.
        
        Args:
            current_price: Current SOL price in USD
            volatility: Daily volatility (annualized)
            trend_bias: Directional bias (-1 to 1)
            dt: Time step in years
        
        Returns:
            New price
        """
        # Convert trend bias to small drift component
        drift = trend_bias * 0.001
        
        # Scale daily volatility to match time period
        period_volatility = volatility / math.sqrt(365.25)
        
        # Generate Brownian motion random component
        random_shock = self.rng.normal(0, period_volatility * math.sqrt(dt))
        
        # Geometric Brownian motion: dS/S = drift*dt + vol*dW
        log_return = drift * dt + random_shock
        
        # Apply exponential to maintain positive prices
        return current_price * math.exp(log_return)
    
    def generate_price_path(self, initial_price: float, periods: int, volatility: float, 
                          trend_bias: float, dt: float) -> np.ndarray:
        """
        Generate a complete price path.
        
        Args:
            initial_price: Starting price
            periods: Number of time periods
            volatility: Daily volatility
            trend_bias: Directional bias
            dt: Time step in years
        
        Returns:
            Array of prices
        """
        prices = np.zeros(periods + 1)
        prices[0] = initial_price
        
        for i in range(1, periods + 1):
            prices[i] = self.update_price(prices[i-1], volatility, trend_bias, dt)
        
        return prices


class SentimentModel:
    """Models market sentiment changes."""
    
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(42)
    
    def update_sentiment(self, current_sentiment: float, price_change: float, dt: float) -> float:
        """
        Update market sentiment based on price movements.
        
        Args:
            current_sentiment: Current sentiment (-1 to 1)
            price_change: Recent price change (percentage)
            dt: Time step
        
        Returns:
            New sentiment value
        """
        # Sentiment follows price movements with some persistence
        sentiment_drift = price_change * 0.1  # Positive price moves increase sentiment
        sentiment_reversion = -current_sentiment * 0.05  # Mean reversion to neutral
        sentiment_noise = self.rng.normal(0, 0.02)  # Random sentiment shifts
        
        new_sentiment = current_sentiment + (sentiment_drift + sentiment_reversion + sentiment_noise) * dt
        
        # Keep sentiment bounded between -1 and 1
        return max(-1.0, min(1.0, new_sentiment))


def calculate_yield_accrual(principal: float, annual_rate: float, dt: float) -> float:
    """
    Calculate continuous compound yield accrual.
    
    Args:
        principal: Principal amount
        annual_rate: Annual yield rate (e.g., 0.07 for 7%)
        dt: Time step in years
    
    Returns:
        Yield amount for this period
    """
    return principal * annual_rate * dt  # Simple continuous compounding
"""Market environment and external factors.

Handles SOL price evolution, market sentiment, and yield calculations.
Provides realistic external market conditions for simulation."""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketEnvironment:
    """External market environment state.
    
    Encapsulates all external market conditions affecting the protocol
    including SOL price, volatility, directional bias, and sentiment.
    """
    sol_price_usd: float = 100.0    # Current SOL price in USD (default: $100)
    volatility: float = 0.05        # Daily price volatility (default: 5%)
    trend_bias: float = 0.0         # Market trend (-1=bearish, 0=neutral, +1=bullish)
    sentiment: float = 0.0          # Market sentiment (-1=fear to +1=greed)


class PriceEvolution:
    """Handles SOL price evolution using geometric Brownian motion.
    
    Implements realistic cryptocurrency price movements with configurable
    volatility and trend bias. Used for external market simulation.
    """
    
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(42)  # Deterministic random generator
    
    def update_price(self, current_price: float, volatility: float, trend_bias: float, dt: float) -> float:
        """
        Update price using geometric Brownian motion.
        
        Core price evolution model simulating realistic crypto price movements
        with volatility clustering and directional bias.
        
        Args:
            current_price: Current SOL price in USD
            volatility: Daily volatility (annualized)
            trend_bias: Directional bias (-1 to 1)
            dt: Time step in years
        
        Returns:
            New price
        """
        # Convert trend bias to small drift component for directional movement
        drift = trend_bias * 0.001
        
        # Scale daily volatility to match simulation time period
        period_volatility = volatility / math.sqrt(365.25)
        
        # Generate Brownian motion random component for price noise
        random_shock = self.rng.normal(0, period_volatility * math.sqrt(dt))
        
        # Geometric Brownian motion formula: dS/S = drift*dt + vol*dW
        log_return = drift * dt + random_shock
        
        # Apply exponential to maintain positive prices and return new price
        return current_price * math.exp(log_return)
    
    def generate_price_path(self, initial_price: float, periods: int, volatility: float, 
                          trend_bias: float, dt: float) -> np.ndarray:
        """
        Generate a complete price path for batch simulation.
        
        Useful for generating entire price series upfront rather than
        step-by-step updates during simulation execution.
        
        Args:
            initial_price: Starting price
            periods: Number of time periods
            volatility: Daily volatility
            trend_bias: Directional bias
            dt: Time step in years
        
        Returns:
            Array of prices for all periods
        """
        prices = np.zeros(periods + 1)  # Initialize price array
        prices[0] = initial_price       # Set starting price
        
        # Generate each subsequent price using GBM
        for i in range(1, periods + 1):
            prices[i] = self.update_price(prices[i-1], volatility, trend_bias, dt)
        
        return prices


class SentimentModel:
    """Models market sentiment changes.
    
    Tracks market psychology based on price movements with mean reversion.
    Influences participant behavior and trading volume patterns.
    """
    
    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState(42)  # Deterministic random generator
    
    def update_sentiment(self, current_sentiment: float, price_change: float, dt: float) -> float:
        """
        Update market sentiment based on price movements.
        
        Implements behavioral finance model where sentiment follows price
        with momentum but also mean-reverts to neutrality over time.
        
        Args:
            current_sentiment: Current sentiment (-1 to 1)
            price_change: Recent price change (percentage)
            dt: Time step
        
        Returns:
            New sentiment value bounded between -1 and 1
        """
        # Sentiment follows price movements with momentum effect
        sentiment_drift = price_change * 0.1  # Positive price moves increase sentiment
        # Mean reversion component pulls sentiment back to neutral
        sentiment_reversion = -current_sentiment * 0.05  # Gradual return to 0
        # Random noise represents unpredictable sentiment shifts
        sentiment_noise = self.rng.normal(0, 0.02)  # Small random component
        
        # Combine all components with time scaling
        new_sentiment = current_sentiment + (sentiment_drift + sentiment_reversion + sentiment_noise) * dt
        
        # Keep sentiment bounded within valid range
        return max(-1.0, min(1.0, new_sentiment))


def calculate_yield_accrual(principal: float, annual_rate: float, dt: float) -> float:
    """
    Calculate continuous compound yield accrual.
    
    Used for JitoSOL staking yield calculations that fund synthetic
    FeelsSOL minting. Assumes continuous compounding at annual rate.
    
    Args:
        principal: Principal amount (total supply)
        annual_rate: Annual yield rate (e.g., 0.07 for 7%)
        dt: Time step in years (1/525600 for minutes)
    
    Returns:
        Yield amount accrued in this time period
    """
    return principal * annual_rate * dt  # Simple continuous compounding formula
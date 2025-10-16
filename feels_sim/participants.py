"""Participant behavior models for the Feels simulation."""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class ParticipantConfig:
    """Configuration for participant behavior parameters."""
    
    # Retail trader parameters
    retail_count: int = 100
    retail_base_frequency: float = 2.0  # trades per day
    retail_size_mean: float = 50.0  # FeelsSOL
    retail_size_variance: float = 0.8
    retail_fee_sensitivity: float = 2.0  # elasticity to fee changes
    retail_sentiment_sensitivity: float = 0.5
    
    # Algo trader parameters
    algo_count: int = 10
    algo_base_frequency: float = 20.0  # trades per day
    algo_size_mean: float = 1000.0  # FeelsSOL
    algo_size_variance: float = 1.2
    algo_fee_sensitivity: float = 0.5
    algo_sentiment_sensitivity: float = 0.2
    
    # Liquidity provider parameters
    lp_count: int = 20
    lp_position_size_mean: float = 5000.0  # FeelsSOL
    lp_position_size_variance: float = 1.0
    lp_fee_sensitivity: float = 3.0  # highly sensitive to fee earnings
    lp_rebalance_frequency: float = 6.0  # hours
    
    # Arbitrageur parameters
    arb_count: int = 5
    arb_size_mean: float = 500.0  # FeelsSOL
    arb_threshold_bps: float = 5.0  # minimum profit threshold in basis points
    arb_response_time: float = 2.0  # minutes to respond to opportunities


@dataclass
class TradeOrder:
    """Represents a trade order from a participant."""
    participant_id: str
    participant_type: str
    size: float  # in FeelsSOL
    is_buy: bool
    timestamp: int
    urgency: float = 0.5  # 0-1 scale, affects slippage tolerance


@dataclass
class LiquidityPosition:
    """Represents a liquidity provider position."""
    participant_id: str
    size: float  # in FeelsSOL
    tick_lower: int
    tick_upper: int
    timestamp_created: int
    fees_earned: float = 0.0


class ParticipantBase(ABC):
    """Base class for all market participants."""
    
    def __init__(self, participant_id: str, config: ParticipantConfig, rng: np.random.RandomState):
        self.id = participant_id
        self.config = config
        self.rng = rng
        self.last_action_time = 0
        
    @abstractmethod
    def should_trade(self, current_time: int, market_conditions: Dict) -> bool:
        """Determine if participant should trade at current time."""
        pass
        
    @abstractmethod
    def generate_trade(self, current_time: int, market_conditions: Dict) -> Optional[TradeOrder]:
        """Generate a trade order if participant decides to trade."""
        pass


class RetailTrader(ParticipantBase):
    """Retail trader with sentiment-driven behavior and fee sensitivity."""
    
    def __init__(self, participant_id: str, config: ParticipantConfig, rng: np.random.RandomState):
        super().__init__(participant_id, config, rng)
        self.sentiment = 0.0  # -1 to 1
        
    def should_trade(self, current_time: int, market_conditions: Dict) -> bool:
        """Retail traders are influenced by sentiment and recent price movements."""
        minutes_since_last = current_time - self.last_action_time
        
        # Base frequency adjusted by fee sensitivity
        fee_multiplier = 1.0 / (1.0 + market_conditions.get('fee_rate', 0.003) * self.config.retail_fee_sensitivity)
        adjusted_frequency = self.config.retail_base_frequency * fee_multiplier
        
        # Convert daily frequency to per-minute probability
        trade_probability = adjusted_frequency / (24 * 60)
        
        # Sentiment boost
        sentiment_factor = 1.0 + abs(market_conditions.get('price_change', 0.0)) * self.config.retail_sentiment_sensitivity
        
        return self.rng.random() < (trade_probability * sentiment_factor)
    
    def generate_trade(self, current_time: int, market_conditions: Dict) -> Optional[TradeOrder]:
        """Generate retail trade with sentiment bias."""
        if not self.should_trade(current_time, market_conditions):
            return None
            
        # Update sentiment based on recent price changes
        price_change = market_conditions.get('price_change', 0.0)
        self.sentiment = 0.7 * self.sentiment + 0.3 * price_change
        
        # Trade size from log-normal distribution
        size = self.rng.lognormal(
            np.log(self.config.retail_size_mean),
            self.config.retail_size_variance
        )
        
        # Buy probability influenced by sentiment
        buy_probability = 0.5 + self.sentiment * 0.3
        is_buy = self.rng.random() < buy_probability
        
        self.last_action_time = current_time
        
        return TradeOrder(
            participant_id=self.id,
            participant_type="retail",
            size=size,
            is_buy=is_buy,
            timestamp=current_time,
            urgency=0.3  # Retail traders are less urgent
        )


class AlgoTrader(ParticipantBase):
    """Algorithmic trader with sophisticated strategies and lower fee sensitivity."""
    
    def should_trade(self, current_time: int, market_conditions: Dict) -> bool:
        """Algo traders trade more frequently and are less fee sensitive."""
        minutes_since_last = current_time - self.last_action_time
        
        # Less sensitive to fees
        fee_multiplier = 1.0 / (1.0 + market_conditions.get('fee_rate', 0.003) * self.config.algo_fee_sensitivity)
        adjusted_frequency = self.config.algo_base_frequency * fee_multiplier
        
        trade_probability = adjusted_frequency / (24 * 60)
        
        # Slight volatility boost (algos like volatility for opportunities)
        volatility_factor = 1.0 + market_conditions.get('volatility', 0.05) * 0.5
        
        return self.rng.random() < (trade_probability * volatility_factor)
    
    def generate_trade(self, current_time: int, market_conditions: Dict) -> Optional[TradeOrder]:
        """Generate algorithmic trade with larger sizes and strategic timing."""
        if not self.should_trade(current_time, market_conditions):
            return None
            
        # Larger trade sizes
        size = self.rng.lognormal(
            np.log(self.config.algo_size_mean),
            self.config.algo_size_variance
        )
        
        # More sophisticated buy/sell logic (simplified mean reversion)
        recent_trend = market_conditions.get('price_change', 0.0)
        # Algos sometimes fade momentum
        buy_probability = 0.5 - recent_trend * 0.2
        is_buy = self.rng.random() < buy_probability
        
        self.last_action_time = current_time
        
        return TradeOrder(
            participant_id=self.id,
            participant_type="algo",
            size=size,
            is_buy=is_buy,
            timestamp=current_time,
            urgency=0.7  # More urgent execution
        )


class LiquidityProvider(ParticipantBase):
    """Liquidity provider that manages positions and earns fees."""
    
    def __init__(self, participant_id: str, config: ParticipantConfig, rng: np.random.RandomState):
        super().__init__(participant_id, config, rng)
        self.positions: List[LiquidityPosition] = []
        self.total_fees_earned = 0.0
        
    def should_provide_liquidity(self, current_time: int, market_conditions: Dict) -> bool:
        """Decide whether to add/remove liquidity based on fee expectations."""
        hours_since_last = (current_time - self.last_action_time) / 60.0
        
        # LPs rebalance periodically
        if hours_since_last < self.config.lp_rebalance_frequency:
            return False
            
        # Fee sensitivity - LPs care about earning fees
        expected_fee_rate = market_conditions.get('fee_rate', 0.003)
        volume_factor = market_conditions.get('volume_factor', 1.0)
        
        # Simple profitability check
        expected_daily_fees = expected_fee_rate * volume_factor * 1000  # simplified
        return expected_daily_fees > 0.001  # minimum threshold
    
    def generate_liquidity_action(self, current_time: int, market_conditions: Dict) -> Optional[LiquidityPosition]:
        """Generate liquidity provision action."""
        if not self.should_provide_liquidity(current_time, market_conditions):
            return None
            
        current_tick = market_conditions.get('current_tick', 0)
        
        # Position size
        size = self.rng.lognormal(
            np.log(self.config.lp_position_size_mean),
            self.config.lp_position_size_variance
        )
        
        # Position range (simplified - around current price)
        tick_spacing = 50  # simplified
        range_width = self.rng.randint(2, 10) * tick_spacing
        tick_lower = current_tick - range_width // 2
        tick_upper = current_tick + range_width // 2
        
        position = LiquidityPosition(
            participant_id=self.id,
            size=size,
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            timestamp_created=current_time
        )
        
        self.positions.append(position)
        self.last_action_time = current_time
        
        return position
    
    def should_trade(self, current_time: int, market_conditions: Dict) -> bool:
        """LPs don't trade directly - they provide liquidity."""
        return False
        
    def generate_trade(self, current_time: int, market_conditions: Dict) -> Optional[TradeOrder]:
        """LPs provide liquidity, not trades."""
        return None


class Arbitrageur(ParticipantBase):
    """Arbitrageur that trades on price discrepancies."""
    
    def should_trade(self, current_time: int, market_conditions: Dict) -> bool:
        """Arbitrageurs trade when profit opportunities exist."""
        # Check for arbitrage opportunity
        price_discrepancy = market_conditions.get('price_discrepancy', 0.0)
        return abs(price_discrepancy) > (self.config.arb_threshold_bps / 10000.0)
    
    def generate_trade(self, current_time: int, market_conditions: Dict) -> Optional[TradeOrder]:
        """Generate arbitrage trade to close price gaps."""
        if not self.should_trade(current_time, market_conditions):
            return None
            
        price_discrepancy = market_conditions.get('price_discrepancy', 0.0)
        
        # Trade size based on opportunity size
        opportunity_factor = abs(price_discrepancy) * 10000  # convert to bps
        size = self.config.arb_size_mean * min(opportunity_factor / 10.0, 3.0)  # cap at 3x
        
        # Trade against the discrepancy
        is_buy = price_discrepancy < 0  # buy if price is below fair value
        
        self.last_action_time = current_time
        
        return TradeOrder(
            participant_id=self.id,
            participant_type="arbitrageur",
            size=size,
            is_buy=is_buy,
            timestamp=current_time,
            urgency=0.9  # Very urgent execution
        )


class ParticipantPool:
    """Manages all market participants and their interactions."""
    
    def __init__(self, config: ParticipantConfig, rng: np.random.RandomState):
        self.config = config
        self.rng = rng
        self.participants: List[ParticipantBase] = []
        self._initialize_participants()
        
    def _initialize_participants(self):
        """Create all participants according to configuration."""
        
        # Create retail traders
        for i in range(self.config.retail_count):
            participant = RetailTrader(f"retail_{i}", self.config, self.rng)
            self.participants.append(participant)
            
        # Create algo traders
        for i in range(self.config.algo_count):
            participant = AlgoTrader(f"algo_{i}", self.config, self.rng)
            self.participants.append(participant)
            
        # Create liquidity providers
        for i in range(self.config.lp_count):
            participant = LiquidityProvider(f"lp_{i}", self.config, self.rng)
            self.participants.append(participant)
            
        # Create arbitrageurs
        for i in range(self.config.arb_count):
            participant = Arbitrageur(f"arb_{i}", self.config, self.rng)
            self.participants.append(participant)
    
    def generate_orders(self, current_time: int, market_conditions: Dict) -> List[TradeOrder]:
        """Generate all trade orders for the current time step."""
        orders = []
        
        for participant in self.participants:
            order = participant.generate_trade(current_time, market_conditions)
            if order:
                orders.append(order)
                
        return orders
    
    def generate_liquidity_actions(self, current_time: int, market_conditions: Dict) -> List[LiquidityPosition]:
        """Generate liquidity provision actions."""
        actions = []
        
        for participant in self.participants:
            if isinstance(participant, LiquidityProvider):
                action = participant.generate_liquidity_action(current_time, market_conditions)
                if action:
                    actions.append(action)
                    
        return actions
    
    def get_participant_metrics(self) -> Dict[str, any]:
        """Get metrics about participant activity."""
        metrics = {
            "total_participants": len(self.participants),
            "retail_count": self.config.retail_count,
            "algo_count": self.config.algo_count,
            "lp_count": self.config.lp_count,
            "arb_count": self.config.arb_count,
            "total_lp_positions": sum(len(p.positions) for p in self.participants if isinstance(p, LiquidityProvider)),
            "total_lp_fees": sum(p.total_fees_earned for p in self.participants if isinstance(p, LiquidityProvider))
        }
        
        return metrics


def calculate_volume_from_orders(orders: List[TradeOrder]) -> float:
    """Calculate total trading volume from a list of orders."""
    return sum(order.size for order in orders)


def calculate_market_conditions(sol_price: float, prev_sol_price: float, 
                              fee_rate: float, volatility: float) -> Dict:
    """Calculate market conditions for participant decision making."""
    price_change = (sol_price - prev_sol_price) / prev_sol_price if prev_sol_price > 0 else 0.0
    
    return {
        "current_price": sol_price,
        "price_change": price_change,
        "fee_rate": fee_rate,
        "volatility": volatility,
        "volume_factor": 1.0 + abs(price_change) * 5.0,  # higher volume during big moves
        "current_tick": int(math.log(sol_price / 100.0) / math.log(1.0001)),  # simplified
        "price_discrepancy": 0.0  # simplified - no external price reference yet
    }
"""Metrics collection and analysis utilities."""

from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

from .core import SimulationSnapshot, SimulationResults


@dataclass
class MetricsSummary:
    """Summary of key simulation metrics."""
    floor_growth_rate_annual: float
    avg_floor_to_market_ratio: float
    pomm_deployments: int
    total_volume: float
    total_fees: float
    final_treasury_balance: float
    final_buffer_balance: float
    final_mintable_feelssol: float


class MetricsCollector:
    """Collects and processes simulation metrics."""
    
    def __init__(self):
        self.snapshots: List[SimulationSnapshot] = []
    
    def add_snapshot(self, snapshot: SimulationSnapshot):
        """Add a snapshot to the collection."""
        self.snapshots.append(snapshot)
    
    def calculate_floor_growth_rate(self) -> float:
        """Calculate annualized floor price growth rate."""
        if len(self.snapshots) < 2:
            return 0.0
        
        initial_floor = self.snapshots[0].floor_price_usd
        final_floor = self.snapshots[-1].floor_price_usd
        
        if initial_floor <= 0:
            return 0.0
        
        # Calculate time period in years
        minutes_elapsed = len(self.snapshots)
        years_elapsed = minutes_elapsed / (365.25 * 24 * 60)
        
        if years_elapsed <= 0:
            return 0.0
        
        # Calculate growth ratio
        growth_ratio = final_floor / initial_floor
        
        # For very short periods or extreme growth, cap the annualized rate
        if years_elapsed < 0.1 or growth_ratio > 1000:  # Less than ~36 days or 1000x growth
            # Return simple percentage change rather than annualized
            return (growth_ratio - 1.0)
        
        # Calculate compound annual growth rate
        growth_rate = growth_ratio ** (1 / years_elapsed) - 1
        
        # Cap at reasonable maximum (1000% annual growth)
        return min(growth_rate, 10.0)
    
    def calculate_floor_to_market_ratio(self) -> float:
        """Calculate average floor-to-market price ratio."""
        if not self.snapshots:
            return 0.0
        
        ratios = []
        for snapshot in self.snapshots:
            if snapshot.sol_price_usd > 0:
                # Approximate market price as SOL price (simplified for Phase 1)
                ratio = snapshot.floor_price_usd / snapshot.sol_price_usd
                ratios.append(ratio)
        
        return np.mean(ratios) if ratios else 0.0
    
    def calculate_pomm_deployment_count(self) -> int:
        """Count total POMM deployments."""
        return sum(1 for s in self.snapshots if s.events.get("pomm_deployed", False))
    
    def calculate_total_volume(self) -> float:
        """Calculate total trading volume."""
        return sum(s.volume_feelssol for s in self.snapshots)
    
    def calculate_total_fees(self) -> float:
        """Calculate total fees collected."""
        return sum(s.fees_collected for s in self.snapshots)


def analyze_results(results: SimulationResults) -> Dict[str, Any]:
    """
    Analyze simulation results and return key metrics.
    
    Args:
        results: Complete simulation results
    
    Returns:
        Dictionary of analysis metrics
    """
    collector = MetricsCollector()
    for snapshot in results.snapshots:
        collector.add_snapshot(snapshot)
    
    if not results.snapshots:
        return {}
    
    final_snapshot = results.snapshots[-1]
    
    analysis = {
        "floor_growth_rate_annual": collector.calculate_floor_growth_rate(),
        "avg_floor_to_market_ratio": collector.calculate_floor_to_market_ratio(),
        "pomm_deployments": collector.calculate_pomm_deployment_count(),
        "total_volume": collector.calculate_total_volume(),
        "total_fees": collector.calculate_total_fees(),
        "final_treasury_balance": final_snapshot.floor_state.treasury_balance,
        "final_buffer_balance": final_snapshot.floor_state.buffer_balance,
        "final_mintable_feelssol": final_snapshot.floor_state.mintable_feelssol,
        "final_deployed_feelssol": final_snapshot.floor_state.deployed_feelssol,
        "buffer_routed_cumulative": final_snapshot.floor_state.buffer_routed_cumulative,
        "mint_cumulative": final_snapshot.floor_state.mint_cumulative,
        "simulation_hours": len(results.snapshots) / 60,
        "initial_floor_price": results.snapshots[0].floor_price_usd,
        "final_floor_price": final_snapshot.floor_price_usd,
        "initial_sol_price": results.snapshots[0].sol_price_usd,
        "final_sol_price": final_snapshot.sol_price_usd
    }
    
    return analysis


def create_summary_plots(results: SimulationResults, save_path: str = None) -> None:
    """
    Create summary plots of simulation results.
    
    Args:
        results: Simulation results to plot
        save_path: Optional path to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    if not results.snapshots:
        print("No data to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Feels Simulation Results")
    
    # Extract time series data
    minutes = [s.timestamp for s in results.snapshots]
    hours = [m / 60 for m in minutes]
    sol_prices = [s.sol_price_usd for s in results.snapshots]
    floor_prices = [s.floor_price_usd for s in results.snapshots]
    volumes = [s.volume_feelssol for s in results.snapshots]
    buffer_balances = [s.floor_state.buffer_balance for s in results.snapshots]
    mintable_balances = [s.floor_state.mintable_feelssol for s in results.snapshots]
    
    # Plot 1: Price evolution
    axes[0, 0].plot(hours, sol_prices, label='SOL Price', alpha=0.7)
    axes[0, 0].plot(hours, floor_prices, label='Floor Price', alpha=0.7)
    axes[0, 0].set_xlabel('Hours')
    axes[0, 0].set_ylabel('Price (USD)')
    axes[0, 0].set_title('Price Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trading volume
    axes[0, 1].plot(hours, volumes, alpha=0.7)
    axes[0, 1].set_xlabel('Hours')
    axes[0, 1].set_ylabel('Volume (FeelsSOL)')
    axes[0, 1].set_title('Trading Volume')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Yield buffer
    axes[1, 0].plot(hours, buffer_balances, label='Buffer', alpha=0.7)
    axes[1, 0].plot(hours, mintable_balances, label='Mintable FeelsSOL', alpha=0.7)
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].set_ylabel('FeelsSOL')
    axes[1, 0].set_title('Automatic Funding Balances')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Floor to market ratio
    floor_to_market = [f/s if s > 0 else 0 for f, s in zip(floor_prices, sol_prices)]
    axes[1, 1].plot(hours, floor_to_market, alpha=0.7)
    axes[1, 1].set_xlabel('Hours')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_title('Floor/Market Price Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

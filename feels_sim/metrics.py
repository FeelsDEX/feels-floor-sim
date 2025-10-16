"""Metrics collection and analysis utilities."""

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import json

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
    lp_yield_apy: float
    protocol_efficiency: float
    buffer_utilization: float


@dataclass
class TimeseriesMetrics:
    """Time-series metrics for detailed analysis."""
    hourly_volume: List[float]
    hourly_fees: List[float]
    floor_deltas: List[float]
    buffer_contributions: List[float]
    mint_contributions: List[float]
    floor_to_market_ratios: List[float]
    participant_volumes: Dict[str, List[float]]
    

@dataclass
class AggregateMetrics:
    """Metrics aggregated across different time horizons."""
    daily: Dict[str, List[float]]
    weekly: Dict[str, List[float]]
    monthly: Dict[str, List[float]]


class MetricsCollector:
    """Collects and processes simulation metrics."""
    
    def __init__(self):
        self.snapshots: List[SimulationSnapshot] = []
        self.hourly_aggregates: List[Dict[str, Any]] = []
    
    def add_snapshot(self, snapshot: SimulationSnapshot):
        """Add a snapshot to the collection."""
        self.snapshots.append(snapshot)
    
    def add_hourly_aggregate(self, aggregate: Dict[str, Any]):
        """Add an hourly aggregate to the collection."""
        self.hourly_aggregates.append(aggregate)
    
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
    
    def calculate_lp_yield_apy(self) -> float:
        """Calculate annualized LP yield based on fee accrual."""
        if not self.hourly_aggregates:
            return 0.0
        
        total_lp_fees = sum(agg.get('lp_fees_earned', 0) for agg in self.hourly_aggregates)
        avg_lp_positions = np.mean([agg.get('total_lp_positions', 0) for agg in self.hourly_aggregates])
        
        if avg_lp_positions <= 0:
            return 0.0
        
        hours_elapsed = len(self.hourly_aggregates)
        if hours_elapsed <= 0:
            return 0.0
        
        # Calculate hourly yield rate
        hourly_yield = total_lp_fees / (avg_lp_positions * hours_elapsed)
        
        # Annualize (8760 hours per year)
        annual_yield = hourly_yield * 8760
        
        # Cap at reasonable maximum
        return min(annual_yield, 10.0)
    
    def calculate_protocol_efficiency(self) -> float:
        """Calculate protocol efficiency as floor advancement per fee dollar."""
        if not self.snapshots or len(self.snapshots) < 2:
            return 0.0
        
        total_fees = self.calculate_total_fees()
        if total_fees <= 0:
            return 0.0
        
        initial_floor = self.snapshots[0].floor_price_usd
        final_floor = self.snapshots[-1].floor_price_usd
        floor_advancement = final_floor - initial_floor
        
        return floor_advancement / total_fees
    
    def calculate_buffer_utilization(self) -> float:
        """Calculate average buffer utilization rate."""
        if not self.snapshots:
            return 0.0
        
        utilizations = []
        for snapshot in self.snapshots:
            deployed = snapshot.floor_state.deployed_feelssol
            available = snapshot.floor_state.buffer_balance + deployed
            if available > 0:
                utilizations.append(deployed / available)
        
        return np.mean(utilizations) if utilizations else 0.0
    
    def derive_hourly_aggregates(self) -> List[Dict[str, float]]:
        """Derive hourly aggregates from minute snapshots."""
        if not self.snapshots:
            return []
        
        hourly_data = []
        
        # Group snapshots by hour
        for hour in range(0, len(self.snapshots), 60):
            hour_snapshots = self.snapshots[hour:hour + 60]
            if not hour_snapshots:
                continue
            
            aggregate = {
                'hour': hour // 60,
                'volume_feelssol': sum(s.volume_feelssol for s in hour_snapshots),
                'fees_collected': sum(s.fees_collected for s in hour_snapshots),
                'buffer_routed': 0.0,  # Derived from floor state changes
                'mint_amount': 0.0,    # Derived from floor state changes
                'pomm_deployments': sum(1 for s in hour_snapshots if s.events.get('pomm_deployed', False)),
                'avg_sol_price': np.mean([s.sol_price_usd for s in hour_snapshots]),
                'avg_floor_price': np.mean([s.floor_price_usd for s in hour_snapshots]),
                'final_buffer_balance': hour_snapshots[-1].floor_state.buffer_balance,
                'final_treasury_balance': hour_snapshots[-1].floor_state.treasury_balance,
                'floor_delta': hour_snapshots[-1].floor_price_usd - hour_snapshots[0].floor_price_usd,
            }
            
            # Calculate buffer routed and mint amounts from state changes
            if hour > 0:
                prev_hour_end = hour - 1
                if prev_hour_end < len(self.snapshots):
                    prev_state = self.snapshots[prev_hour_end].floor_state
                    current_state = hour_snapshots[-1].floor_state
                    
                    # Estimate buffer routed from cumulative changes
                    buffer_change = current_state.buffer_routed_cumulative - prev_state.buffer_routed_cumulative
                    mint_change = current_state.mint_cumulative - prev_state.mint_cumulative
                    
                    aggregate['buffer_routed'] = max(0.0, buffer_change)
                    aggregate['mint_amount'] = max(0.0, mint_change)
            
            # Add participant metrics if available
            if hasattr(hour_snapshots[0], 'participant_volumes'):
                for participant_type in ['retail', 'algo', 'lp', 'arbitrageur']:
                    volumes = [getattr(s, 'participant_volumes', {}).get(participant_type, 0) for s in hour_snapshots]
                    aggregate[f'{participant_type}_volume'] = sum(volumes)
            
            hourly_data.append(aggregate)
        
        return hourly_data
    
    def derive_daily_aggregates(self) -> List[Dict[str, float]]:
        """Derive daily aggregates from hourly data."""
        hourly_data = self.derive_hourly_aggregates()
        if not hourly_data:
            return []
        
        daily_data = []
        
        # Group hourly data by day (24 hours)
        for day in range(0, len(hourly_data), 24):
            day_hours = hourly_data[day:day + 24]
            if not day_hours:
                continue
            
            aggregate = {
                'day': day // 24,
                'volume_feelssol': sum(h['volume_feelssol'] for h in day_hours),
                'fees_collected': sum(h['fees_collected'] for h in day_hours),
                'buffer_routed': sum(h['buffer_routed'] for h in day_hours),
                'mint_amount': sum(h['mint_amount'] for h in day_hours),
                'pomm_deployments': sum(h['pomm_deployments'] for h in day_hours),
                'avg_sol_price': np.mean([h['avg_sol_price'] for h in day_hours]),
                'avg_floor_price': np.mean([h['avg_floor_price'] for h in day_hours]),
                'floor_delta': sum(h['floor_delta'] for h in day_hours),
                'final_buffer_balance': day_hours[-1]['final_buffer_balance'],
                'final_treasury_balance': day_hours[-1]['final_treasury_balance'],
            }
            
            daily_data.append(aggregate)
        
        return daily_data
    
    def derive_weekly_aggregates(self) -> List[Dict[str, float]]:
        """Derive weekly aggregates from daily data."""
        daily_data = self.derive_daily_aggregates()
        if not daily_data:
            return []
        
        weekly_data = []
        
        # Group daily data by week (7 days)
        for week in range(0, len(daily_data), 7):
            week_days = daily_data[week:week + 7]
            if not week_days:
                continue
            
            aggregate = {
                'week': week // 7,
                'volume_feelssol': sum(d['volume_feelssol'] for d in week_days),
                'fees_collected': sum(d['fees_collected'] for d in week_days),
                'buffer_routed': sum(d['buffer_routed'] for d in week_days),
                'mint_amount': sum(d['mint_amount'] for d in week_days),
                'pomm_deployments': sum(d['pomm_deployments'] for d in week_days),
                'avg_sol_price': np.mean([d['avg_sol_price'] for d in week_days]),
                'avg_floor_price': np.mean([d['avg_floor_price'] for d in week_days]),
                'floor_delta': sum(d['floor_delta'] for d in week_days),
                'final_buffer_balance': week_days[-1]['final_buffer_balance'],
                'final_treasury_balance': week_days[-1]['final_treasury_balance'],
            }
            
            weekly_data.append(aggregate)
        
        return weekly_data
    
    def export_metrics(self, file_path: str = None, format: str = 'json') -> None:
        """Export metrics to file (defaults to experiments/outputs/data/)."""
        metrics_data = {
            'summary': {
                'floor_growth_rate_annual': self.calculate_floor_growth_rate(),
                'avg_floor_to_market_ratio': self.calculate_floor_to_market_ratio(),
                'pomm_deployments': self.calculate_pomm_deployment_count(),
                'total_volume': self.calculate_total_volume(),
                'total_fees': self.calculate_total_fees(),
                'lp_yield_apy': self.calculate_lp_yield_apy(),
                'protocol_efficiency': self.calculate_protocol_efficiency(),
                'buffer_utilization': self.calculate_buffer_utilization(),
            },
            'aggregates': {
                'hourly': self.derive_hourly_aggregates(),
                'daily': self.derive_daily_aggregates(),
                'weekly': self.derive_weekly_aggregates(),
            }
        }
        
        if format.lower() == 'json':
            if file_path:
                # Ensure the directory exists
                import os
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"Metrics exported to {file_path}")
            else:
                # Default to experiments/outputs/data/ directory
                import os
                from datetime import datetime
                
                output_dir = "experiments/outputs/data"
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_path = f"{output_dir}/simulation_metrics_{timestamp}.json"
                
                with open(default_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"Metrics exported to {default_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")


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
    
    # Add hourly aggregates if available
    if hasattr(results, 'hourly_aggregates'):
        for aggregate in results.hourly_aggregates:
            collector.add_hourly_aggregate(aggregate)
    
    if not results.snapshots:
        return {}
    
    final_snapshot = results.snapshots[-1]
    
    analysis = {
        "floor_growth_rate_annual": collector.calculate_floor_growth_rate(),
        "avg_floor_to_market_ratio": collector.calculate_floor_to_market_ratio(),
        "pomm_deployments": collector.calculate_pomm_deployment_count(),
        "total_volume": collector.calculate_total_volume(),
        "total_fees": collector.calculate_total_fees(),
        "lp_yield_apy": collector.calculate_lp_yield_apy(),
        "protocol_efficiency": collector.calculate_protocol_efficiency(),
        "buffer_utilization": collector.calculate_buffer_utilization(),
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


def calculate_floor_floor_ratio_stats(results: SimulationResults) -> Dict[str, float]:
    """Calculate statistical measures of floor/market price ratio."""
    if not results.snapshots:
        return {}
    
    ratios = []
    for snapshot in results.snapshots:
        if snapshot.sol_price_usd > 0:
            ratio = snapshot.floor_price_usd / snapshot.sol_price_usd
            ratios.append(ratio)
    
    if not ratios:
        return {}
    
    return {
        'mean_floor_ratio': np.mean(ratios),
        'median_floor_ratio': np.median(ratios),
        'std_floor_ratio': np.std(ratios),
        'min_floor_ratio': np.min(ratios),
        'max_floor_ratio': np.max(ratios),
        'p25_floor_ratio': np.percentile(ratios, 25),
        'p75_floor_ratio': np.percentile(ratios, 75)
    }


def calculate_pomm_efficiency_metrics(results: SimulationResults) -> Dict[str, float]:
    """Calculate POMM deployment efficiency metrics."""
    if not results.snapshots:
        return {}
    
    deployments = [s for s in results.snapshots if s.events.get('pomm_deployed', False)]
    
    if not deployments:
        return {'pomm_count': 0, 'avg_deployment_size': 0, 'deployment_frequency': 0}
    
    # Estimate deployment sizes from buffer balance changes
    deployment_sizes = []
    for i, deployment in enumerate(deployments):
        if i > 0:
            # Look at buffer balance change around deployment
            prev_deployment = deployments[i-1]
            buffer_change = prev_deployment.floor_state.buffer_balance - deployment.floor_state.buffer_balance
            if buffer_change > 0:
                deployment_sizes.append(buffer_change)
        else:
            # For first deployment, estimate from current buffer balance
            if deployment.floor_state.buffer_balance > 0:
                deployment_sizes.append(deployment.floor_state.buffer_balance)
    
    hours_elapsed = len(results.snapshots) / 60
    deployment_frequency = len(deployments) / max(hours_elapsed, 0.1)  # deployments per hour
    
    return {
        'pomm_count': len(deployments),
        'avg_deployment_size': np.mean(deployment_sizes) if deployment_sizes else 0,
        'median_deployment_size': np.median(deployment_sizes) if deployment_sizes else 0,
        'deployment_frequency': deployment_frequency,
        'total_deployed_amount': sum(deployment_sizes)
    }


def calculate_volume_elasticity(low_fee_results: SimulationResults, high_fee_results: SimulationResults) -> Dict[str, float]:
    """Calculate fee elasticity of trading volume."""
    low_volume = sum(s.volume_feelssol for s in low_fee_results.snapshots)
    high_volume = sum(s.volume_feelssol for s in high_fee_results.snapshots)
    
    if low_volume == 0 or high_volume == 0:
        return {'elasticity': 0, 'volume_change_pct': 0}
    
    volume_change_pct = (low_volume - high_volume) / high_volume
    
    # Simple elasticity calculation (percentage change in volume / percentage change in fee)
    # This would require fee rate information to be precise
    
    return {
        'low_fee_volume': low_volume,
        'high_fee_volume': high_volume,
        'volume_change_pct': volume_change_pct,
        'elasticity_observed': low_volume > high_volume  # Boolean indicating if elasticity exists
    }


def create_summary_plots(results: SimulationResults, save_path: str = None) -> None:
    """
    Create summary plots of simulation results.
    
    Args:
        results: Simulation results to plot
        save_path: Optional path to save plots (defaults to experiments/outputs/plots/)
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feels Simulation Results", fontsize=16)
    
    # Extract time series data
    minutes = [s.timestamp for s in results.snapshots]
    hours = [m / 60 for m in minutes]
    sol_prices = [s.sol_price_usd for s in results.snapshots]
    floor_prices = [s.floor_price_usd for s in results.snapshots]
    volumes = [s.volume_feelssol for s in results.snapshots]
    buffer_balances = [s.floor_state.buffer_balance for s in results.snapshots]
    mintable_balances = [s.floor_state.mintable_feelssol for s in results.snapshots]
    treasury_balances = [s.floor_state.treasury_balance for s in results.snapshots]
    
    # Plot 1: Price evolution
    axes[0, 0].plot(hours, sol_prices, label='SOL Price', alpha=0.8, linewidth=1.5)
    axes[0, 0].plot(hours, floor_prices, label='Floor Price', alpha=0.8, linewidth=1.5)
    axes[0, 0].set_xlabel('Hours')
    axes[0, 0].set_ylabel('Price (USD)')
    axes[0, 0].set_title('Price Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trading volume
    axes[0, 1].plot(hours, volumes, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Hours')
    axes[0, 1].set_ylabel('Volume (FeelsSOL)')
    axes[0, 1].set_title('Trading Volume')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Floor to market ratio
    floor_to_market = [f/s if s > 0 else 0 for f, s in zip(floor_prices, sol_prices)]
    axes[0, 2].plot(hours, floor_to_market, alpha=0.7, color='purple')
    axes[0, 2].set_xlabel('Hours')
    axes[0, 2].set_ylabel('Ratio')
    axes[0, 2].set_title('Floor/Market Price Ratio')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Buffer and funding sources
    axes[1, 0].plot(hours, buffer_balances, label='Buffer', alpha=0.7)
    axes[1, 0].plot(hours, mintable_balances, label='Mintable FeelsSOL', alpha=0.7)
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].set_ylabel('FeelsSOL')
    axes[1, 0].set_title('Funding Sources')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Treasury accumulation
    axes[1, 1].plot(hours, treasury_balances, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Hours')
    axes[1, 1].set_ylabel('FeelsSOL')
    axes[1, 1].set_title('Treasury Balance')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: POMM deployment events
    pomm_events = [(h, 1) for h, s in zip(hours, results.snapshots) if s.events.get('pomm_deployed', False)]
    if pomm_events:
        pomm_hours, pomm_markers = zip(*pomm_events)
        axes[1, 2].scatter(pomm_hours, pomm_markers, alpha=0.7, color='red', s=50)
        axes[1, 2].set_xlabel('Hours')
        axes[1, 2].set_ylabel('Deployment Event')
        axes[1, 2].set_title('POMM Deployments')
        axes[1, 2].set_ylim(0, 1.5)
    else:
        axes[1, 2].text(0.5, 0.5, 'No POMM\nDeployments', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('POMM Deployments')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    else:
        # Default to experiments/outputs/plots/ directory
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/simulation_summary_{timestamp}.png"
        
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to {default_path}")
    
    plt.close()


def create_detailed_analysis_plots(results: SimulationResults, save_path: str = None) -> None:
    """Create detailed analysis plots with aggregated data."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    if not results.snapshots:
        print("No data to plot")
        return
    
    # Create metrics collector for aggregation
    collector = MetricsCollector()
    for snapshot in results.snapshots:
        collector.add_snapshot(snapshot)
    
    hourly_data = collector.derive_hourly_aggregates()
    if not hourly_data:
        print("No hourly data available")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Detailed Analysis - Hourly Aggregates", fontsize=16)
    
    hours = [h['hour'] for h in hourly_data]
    hourly_volumes = [h['volume_feelssol'] for h in hourly_data]
    hourly_fees = [h['fees_collected'] for h in hourly_data]
    floor_deltas = [h['floor_delta'] for h in hourly_data]
    pomm_counts = [h['pomm_deployments'] for h in hourly_data]
    
    # Plot 1: Hourly volume
    axes[0, 0].bar(hours, hourly_volumes, alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Hours')
    axes[0, 0].set_ylabel('Volume (FeelsSOL)')
    axes[0, 0].set_title('Hourly Trading Volume')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hourly fees
    axes[0, 1].bar(hours, hourly_fees, alpha=0.7, color='forestgreen')
    axes[0, 1].set_xlabel('Hours')
    axes[0, 1].set_ylabel('Fees (FeelsSOL)')
    axes[0, 1].set_title('Hourly Fee Collection')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Floor price advances
    axes[1, 0].bar(hours, floor_deltas, alpha=0.7, color='darkorange')
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].set_ylabel('Floor Advance (USD)')
    axes[1, 0].set_title('Hourly Floor Price Advances')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: POMM deployment frequency
    axes[1, 1].bar(hours, pomm_counts, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Hours')
    axes[1, 1].set_ylabel('POMM Deployments')
    axes[1, 1].set_title('Hourly POMM Deployments')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base_path = save_path.replace('.png', '_detailed.png')
        plt.savefig(base_path, dpi=150, bbox_inches='tight')
        print(f"Detailed plots saved to {base_path}")
    else:
        # Default to experiments/outputs/plots/ directory
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/simulation_detailed_{timestamp}.png"
        
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Detailed plots saved to {default_path}")
    
    plt.close()


def generate_summary_report(results: SimulationResults, file_path: str = None) -> str:
    """Generate a markdown summary report of simulation results."""
    analysis = analyze_results(results)
    pomm_metrics = calculate_pomm_efficiency_metrics(results)
    ratio_stats = calculate_floor_floor_ratio_stats(results)
    
    report = f"""# Feels Simulation Summary Report

## Simulation Overview
- **Duration**: {analysis.get('simulation_hours', 0):.1f} hours
- **Initial SOL Price**: ${analysis.get('initial_sol_price', 0):.2f}
- **Final SOL Price**: ${analysis.get('final_sol_price', 0):.2f}
- **Initial Floor Price**: ${analysis.get('initial_floor_price', 0):.4f}
- **Final Floor Price**: ${analysis.get('final_floor_price', 0):.4f}

## Key Metrics

### Floor Price Performance
- **Annual Growth Rate**: {analysis.get('floor_growth_rate_annual', 0):.2%}
- **Average Floor/Market Ratio**: {analysis.get('avg_floor_to_market_ratio', 0):.2%}
- **Protocol Efficiency**: {analysis.get('protocol_efficiency', 0):.4f} USD/FeelsSOL

### Trading Activity
- **Total Volume**: {analysis.get('total_volume', 0):,.0f} FeelsSOL
- **Total Fees**: {analysis.get('total_fees', 0):,.2f} FeelsSOL
- **LP Yield APY**: {analysis.get('lp_yield_apy', 0):.2%}

### POMM Performance
- **Total Deployments**: {pomm_metrics.get('pomm_count', 0)}
- **Average Deployment Size**: {pomm_metrics.get('avg_deployment_size', 0):,.2f} FeelsSOL
- **Deployment Frequency**: {pomm_metrics.get('deployment_frequency', 0):.2f} per hour
- **Buffer Utilization**: {analysis.get('buffer_utilization', 0):.2%}

### Final Balances
- **Buffer Balance**: {analysis.get('final_buffer_balance', 0):,.2f} FeelsSOL
- **Treasury Balance**: {analysis.get('final_treasury_balance', 0):,.2f} FeelsSOL
- **Mintable FeelsSOL**: {analysis.get('final_mintable_feelssol', 0):,.2f} FeelsSOL
- **Deployed FeelsSOL**: {analysis.get('final_deployed_feelssol', 0):,.2f} FeelsSOL

### Floor/Market Ratio Statistics
- **Mean**: {ratio_stats.get('mean_floor_ratio', 0):.3f}
- **Median**: {ratio_stats.get('median_floor_ratio', 0):.3f}
- **Standard Deviation**: {ratio_stats.get('std_floor_ratio', 0):.3f}
- **Min**: {ratio_stats.get('min_floor_ratio', 0):.3f}
- **Max**: {ratio_stats.get('max_floor_ratio', 0):.3f}

## Funding Sources
- **Buffer Routed (Cumulative)**: {analysis.get('buffer_routed_cumulative', 0):,.2f} FeelsSOL
- **Mint (Cumulative)**: {analysis.get('mint_cumulative', 0):,.2f} FeelsSOL

---
*Report generated from Feels simulation data*
"""
    
    if file_path:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {file_path}")
    else:
        # Default to experiments/outputs/reports/ directory
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/reports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/simulation_report_{timestamp}.md"
        
        with open(default_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {default_path}")
    
    return report

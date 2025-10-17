"""Visualization and reporting functions for simulation results with seaborn styling."""

import pandas as pd
import numpy as np
from typing import Dict

from .analysis import analyze_results, calculate_pomm_efficiency_metrics, calculate_floor_floor_ratio_stats
from .state import SimulationResults


def create_summary_plots(results: SimulationResults, save_path: str = None) -> None:
    """
    Create summary plots of simulation results using seaborn for enhanced aesthetics.
    
    Args:
        results: Simulation results to plot
        save_path: Optional path to save plots (defaults to experiments/outputs/plots/)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and/or seaborn not available, skipping plots")
        return
    
    if not results.snapshots:
        print("No data to plot")
        return
    
    # Set seaborn style for publication-quality plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feels Simulation Results", fontsize=16, y=0.98)
    
    # Create DataFrame for easier seaborn plotting
    data = []
    for s in results.snapshots:
        data.append({
            'timestamp': s.timestamp,
            'hours': s.timestamp / 60,
            'sol_price_usd': s.sol_price_usd,
            'floor_price_usd': s.floor_price_usd,
            'volume_feelssol': s.volume_feelssol,
            'buffer_balance': s.floor_state.buffer_balance,
            'mintable_feelssol': s.floor_state.mintable_feelssol,
            'treasury_balance': s.floor_state.treasury_balance,
            'floor_to_market_ratio': s.floor_price_usd / s.sol_price_usd if s.sol_price_usd > 0 else 0,
            'pomm_deployed': s.events.get('pomm_deployed', False)
        })
    
    df = pd.DataFrame(data)
    
    # Plot 1: Price evolution with confidence bands
    sns.lineplot(data=df, x='hours', y='sol_price_usd', label='SOL Price', 
                alpha=0.8, linewidth=2, ax=axes[0, 0])
    sns.lineplot(data=df, x='hours', y='floor_price_usd', label='Floor Price', 
                alpha=0.8, linewidth=2, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Hours')
    axes[0, 0].set_ylabel('Price (USD)')
    axes[0, 0].set_title('Price Evolution')
    axes[0, 0].legend()
    
    # Plot 2: Trading volume with trend line
    sns.scatterplot(data=df, x='hours', y='volume_feelssol', alpha=0.6, 
                   color='forestgreen', s=30, ax=axes[0, 1])
    sns.regplot(data=df, x='hours', y='volume_feelssol', scatter=False, 
               color='darkgreen', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Hours')
    axes[0, 1].set_ylabel('Volume (FeelsSOL)')
    axes[0, 1].set_title('Trading Volume with Trend')
    
    # Plot 3: Floor to market ratio with statistical bands
    sns.lineplot(data=df, x='hours', y='floor_to_market_ratio', 
                color='purple', linewidth=2, ax=axes[0, 2])
    # Add rolling mean for smoothing
    if len(df) > 10:
        df['ratio_smooth'] = df['floor_to_market_ratio'].rolling(window=10, center=True).mean()
        sns.lineplot(data=df, x='hours', y='ratio_smooth', 
                    color='darkviolet', linestyle='--', alpha=0.7, ax=axes[0, 2])
    axes[0, 2].set_xlabel('Hours')
    axes[0, 2].set_ylabel('Ratio')
    axes[0, 2].set_title('Floor/Market Price Ratio')
    
    # Plot 4: Funding sources stacked area
    axes[1, 0].fill_between(df['hours'], 0, df['buffer_balance'], 
                           alpha=0.7, label='Buffer', color=sns.color_palette()[0])
    axes[1, 0].fill_between(df['hours'], df['buffer_balance'], 
                           df['buffer_balance'] + df['mintable_feelssol'], 
                           alpha=0.7, label='Mintable FeelsSOL', color=sns.color_palette()[1])
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].set_ylabel('FeelsSOL')
    axes[1, 0].set_title('Funding Sources (Stacked)')
    axes[1, 0].legend()
    
    # Plot 5: Treasury accumulation with growth rate annotation
    sns.lineplot(data=df, x='hours', y='treasury_balance', 
                color='darkorange', linewidth=2, ax=axes[1, 1])
    # Add growth rate if data is available
    if len(df) > 1:
        initial_treasury = df['treasury_balance'].iloc[0]
        final_treasury = df['treasury_balance'].iloc[-1]
        growth = ((final_treasury / initial_treasury) - 1) * 100 if initial_treasury > 0 else 0
        axes[1, 1].text(0.05, 0.95, f'Growth: {growth:.1f}%', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].set_xlabel('Hours')
    axes[1, 1].set_ylabel('FeelsSOL')
    axes[1, 1].set_title('Treasury Balance')
    
    # Plot 6: POMM deployment heatmap/timeline
    pomm_df = df[df['pomm_deployed'] == True]
    if len(pomm_df) > 0:
        # Create deployment intensity over time
        hourly_pomm = df.groupby(df['hours'].astype(int))['pomm_deployed'].sum().reset_index()
        if len(hourly_pomm) > 0:
            sns.barplot(data=hourly_pomm, x='hours', y='pomm_deployed', 
                       color='crimson', alpha=0.7, ax=axes[1, 2])
            axes[1, 2].set_xlabel('Hours')
            axes[1, 2].set_ylabel('POMM Deployments')
            axes[1, 2].set_title(f'POMM Deployments ({len(pomm_df)} total)')
        else:
            axes[1, 2].text(0.5, 0.5, 'No POMM\nDeployments', 
                          ha='center', va='center', transform=axes[1, 2].transAxes,
                          fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 2].set_title('POMM Deployments')
    else:
        axes[1, 2].text(0.5, 0.5, 'No POMM\nDeployments', 
                       ha='center', va='center', transform=axes[1, 2].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1, 2].set_title('POMM Deployments')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Enhanced plots saved to {save_path}")
    else:
        # Default to experiments/outputs/plots/ directory
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/simulation_summary_{timestamp}.png"
        
        plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Enhanced plots saved to {default_path}")
    
    plt.close()


def create_detailed_analysis_plots(results: SimulationResults, save_path: str = None) -> None:
    """Create detailed analysis plots with aggregated data using seaborn statistical plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and/or seaborn not available, skipping plots")
        return
    
    if not results.snapshots:
        print("No data to plot")
        return
    
    # Use direct function calls from metrics module for aggregation
    from .metrics import calculate_hourly_aggregates
    hourly_data = calculate_hourly_aggregates(results.snapshots)
    if not hourly_data:
        print("No hourly data available")
        return
    
    # Set seaborn style for detailed analysis
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    
    # Create enhanced figure layout
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    fig.suptitle("Detailed Statistical Analysis - Hourly Aggregates", fontsize=16, y=0.98)
    
    # Convert to DataFrame for seaborn
    hourly_df = pd.DataFrame(hourly_data)
    
    # Plot 1: Volume distribution and trend
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=hourly_df, x='volume_feelssol', kde=True, alpha=0.7, 
                color='steelblue', ax=ax1)
    ax1.axvline(hourly_df['volume_feelssol'].mean(), color='red', linestyle='--', 
               label=f'Mean: {hourly_df["volume_feelssol"].mean():.1f}')
    ax1.set_title('Volume Distribution')
    ax1.set_xlabel('Volume (FeelsSOL)')
    ax1.legend()
    
    # Plot 2: Volume over time with confidence interval
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=hourly_df, x='hour', y='volume_feelssol', 
                marker='o', markersize=4, ax=ax2)
    sns.regplot(data=hourly_df, x='hour', y='volume_feelssol', 
               scatter=False, color='gray', ax=ax2)
    ax2.set_title('Volume Trend Over Time')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Volume (FeelsSOL)')
    
    # Plot 3: Fee collection efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    # Create fee efficiency metric
    hourly_df['fee_efficiency'] = hourly_df['fees_collected'] / (hourly_df['volume_feelssol'] + 1e-8)
    sns.boxplot(data=hourly_df, y='fee_efficiency', ax=ax3)
    sns.stripplot(data=hourly_df, y='fee_efficiency', alpha=0.6, ax=ax3)
    ax3.set_title('Fee Collection Efficiency')
    ax3.set_ylabel('Fees / Volume Ratio')
    
    # Plot 4: Floor advancement correlation
    ax4 = fig.add_subplot(gs[1, 1])
    if 'floor_delta' in hourly_df.columns:
        sns.scatterplot(data=hourly_df, x='volume_feelssol', y='floor_delta', 
                       size='fees_collected', alpha=0.7, ax=ax4)
        # Add correlation coefficient
        correlation = hourly_df['volume_feelssol'].corr(hourly_df['floor_delta'])
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax4.set_title('Volume vs Floor Advancement')
        ax4.set_xlabel('Volume (FeelsSOL)')
        ax4.set_ylabel('Floor Advance (USD)')
    
    # Plot 5: POMM deployment heatmap
    ax5 = fig.add_subplot(gs[2, :])
    if 'pomm_deployments' in hourly_df.columns and hourly_df['pomm_deployments'].sum() > 0:
        # Create time-series heatmap of POMM activity
        hourly_df['day'] = hourly_df['hour'] // 24
        hourly_df['hour_of_day'] = hourly_df['hour'] % 24
        
        if len(hourly_df['day'].unique()) > 1:
            # Pivot for heatmap
            pomm_pivot = hourly_df.pivot_table(
                values='pomm_deployments', 
                index='day', 
                columns='hour_of_day', 
                fill_value=0
            )
            
            sns.heatmap(pomm_pivot, annot=True, fmt='g', cmap='Reds', 
                       cbar_kws={'label': 'POMM Deployments'}, ax=ax5)
            ax5.set_title('POMM Deployment Activity by Day and Hour')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('Simulation Day')
        else:
            # Single day - show hourly distribution
            sns.barplot(data=hourly_df, x='hour_of_day', y='pomm_deployments',
                       palette='Reds', ax=ax5)
            ax5.set_title('POMM Deployment Activity by Hour')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('POMM Deployments')
    else:
        ax5.text(0.5, 0.5, 'No POMM Deployment Data Available', 
                ha='center', va='center', transform=ax5.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray'))
        ax5.set_title('POMM Deployment Analysis')
    
    # Add summary statistics box
    if len(hourly_df) > 0:
        summary_text = f"""Summary Statistics:
        • Avg Volume: {hourly_df['volume_feelssol'].mean():.1f} FeelsSOL/hr
        • Avg Fees: {hourly_df['fees_collected'].mean():.2f} FeelsSOL/hr
        • Total POMM: {hourly_df['pomm_deployments'].sum():.0f} deployments
        • Simulation Hours: {len(hourly_df)} hours"""
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                verticalalignment='bottom')
    
    if save_path:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base_path = save_path.replace('.png', '_detailed.png')
        plt.savefig(base_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Enhanced detailed plots saved to {base_path}")
    else:
        # Default to experiments/outputs/plots/ directory
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/simulation_detailed_{timestamp}.png"
        
        plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Enhanced detailed plots saved to {default_path}")
    
    plt.close()


def create_parameter_sweep_plots(sweep_results: pd.DataFrame, save_path: str = None) -> None:
    """
    Create seaborn plots for parameter sweep analysis with statistical insights.
    
    Args:
        sweep_results: DataFrame with columns like scenario, base_fee_bps, trend_bias, 
                      volatility_daily, final_buffer_balance, etc.
        save_path: Optional path to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and/or seaborn not available, skipping plots")
        return
    
    if sweep_results.empty:
        print("No sweep data to plot")
        return
    
    # Set seaborn style for parameter analysis
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.0)
    
    # Create comprehensive parameter sweep visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)
    
    fig.suptitle("Parameter Sweep Analysis - Statistical Insights", fontsize=18, y=0.98)
    
    # Plot 1: Buffer balance by scenario (boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=sweep_results, x='scenario', y='final_buffer_balance', ax=ax1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Buffer Balance Distribution by Scenario')
    ax1.set_ylabel('Final Buffer Balance')
    
    # Plot 2: Fee rate impact on performance
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(data=sweep_results, x='base_fee_bps', y='final_buffer_balance', 
                   hue='scenario', alpha=0.7, ax=ax2)
    sns.regplot(data=sweep_results, x='base_fee_bps', y='final_buffer_balance', 
               scatter=False, color='gray', ax=ax2)
    ax2.set_title('Fee Rate vs Buffer Performance')
    ax2.set_xlabel('Base Fee (bps)')
    ax2.set_ylabel('Final Buffer Balance')
    
    # Plot 3: Volatility impact heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    if 'volatility_daily' in sweep_results.columns and 'trend_bias' in sweep_results.columns:
        pivot_data = sweep_results.pivot_table(
            values='final_buffer_balance', 
            index='volatility_daily', 
            columns='trend_bias',
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=ax3,
                   cbar_kws={'label': 'Avg Buffer Balance'})
        ax3.set_title('Buffer Balance: Volatility vs Trend Bias')
    
    # Plot 4: Volume vs buffer correlation
    ax4 = fig.add_subplot(gs[1, 0])
    if 'total_volume' in sweep_results.columns:
        sns.scatterplot(data=sweep_results, x='total_volume', y='final_buffer_balance',
                       hue='scenario', size='total_fees', alpha=0.7, ax=ax4)
        correlation = sweep_results['total_volume'].corr(sweep_results['final_buffer_balance'])
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax4.set_title('Volume vs Buffer Balance')
        ax4.set_xlabel('Total Volume')
        ax4.set_ylabel('Final Buffer Balance')
    
    # Plot 5: Treasury performance
    ax5 = fig.add_subplot(gs[1, 1])
    if 'final_treasury_balance' in sweep_results.columns:
        sns.violinplot(data=sweep_results, x='scenario', y='final_treasury_balance', ax=ax5)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        ax5.set_title('Treasury Balance Distribution')
        ax5.set_ylabel('Final Treasury Balance')
    
    # Plot 6: POMM deployment efficiency
    ax6 = fig.add_subplot(gs[1, 2])
    if 'pomm_deployments' in sweep_results.columns:
        # Create efficiency metric
        sweep_results['pomm_efficiency'] = sweep_results['final_buffer_balance'] / (sweep_results['pomm_deployments'] + 1)
        sns.boxplot(data=sweep_results, x='scenario', y='pomm_efficiency', ax=ax6)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.set_title('POMM Deployment Efficiency')
        ax6.set_ylabel('Buffer Balance per POMM')
    
    # Plot 7: Multi-parameter relationship
    ax7 = fig.add_subplot(gs[2, :])
    if len(sweep_results.columns) >= 4:
        # Create a correlation matrix for key metrics
        numeric_cols = sweep_results.select_dtypes(include=[np.number]).columns
        correlation_matrix = sweep_results[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax7, cbar_kws={'label': 'Correlation Coefficient'})
        ax7.set_title('Parameter Correlation Matrix')
    
    # Plot 8: Performance ranking
    ax8 = fig.add_subplot(gs[3, 0])
    if 'scenario' in sweep_results.columns:
        scenario_performance = sweep_results.groupby('scenario')['final_buffer_balance'].agg(['mean', 'std']).reset_index()
        scenario_performance = scenario_performance.sort_values('mean', ascending=True)
        
        sns.barplot(data=scenario_performance, x='mean', y='scenario', 
                   orient='h', ax=ax8, palette='viridis')
        ax8.set_title('Average Performance by Scenario')
        ax8.set_xlabel('Mean Buffer Balance')
    
    # Plot 9: Risk-return analysis
    ax9 = fig.add_subplot(gs[3, 1])
    if 'scenario' in sweep_results.columns:
        risk_return = sweep_results.groupby('scenario').agg({
            'final_buffer_balance': ['mean', 'std'],
            'total_volume': 'mean'
        }).reset_index()
        risk_return.columns = ['scenario', 'mean_buffer', 'std_buffer', 'mean_volume']
        
        sns.scatterplot(data=risk_return, x='std_buffer', y='mean_buffer', 
                       size='mean_volume', hue='scenario', s=200, alpha=0.8, ax=ax9)
        ax9.set_title('Risk-Return Analysis')
        ax9.set_xlabel('Buffer Balance Std Dev (Risk)')
        ax9.set_ylabel('Mean Buffer Balance (Return)')
    
    # Plot 10: Summary statistics table
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')
    
    # Create summary statistics
    summary_stats = []
    for scenario in sweep_results['scenario'].unique():
        scenario_data = sweep_results[sweep_results['scenario'] == scenario]
        stats = {
            'Scenario': scenario,
            'Count': len(scenario_data),
            'Mean Buffer': f"{scenario_data['final_buffer_balance'].mean():.0f}",
            'Max Buffer': f"{scenario_data['final_buffer_balance'].max():.0f}",
            'Mean Volume': f"{scenario_data.get('total_volume', pd.Series([0])).mean():.0f}"
        }
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create table
    table_data = []
    for col in summary_df.columns:
        table_data.append(summary_df[col].values)
    
    table = ax10.table(cellText=list(zip(*table_data)), 
                      rowLabels=summary_df.columns,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax10.set_title('Summary Statistics', pad=20)
    
    if save_path:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        base_path = save_path.replace('.png', '_sweep_analysis.png')
        plt.savefig(base_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Parameter sweep analysis saved to {base_path}")
    else:
        # Default to experiments/outputs/plots/ directory
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"{output_dir}/parameter_sweep_{timestamp}.png"
        
        plt.savefig(default_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Parameter sweep analysis saved to {default_path}")
    
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
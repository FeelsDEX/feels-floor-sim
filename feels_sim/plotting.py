"""Visualization and reporting functions for simulation results with centralized DRY styling."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .analysis import analyze_results, calculate_pomm_efficiency_metrics, calculate_floor_floor_ratio_stats
from .state import SimulationResults


@dataclass
class PlotStyle:
    """Configuration for consistent plot styling across the application."""
    
    # Seaborn theme settings
    style: str = "whitegrid"
    context: str = "notebook"
    palette: str = "husl"
    font_scale: float = 1.0
    
    # Figure settings
    figure_size_summary: Tuple[int, int] = (18, 10)
    figure_size_detailed: Tuple[int, int] = (16, 12)
    figure_size_sweep: Tuple[int, int] = (20, 16)
    figure_size_single: Tuple[int, int] = (12, 8)
    
    # Professional styling
    dpi: int = 300
    facecolor: str = 'white'
    edgecolor: str = 'white'
    linewidth: float = 1.5
    alpha: float = 0.8
    marker_edgewidth: float = 2
    
    # Color scheme
    primary_color: str = '#1f77b4'
    secondary_color: str = '#ff7f0e'
    accent_color: str = '#2ca02c'
    error_color: str = '#d62728'
    
    @property
    def color_palette(self) -> list:
        """Get standardized color palette."""
        try:
            import seaborn as sns
            return sns.color_palette(self.palette, 8)
        except ImportError:
            return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


# Global style configuration
DEFAULT_STYLE = PlotStyle()


def setup_plot_style(style: Optional[PlotStyle] = None) -> None:
    """Setup consistent plotting style for all visualizations.
    
    Args:
        style: Custom style configuration. Uses DEFAULT_STYLE if None.
    """
    if style is None:
        style = DEFAULT_STYLE
        
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set seaborn theme with modern approach
        sns.set_theme(
            context=style.context,
            style=style.style,
            palette=style.palette,
            font_scale=style.font_scale
        )
        
        # Configure matplotlib rcParams for professional output
        plt.rcParams.update({
            'figure.facecolor': style.facecolor,
            'savefig.facecolor': style.facecolor,
            'savefig.dpi': style.dpi,
            'savefig.bbox': 'tight',
            'axes.edgecolor': '#2E2E2E',
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        
    except ImportError:
        print("matplotlib and/or seaborn not available for styling")


def apply_professional_styling(ax, style: Optional[PlotStyle] = None) -> None:
    """Apply professional styling to a matplotlib axes object.
    
    Args:
        ax: Matplotlib axes object
        style: Custom style configuration. Uses DEFAULT_STYLE if None.
    """
    if style is None:
        style = DEFAULT_STYLE
        
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set spine colors
    ax.spines['left'].set_color('#2E2E2E')
    ax.spines['bottom'].set_color('#2E2E2E')
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def create_figure_with_style(nrows: int = 1, ncols: int = 1, 
                           figsize: Optional[Tuple[int, int]] = None,
                           style: Optional[PlotStyle] = None) -> Tuple[Any, Any]:
    """Create a matplotlib figure with consistent styling.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns  
        figsize: Figure size override
        style: Custom style configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    if style is None:
        style = DEFAULT_STYLE
        
    if figsize is None:
        figsize = style.figure_size_single
        
    try:
        import matplotlib.pyplot as plt
        
        setup_plot_style(style)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Apply professional styling to all axes
        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                apply_professional_styling(ax, style)
        else:
            apply_professional_styling(axes, style)
            
        return fig, axes
        
    except ImportError:
        print("matplotlib not available")
        return None, None


def save_plot(fig, save_path: Optional[str] = None, 
             default_name: str = "plot",
             style: Optional[PlotStyle] = None) -> str:
    """Save plot with consistent settings and automatic path generation.
    
    Args:
        fig: Matplotlib figure object
        save_path: Custom save path. Auto-generates if None.
        default_name: Default filename prefix
        style: Custom style configuration
        
    Returns:
        Actual save path used
    """
    if style is None:
        style = DEFAULT_STYLE
        
    if save_path:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        actual_path = save_path
    else:
        # Auto-generate path
        import os
        from datetime import datetime
        
        output_dir = "experiments/outputs/plots"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_path = f"{output_dir}/{default_name}_{timestamp}.png"
    
    try:
        fig.savefig(
            actual_path, 
            dpi=style.dpi, 
            bbox_inches='tight', 
            facecolor=style.facecolor
        )
        print(f"Plot saved to {actual_path}")
        return actual_path
    except Exception as e:
        print(f"Error saving plot: {e}")
        return ""


def create_line_plot(df: pd.DataFrame, x: str, y: str, ax=None, 
                    style: Optional[PlotStyle] = None, **kwargs) -> Any:
    """Create a standardized line plot with consistent styling.
    
    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        ax: Matplotlib axes object (optional)
        style: Custom style configuration
        **kwargs: Additional arguments passed to sns.lineplot
        
    Returns:
        Matplotlib axes object
    """
    if style is None:
        style = DEFAULT_STYLE
        
    try:
        import seaborn as sns
        
        # Set default styling parameters
        plot_kwargs = {
            'alpha': style.alpha,
            'linewidth': style.linewidth
        }
        plot_kwargs.update(kwargs)
        
        return sns.lineplot(data=df, x=x, y=y, ax=ax, **plot_kwargs)
        
    except ImportError:
        print("seaborn not available")
        return ax


def create_scatter_plot(df: pd.DataFrame, x: str, y: str, ax=None,
                       style: Optional[PlotStyle] = None, **kwargs) -> Any:
    """Create a standardized scatter plot with consistent styling.
    
    Args:
        df: DataFrame with data
        x: Column name for x-axis
        y: Column name for y-axis
        ax: Matplotlib axes object (optional)
        style: Custom style configuration
        **kwargs: Additional arguments passed to sns.scatterplot
        
    Returns:
        Matplotlib axes object
    """
    if style is None:
        style = DEFAULT_STYLE
        
    try:
        import seaborn as sns
        
        # Set default styling parameters
        plot_kwargs = {
            'alpha': style.alpha - 0.2,
            's': 50
        }
        plot_kwargs.update(kwargs)
        
        return sns.scatterplot(data=df, x=x, y=y, ax=ax, **plot_kwargs)
        
    except ImportError:
        print("seaborn not available")
        return ax


def create_box_plot(df: pd.DataFrame, x: str = None, y: str = None, ax=None,
                   style: Optional[PlotStyle] = None, **kwargs) -> Any:
    """Create a standardized box plot with consistent styling.
    
    Args:
        df: DataFrame with data
        x: Column name for x-axis (optional)
        y: Column name for y-axis
        ax: Matplotlib axes object (optional)
        style: Custom style configuration
        **kwargs: Additional arguments passed to sns.boxplot
        
    Returns:
        Matplotlib axes object
    """
    if style is None:
        style = DEFAULT_STYLE
        
    try:
        import seaborn as sns
        
        # Set default styling parameters
        plot_kwargs = {}
        plot_kwargs.update(kwargs)
        
        return sns.boxplot(data=df, x=x, y=y, ax=ax, **plot_kwargs)
        
    except ImportError:
        print("seaborn not available")
        return ax


def create_heatmap(data, ax=None, style: Optional[PlotStyle] = None, **kwargs) -> Any:
    """Create a standardized heatmap with consistent styling.
    
    Args:
        data: Data for heatmap (DataFrame or array-like)
        ax: Matplotlib axes object (optional)
        style: Custom style configuration
        **kwargs: Additional arguments passed to sns.heatmap
        
    Returns:
        Matplotlib axes object
    """
    if style is None:
        style = DEFAULT_STYLE
        
    try:
        import seaborn as sns
        
        # Set default styling parameters
        plot_kwargs = {
            'annot': True,
            'fmt': '.2f',
            'cmap': 'RdYlBu_r'
        }
        plot_kwargs.update(kwargs)
        
        return sns.heatmap(data, ax=ax, **plot_kwargs)
        
    except ImportError:
        print("seaborn not available")
        return ax


def create_summary_plots(results: SimulationResults, save_path: str = None, 
                        style: Optional[PlotStyle] = None) -> None:
    """
    Create summary plots of simulation results using centralized styling.
    
    Args:
        results: Simulation results to plot
        save_path: Optional path to save plots (defaults to experiments/outputs/plots/)
        style: Custom style configuration. Uses DEFAULT_STYLE if None.
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
    
    if style is None:
        style = DEFAULT_STYLE
    
    # Create figure with consistent styling
    fig, axes = create_figure_with_style(2, 3, style.figure_size_summary, style)
    if fig is None:
        return
        
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
    base_floor = df['floor_price_usd'].iloc[0]
    base_sol = df['sol_price_usd'].iloc[0]
    df['floor_change_bps'] = (df['floor_price_usd'] / base_floor - 1.0) * 10_000
    df['sol_change_pct'] = (df['sol_price_usd'] / base_sol - 1.0) * 100

    price_ax = axes[0, 0]
    sol_line = price_ax.plot(
        df['hours'],
        df['sol_change_pct'],
        label='SOL Δ (%)',
        color=style.primary_color,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    price_ax.set_xlabel('Hours')
    price_ax.set_ylabel('SOL Change (%)')
    sol_min, sol_max = df['sol_change_pct'].min(), df['sol_change_pct'].max()
    if sol_max == sol_min:
        sol_min -= 0.01
        sol_max += 0.01
    sol_pad = max(0.02, 0.1 * abs(sol_max - sol_min))
    price_ax.set_ylim(sol_min - sol_pad, sol_max + sol_pad)

    floor_ax = price_ax.twinx()
    floor_line = floor_ax.plot(
        df['hours'],
        df['floor_change_bps'],
        label='Floor Δ (bps)',
        color=style.secondary_color,
        linewidth=style.linewidth,
        linestyle='--',
        alpha=style.alpha,
    )
    floor_ax.set_ylabel('Floor Change (bps)')
    floor_min, floor_max = df['floor_change_bps'].min(), df['floor_change_bps'].max()
    if floor_max == floor_min:
        floor_min -= 1.0
        floor_max += 1.0
    floor_pad = max(1.0, 0.1 * abs(floor_max - floor_min))
    floor_ax.set_ylim(floor_min - floor_pad, floor_max + floor_pad)

    price_ax.set_title('Price & Floor Evolution')
    lines = sol_line + floor_line
    labels = [line.get_label() for line in lines]
    price_ax.legend(lines, labels, loc='upper left', frameon=True)
    
    # Plot 2: Trading volume with trend line
    sns.scatterplot(data=df, x='hours', y='volume_feelssol', alpha=style.alpha-0.2, 
                   color=style.accent_color, s=30, ax=axes[0, 1])
    sns.regplot(data=df, x='hours', y='volume_feelssol', scatter=False, 
               color=style.accent_color, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Hours')
    axes[0, 1].set_ylabel('Volume (FeelsSOL)')
    axes[0, 1].set_title('Trading Volume with Trend')
    
    # Plot 3: Floor to market ratio with statistical bands
    sns.lineplot(data=df, x='hours', y='floor_to_market_ratio', 
                color=style.secondary_color, linewidth=style.linewidth, ax=axes[0, 2])
    # Add rolling mean for smoothing
    if len(df) > 10:
        df['ratio_smooth'] = df['floor_to_market_ratio'].rolling(window=10, center=True).mean()
        sns.lineplot(data=df, x='hours', y='ratio_smooth', 
                    color=style.secondary_color, linestyle='--', alpha=style.alpha-0.1, ax=axes[0, 2])
    axes[0, 2].set_xlabel('Hours')
    axes[0, 2].set_ylabel('Ratio')
    axes[0, 2].set_title('Floor/Market Price Ratio')
    
    # Plot 4: Funding sources stacked area
    colors = style.color_palette
    axes[1, 0].fill_between(df['hours'], 0, df['buffer_balance'], 
                           alpha=style.alpha-0.1, label='Buffer', color=colors[0])
    axes[1, 0].fill_between(df['hours'], df['buffer_balance'], 
                           df['buffer_balance'] + df['mintable_feelssol'], 
                           alpha=style.alpha-0.1, label='Mintable FeelsSOL', color=colors[1])
    axes[1, 0].set_xlabel('Hours')
    axes[1, 0].set_ylabel('FeelsSOL')
    axes[1, 0].set_title('Funding Sources (Stacked)')
    axes[1, 0].legend()
    
    # Plot 5: Treasury accumulation with growth rate annotation
    sns.lineplot(data=df, x='hours', y='treasury_balance', 
                color=style.primary_color, linewidth=style.linewidth, ax=axes[1, 1])
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
                       color=style.error_color, alpha=style.alpha-0.1, ax=axes[1, 2])
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
    
    # Save plot with centralized function
    save_plot(fig, save_path, "simulation_summary", style)
    plt.close()


def create_detailed_analysis_plots(results: SimulationResults, save_path: str = None,
                                  style: Optional[PlotStyle] = None) -> None:
    """Create detailed analysis plots with aggregated data using centralized styling."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib and/or seaborn not available, skipping plots")
        return
    
    if not results.snapshots:
        print("No data to plot")
        return
    
    if style is None:
        style = DEFAULT_STYLE
    
    # Use direct function calls from metrics module for aggregation
    from .metrics import calculate_hourly_aggregates
    hourly_data = calculate_hourly_aggregates(results.snapshots)
    if not hourly_data:
        print("No hourly data available")
        return
    
    # Setup consistent styling
    setup_plot_style(style)
    
    # Create enhanced figure layout
    fig = plt.figure(figsize=style.figure_size_detailed)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
    
    fig.suptitle("Detailed Statistical Analysis - Hourly Aggregates", fontsize=16, y=0.98)
    
    # Convert to DataFrame for seaborn and add hour column
    hourly_df = pd.DataFrame(hourly_data)
    # Add hour column for plotting (sequential hour numbers)
    hourly_df['hour'] = range(len(hourly_df))
    
    # Plot 1: Volume distribution and trend
    ax1 = fig.add_subplot(gs[0, 0])
    apply_professional_styling(ax1, style)
    sns.histplot(data=hourly_df, x='volume_feelssol', kde=True, alpha=style.alpha-0.1, 
                color=style.primary_color, ax=ax1)
    ax1.axvline(hourly_df['volume_feelssol'].mean(), color=style.error_color, linestyle='--', 
               label=f'Mean: {hourly_df["volume_feelssol"].mean():.1f}')
    ax1.set_title('Volume Distribution')
    ax1.set_xlabel('Volume (FeelsSOL)')
    ax1.legend()
    
    # Plot 2: Volume over time with confidence interval
    ax2 = fig.add_subplot(gs[0, 1])
    apply_professional_styling(ax2, style)
    sns.lineplot(data=hourly_df, x='hour', y='volume_feelssol', 
                marker='o', markersize=4, ax=ax2)
    sns.regplot(data=hourly_df, x='hour', y='volume_feelssol', 
               scatter=False, color=style.secondary_color, ax=ax2)
    ax2.set_title('Volume Trend Over Time')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Volume (FeelsSOL)')
    
    # Plot 3: Fee collection efficiency
    ax3 = fig.add_subplot(gs[1, 0])
    apply_professional_styling(ax3, style)
    # Create fee efficiency metric
    hourly_df['fee_efficiency'] = hourly_df['fees_collected'] / (hourly_df['volume_feelssol'] + 1e-8)
    sns.boxplot(data=hourly_df, y='fee_efficiency', ax=ax3)
    sns.stripplot(data=hourly_df, y='fee_efficiency', alpha=style.alpha-0.2, ax=ax3)
    ax3.set_title('Fee Collection Efficiency')
    ax3.set_ylabel('Fees / Volume Ratio')
    
    # Plot 4: Floor advancement correlation
    ax4 = fig.add_subplot(gs[1, 1])
    apply_professional_styling(ax4, style)
    if 'floor_delta' in hourly_df.columns:
        sns.scatterplot(data=hourly_df, x='volume_feelssol', y='floor_delta', 
                       size='fees_collected', alpha=style.alpha-0.1, ax=ax4)
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
    apply_professional_styling(ax5, style)
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
    
    # Save plot with centralized function
    if save_path:
        base_path = save_path.replace('.png', '_detailed.png')
        save_plot(fig, base_path, "simulation_detailed", style)
    else:
        save_plot(fig, None, "simulation_detailed", style)
    
    plt.close()


def create_parameter_sweep_plots(sweep_results: pd.DataFrame, save_path: str = None,
                                style: Optional[PlotStyle] = None) -> None:
    """
    Create plots for parameter sweep analysis with centralized styling.
    
    Args:
        sweep_results: DataFrame with columns like scenario, base_fee_bps, trend_bias, 
                      volatility_daily, final_buffer_balance, etc.
        save_path: Optional path to save plots
        style: Custom style configuration. Uses DEFAULT_STYLE if None.
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
    
    if style is None:
        style = DEFAULT_STYLE
    
    # Setup consistent styling
    setup_plot_style(style)
    
    # Create comprehensive parameter sweep visualization
    fig = plt.figure(figsize=style.figure_size_sweep)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)
    
    fig.suptitle("Parameter Sweep Analysis - Statistical Insights", fontsize=18, y=0.98)
    
    # Plot 1: Buffer balance by scenario (boxplot)
    ax1 = fig.add_subplot(gs[0, 0])
    apply_professional_styling(ax1, style)
    sns.boxplot(data=sweep_results, x='scenario', y='final_buffer_balance', ax=ax1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Buffer Balance Distribution by Scenario')
    ax1.set_ylabel('Final Buffer Balance')
    
    # Plot 2: Fee rate impact on performance
    ax2 = fig.add_subplot(gs[0, 1])
    apply_professional_styling(ax2, style)
    sns.scatterplot(data=sweep_results, x='base_fee_bps', y='final_buffer_balance', 
                   hue='scenario', alpha=style.alpha-0.1, ax=ax2)
    sns.regplot(data=sweep_results, x='base_fee_bps', y='final_buffer_balance', 
               scatter=False, color=style.secondary_color, ax=ax2)
    ax2.set_title('Fee Rate vs Buffer Performance')
    ax2.set_xlabel('Base Fee (bps)')
    ax2.set_ylabel('Final Buffer Balance')
    
    # Plot 3: Volatility impact heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    apply_professional_styling(ax3, style)
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
    apply_professional_styling(ax4, style)
    if 'total_volume' in sweep_results.columns:
        sns.scatterplot(data=sweep_results, x='total_volume', y='final_buffer_balance',
                       hue='scenario', size='total_fees', alpha=style.alpha-0.1, ax=ax4)
        correlation = sweep_results['total_volume'].corr(sweep_results['final_buffer_balance'])
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax4.set_title('Volume vs Buffer Balance')
        ax4.set_xlabel('Total Volume')
        ax4.set_ylabel('Final Buffer Balance')
    
    # Plot 5: Treasury performance
    ax5 = fig.add_subplot(gs[1, 1])
    apply_professional_styling(ax5, style)
    if 'final_treasury_balance' in sweep_results.columns:
        sns.violinplot(data=sweep_results, x='scenario', y='final_treasury_balance', ax=ax5)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        ax5.set_title('Treasury Balance Distribution')
        ax5.set_ylabel('Final Treasury Balance')
    
    # Plot 6: POMM deployment efficiency
    ax6 = fig.add_subplot(gs[1, 2])
    apply_professional_styling(ax6, style)
    if 'pomm_deployments' in sweep_results.columns:
        # Create efficiency metric
        sweep_results['pomm_efficiency'] = sweep_results['final_buffer_balance'] / (sweep_results['pomm_deployments'] + 1)
        sns.boxplot(data=sweep_results, x='scenario', y='pomm_efficiency', ax=ax6)
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.set_title('POMM Deployment Efficiency')
        ax6.set_ylabel('Buffer Balance per POMM')
    
    # Plot 7: Multi-parameter relationship
    ax7 = fig.add_subplot(gs[2, :])
    apply_professional_styling(ax7, style)
    if len(sweep_results.columns) >= 4:
        # Create a correlation matrix for key metrics
        numeric_cols = sweep_results.select_dtypes(include=[np.number]).columns
        correlation_matrix = sweep_results[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax7, cbar_kws={'label': 'Correlation Coefficient'})
        ax7.set_title('Parameter Correlation Matrix')
    
    # Plot 8: Performance ranking
    ax8 = fig.add_subplot(gs[3, 0])
    apply_professional_styling(ax8, style)
    if 'scenario' in sweep_results.columns:
        scenario_performance = sweep_results.groupby('scenario')['final_buffer_balance'].agg(['mean', 'std']).reset_index()
        scenario_performance = scenario_performance.sort_values('mean', ascending=True)
        
        sns.barplot(data=scenario_performance, x='mean', y='scenario', 
                   orient='h', ax=ax8, palette='viridis')
        ax8.set_title('Average Performance by Scenario')
        ax8.set_xlabel('Mean Buffer Balance')
    
    # Plot 9: Risk-return analysis
    ax9 = fig.add_subplot(gs[3, 1])
    apply_professional_styling(ax9, style)
    if 'scenario' in sweep_results.columns:
        risk_return = sweep_results.groupby('scenario').agg({
            'final_buffer_balance': ['mean', 'std'],
            'total_volume': 'mean'
        }).reset_index()
        risk_return.columns = ['scenario', 'mean_buffer', 'std_buffer', 'mean_volume']
        
        sns.scatterplot(data=risk_return, x='std_buffer', y='mean_buffer', 
                       size='mean_volume', hue='scenario', s=200, alpha=style.alpha, ax=ax9)
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
    
    # Save plot with centralized function
    if save_path:
        base_path = save_path.replace('.png', '_sweep_analysis.png')
        save_plot(fig, base_path, "parameter_sweep", style)
    else:
        save_plot(fig, None, "parameter_sweep", style)
    
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


# Convenience functions for common plot types
def plot_price_evolution(df: pd.DataFrame, style: Optional[PlotStyle] = None) -> Tuple[Any, Any]:
    """Create a price evolution plot focusing on FeelsSOL floor price dynamics.
    
    Args:
        df: DataFrame with 'hour', 'floor_price_usd', 'sol_price_usd' columns
        style: Custom style configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    if style is None:
        style = DEFAULT_STYLE
        
    fig, ax = create_figure_with_style(1, 1, style.figure_size_single, style)
    if fig is None:
        return None, None
    
    # Calculate floor-to-market ratio as percentage
    df = df.copy()
    df['floor_to_market_ratio_pct'] = (df['floor_price_usd'] / df['sol_price_usd']) * 100
    
    # Primary plot: Floor Price in USD
    create_line_plot(df, 'hour', 'floor_price_usd', ax=ax, 
                    label='FeelsSOL Floor Price (USD)', color=style.primary_color, style=style)
    
    # Secondary y-axis for floor/market ratio
    ax2 = ax.twinx()
    create_line_plot(df, 'hour', 'floor_to_market_ratio_pct', ax=ax2,
                    label='Floor/Market Ratio (%)', color=style.secondary_color, style=style)
    
    # Styling
    ax.set_xlabel('Hours')
    ax.set_ylabel('FeelsSOL Floor Price (USD)', color=style.primary_color)
    ax2.set_ylabel('Floor/Market Ratio (%)', color=style.secondary_color)
    ax.set_title('FeelsSOL Floor Price Evolution')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    return fig, ax


def plot_volume_analysis(df: pd.DataFrame, style: Optional[PlotStyle] = None) -> Tuple[Any, Any]:
    """Create a volume analysis plot with trend line.
    
    Args:
        df: DataFrame with 'hour', 'volume_feelssol' columns
        style: Custom style configuration
        
    Returns:
        Tuple of (figure, axes)
    """
    if style is None:
        style = DEFAULT_STYLE
        
    fig, ax = create_figure_with_style(1, 1, style.figure_size_single, style)
    if fig is None:
        return None, None
        
    create_scatter_plot(df, 'hour', 'volume_feelssol', ax=ax,
                       color=style.accent_color, style=style)
    
    try:
        import seaborn as sns
        sns.regplot(data=df, x='hour', y='volume_feelssol', 
                   scatter=False, color=style.accent_color, ax=ax)
    except ImportError:
        pass
    
    ax.set_xlabel('Hours')
    ax.set_ylabel('Volume (FeelsSOL)')
    ax.set_title('Trading Volume with Trend')
    
    return fig, ax


# Export key functions and classes for easy imports
__all__ = [
    'PlotStyle',
    'DEFAULT_STYLE',
    'setup_plot_style',
    'apply_professional_styling',
    'create_figure_with_style',
    'save_plot',
    'create_line_plot',
    'create_scatter_plot', 
    'create_box_plot',
    'create_heatmap',
    'plot_price_evolution',
    'plot_volume_analysis',
    'create_summary_plots',
    'create_detailed_analysis_plots',
    'create_parameter_sweep_plots',
    'generate_summary_report'
]

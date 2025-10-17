"""Feels Floor Price Simulation Package

Core simulation framework for analyzing Feels Protocol dynamics:
- Floor price advancement through Protocol-Owned Market Making (POMM)
- Fee distribution optimization (Buffer/protocol/creator splits)
- Participant behavior modeling (retail, algo, LP, arbitrageur)
- JIT liquidity system for market bootstrapping
- Comprehensive metrics and reporting
"""

__version__ = "0.1.0"

# Import core simulation components
from .core import FeelsSimulation, FeelsMarketModel
from .state import (
    MarketState,
    FloorFundingState,
    SimulationSnapshot,
    SimulationResults,
)
from .config import SimulationConfig

# Import participant system
from .participants import (
    ParticipantConfig,
    ParticipantRegistry,
    LiquidityPosition,
    aggregate_participant_metrics,
    create_participant_registry,
)

# Import pricing utilities
from .pricing import (
    tick_to_price,
    price_to_tick,
    tick_to_sqrt_price_x64,
    JitMetrics,
    get_twap_from_dataframe,
    get_volatility_from_dataframe,
)

# Import metrics and analysis
from .metrics import (
    calculate_key_metrics,
    calculate_hourly_aggregates,
    calculate_daily_aggregates,
    calculate_weekly_aggregates,
    calculate_floor_to_market_ratio_stats,
    calculate_pomm_efficiency_metrics,
    export_metrics_to_file,
    snapshots_to_dataframe,
    calculate_twap_and_volatility,
)
from .analysis import (
    analyze_results,
    calculate_floor_floor_ratio_stats,
    calculate_pomm_efficiency_metrics,
    calculate_volume_elasticity,
)
from .plotting import (
    create_summary_plots,
    create_detailed_analysis_plots,
    create_parameter_sweep_plots,
    generate_summary_report,
    setup_plot_style,
    apply_professional_styling,
    create_figure_with_style,
    save_plot,
    create_line_plot,
    create_scatter_plot,
    create_box_plot,
    create_heatmap,
    plot_price_evolution,
    plot_volume_analysis,
    PlotStyle,
    DEFAULT_STYLE,
)

# Import other utilities
from .market import MarketEnvironment

__all__ = [
    # Core simulation
    "FeelsSimulation",
    "FeelsMarketModel",
    "SimulationConfig",
    
    # State management
    "MarketState",
    "FloorFundingState", 
    "SimulationSnapshot",
    "SimulationResults",
    
    # Participants
    "ParticipantConfig",
    "LiquidityPosition",
    "ParticipantRegistry",
    "aggregate_participant_metrics",
    "create_participant_registry",
    
    # Pricing utilities
    "tick_to_price",
    "price_to_tick",
    "tick_to_sqrt_price_x64",
    "JitMetrics",
    "get_twap_from_dataframe",
    "get_volatility_from_dataframe",
    
    # Metrics and analysis
    "calculate_key_metrics",
    "calculate_hourly_aggregates",
    "calculate_daily_aggregates",
    "calculate_weekly_aggregates",
    "export_metrics_to_file",
    "snapshots_to_dataframe",
    "calculate_twap_and_volatility",
    "analyze_results",
    "calculate_floor_floor_ratio_stats",
    "calculate_pomm_efficiency_metrics",
    "calculate_volume_elasticity",
    
    # Visualization
    "create_summary_plots",
    "create_detailed_analysis_plots",
    "create_parameter_sweep_plots",
    "generate_summary_report",
    "setup_plot_style",
    "apply_professional_styling",
    "create_figure_with_style",
    "save_plot",
    "create_line_plot",
    "create_scatter_plot",
    "create_box_plot",
    "create_heatmap",
    "plot_price_evolution",
    "plot_volume_analysis",
    "PlotStyle",
    "DEFAULT_STYLE",
    
    # Utilities
    "MarketEnvironment",
]
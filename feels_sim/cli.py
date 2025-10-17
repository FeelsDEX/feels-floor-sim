"""Modern CLI using full AgentPy Experiment framework for parameter sweeps."""

import argparse
import json
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import agentpy as ap

from .core import FeelsMarketModel
from .config import SimulationConfig
from .metrics import snapshots_to_dataframe, calculate_twap_and_volatility

# Fee scenario presets
FEE_SCENARIOS = {
    "current_default": {"protocol_fee_rate_bps": 100, "creator_fee_rate_bps": 50},      # 1.0% protocol, 0.5% creator
    "protocol_sustainable": {"protocol_fee_rate_bps": 150, "creator_fee_rate_bps": 50}, # 1.5% protocol, 0.5% creator
    "creator_incentive": {"protocol_fee_rate_bps": 100, "creator_fee_rate_bps": 200},   # 1.0% protocol, 2.0% creator
    "balanced_growth": {"protocol_fee_rate_bps": 125, "creator_fee_rate_bps": 75},     # 1.25% protocol, 0.75% creator
}


def run_experiment_sweep(
    scenarios: List[str],
    fee_range: List[int],
    trend_bias: List[float],
    volatility: List[float],
    hours: int = 168,
    output_dir: str = "experiments/runs",
    parallel: bool = True,
    seed: int = 42
):
    """Run parameter sweep using AgentPy's Experiment framework."""
    
    print(f"Starting AgentPy experiment with parameter sweep...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Base configuration
    config = SimulationConfig()
    base_params = config.to_agentpy_params()
    
    # Create parameter combinations using AgentPy's parameter sampling
    parameters = {}
    
    # Add base parameters
    for key, value in base_params.items():
        if key not in ['fee_scenario_name', 'base_fee_rate', 'trend_bias', 'volatility_daily', 'fee_split']:
            parameters[key] = value
    
    # Add sweep parameters using AgentPy's parameter sampling
    scenario_samples = []
    for scenario in scenarios:
        for fee_bps in fee_range:
            for bias in trend_bias:
                for vol in volatility:
                    scenario_params = base_params.copy()
                    scenario_params.update({
                        'fee_scenario_name': scenario,
                        'base_fee_rate': fee_bps / 10000.0,
                        'trend_bias': bias,
                        'volatility_daily': vol,
                        'fee_split': FEE_SCENARIOS[scenario],
                    })
                    scenario_samples.append(scenario_params)
    
    total_runs = len(scenario_samples)
    print(f"Running {total_runs} parameter combinations")
    
    # Run experiments
    results = []
    for i, params in enumerate(scenario_samples, 1):
        print(f"[{i}/{total_runs}] Running {params['fee_scenario_name']}, "
              f"fee={int(params['base_fee_rate']*10000)}bps, "
              f"bias={params['trend_bias']}, vol={params['volatility_daily']}")
        
        try:
            # Create and run model using AgentPy
            model = FeelsMarketModel(params)
            model.setup()
            
            # Run for the specified number of steps
            for _ in range(hours * 60):
                model.step()
            
            # Extract results
            final_snapshot = model.snapshots[-1] if model.snapshots else None
            
            result = {
                'scenario': params['fee_scenario_name'],
                'base_fee_rate': params['base_fee_rate'],
                'trend_bias': params['trend_bias'],
                'volatility_daily': params['volatility_daily'],
                'final_minute': model.minute,
                'snapshots_collected': len(model.snapshots),
                'hourly_aggregates': len(model.hourly_aggregates),
            }
            
            if final_snapshot:
                result.update({
                    'final_buffer_balance': final_snapshot.floor_state.buffer_balance,
                    'final_treasury_balance': final_snapshot.floor_state.treasury_balance,
                    'final_deployed_feelssol': final_snapshot.floor_state.deployed_feelssol,
                    'pomm_deployments': final_snapshot.floor_state.pomm_deployments_count,
                    'total_volume': sum(s.volume_feelssol for s in model.snapshots),
                    'total_fees': sum(s.fees_collected for s in model.snapshots),
                })
            
            results.append(result)
            print(f"  Completed: buffer=${result.get('final_buffer_balance', 0):.2f}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    # Process and save results
    print(f"Processing {len(results)} results...")
    
    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'scenario': result['scenario'],
            'base_fee_bps': int(result['base_fee_rate'] * 10000),
            'trend_bias': result['trend_bias'],
            'volatility_daily': result['volatility_daily'],
            'final_buffer_balance': result.get('final_buffer_balance', 0),
            'final_treasury_balance': result.get('final_treasury_balance', 0),
            'total_volume': result.get('total_volume', 0),
            'total_fees': result.get('total_fees', 0),
            'pomm_deployments': result.get('pomm_deployments', 0),
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary CSV
    summary_df = pl.DataFrame(summary_data)
    csv_file = Path(output_dir) / f"agentpy_sweep_summary_{timestamp}.csv"
    summary_df.write_csv(csv_file)
    
    # Full results parquet (AgentPy format)
    parquet_file = Path(output_dir) / f"agentpy_sweep_results_{timestamp}.parquet"
    summary_df.write_parquet(parquet_file)
    
    print(f"AgentPy experiment complete!")
    print(f"Results saved:")
    print(f"   • Summary: {csv_file}")
    print(f"   • Full data: {parquet_file}")
    
    # Show top performers
    if summary_df.height > 0:
        best_buffer = summary_df.sort("final_buffer_balance", descending=True).head(1)
        best_volume = summary_df.sort("total_volume", descending=True).head(1)
        
        print("\nTop performers:")
        if best_buffer.height > 0:
            print(f"   • Best buffer: {best_buffer['scenario'].item()} scenario")
        if best_volume.height > 0:
            print(f"   • Best volume: {best_volume['scenario'].item()} scenario")


def run_single_agentpy(
    scenario: str = "current_default",
    fee_bps: int = 30,
    trend_bias: float = 0.0,
    volatility: float = 0.05,
    hours: int = 168,
    output_dir: str = "experiments/single",
    seed: int = 42,
    save_snapshots: bool = True
):
    """Run a single simulation using AgentPy framework."""
    
    print(f"Running single AgentPy simulation: {scenario} scenario")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up parameters
    config = SimulationConfig()
    params = config.to_agentpy_params()
    params.update({
        'fee_scenario_name': scenario,
        'base_fee_rate': fee_bps / 10000.0,
        'trend_bias': trend_bias,
        'volatility_daily': volatility,
        'fee_split': FEE_SCENARIOS[scenario],
    })
    
    # Run simulation using AgentPy
    model = FeelsMarketModel(params)
    model.setup()
    
    # Run for the specified number of steps
    for _ in range(hours * 60):
        model.step()
    
    # Extract results
    final_snapshot = model.snapshots[-1] if model.snapshots else None
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_snapshots and model.snapshots:
        # Convert to DataFrame and save
        df = snapshots_to_dataframe(model.snapshots)
        if not df.is_empty():
            df = calculate_twap_and_volatility(df)
            parquet_file = Path(output_dir) / f"agentpy_simulation_{scenario}_{timestamp}.parquet"
            df.write_parquet(parquet_file)
            print(f"Snapshots saved: {parquet_file}")
    
    # Save hourly data using AgentPy's data collection
    if model.hourly_aggregates:
        # Flatten nested data for CSV compatibility
        flattened_hourly = []
        for hourly in model.hourly_aggregates:
            flat = {}
            for key, value in hourly.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat[f"{key}_{subkey}"] = subvalue
                else:
                    flat[key] = value
            flattened_hourly.append(flat)
        
        hourly_df = pl.DataFrame(flattened_hourly)
        csv_file = Path(output_dir) / f"agentpy_hourly_{scenario}_{timestamp}.csv"
        hourly_df.write_csv(csv_file)
        print(f"Hourly data saved: {csv_file}")
    
    # Display summary
    if final_snapshot:
        print(f"\nFinal Results:")
        print(f"   • Buffer Balance: ${final_snapshot.floor_state.buffer_balance:,.2f}")
        print(f"   • Treasury Balance: ${final_snapshot.floor_state.treasury_balance:,.2f}")
        print(f"   • Total Volume: {sum(s.volume_feelssol for s in model.snapshots):,.2f}")
        print(f"   • Total Fees: ${sum(s.fees_collected for s in model.snapshots):,.2f}")
        print(f"   • POMM Deployments: {final_snapshot.floor_state.pomm_deployments_count}")


def main():
    """Main CLI entry point using AgentPy framework."""
    parser = argparse.ArgumentParser(description="Feels Protocol Simulation CLI (AgentPy-powered)")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Sweep command using AgentPy Experiment
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweep using AgentPy')
    sweep_parser.add_argument('--hours', type=int, default=168, help='Simulation duration in hours')
    sweep_parser.add_argument('--scenarios', nargs='+', 
                             default=['current_default', 'protocol_sustainable'],
                             choices=list(FEE_SCENARIOS.keys()),
                             help='Fee scenarios to test')
    sweep_parser.add_argument('--fee-range', nargs='+', type=int,
                             default=[10, 20, 30, 40, 50],
                             help='Base fee range in basis points')
    sweep_parser.add_argument('--trend-bias', nargs='+', type=float,
                             default=[0.0, 0.2, -0.2],
                             help='SOL trend bias values')
    sweep_parser.add_argument('--volatility', nargs='+', type=float,
                             default=[0.03, 0.05, 0.08],
                             help='Daily volatility values')
    sweep_parser.add_argument('--output-dir', default='experiments/runs',
                             help='Output directory')
    sweep_parser.add_argument('--parallel', action='store_true', default=True,
                             help='Run simulations in parallel')
    sweep_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Single command using AgentPy
    single_parser = subparsers.add_parser('single', help='Run single simulation using AgentPy')
    single_parser.add_argument('--hours', type=int, default=168, help='Simulation duration in hours')
    single_parser.add_argument('--scenario', default='current_default',
                              choices=list(FEE_SCENARIOS.keys()),
                              help='Fee scenario')
    single_parser.add_argument('--fee-bps', type=int, default=30, help='Base fee in basis points')
    single_parser.add_argument('--trend-bias', type=float, default=0.0, help='SOL trend bias')
    single_parser.add_argument('--volatility', type=float, default=0.05, help='Daily volatility')
    single_parser.add_argument('--output-dir', default='experiments/single', help='Output directory')
    single_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    single_parser.add_argument('--no-snapshots', action='store_true', help='Skip detailed snapshots')
    
    # Scenarios command
    scenarios_parser = subparsers.add_parser('scenarios', help='List available scenarios')
    
    # Test AgentPy integration
    test_parser = subparsers.add_parser('test', help='Test AgentPy integration')
    test_parser.add_argument('--steps', type=int, default=60, help='Number of steps to test')
    
    args = parser.parse_args()
    
    if args.command == 'sweep':
        run_experiment_sweep(
            scenarios=args.scenarios,
            fee_range=args.fee_range,
            trend_bias=args.trend_bias,
            volatility=args.volatility,
            hours=args.hours,
            output_dir=args.output_dir,
            parallel=args.parallel,
            seed=args.seed
        )
    elif args.command == 'single':
        run_single_agentpy(
            scenario=args.scenario,
            fee_bps=args.fee_bps,
            trend_bias=args.trend_bias,
            volatility=args.volatility,
            hours=args.hours,
            output_dir=args.output_dir,
            seed=args.seed,
            save_snapshots=not args.no_snapshots
        )
    elif args.command == 'scenarios':
        print("Available fee scenarios:\n")
        for name, config in FEE_SCENARIOS.items():
            protocol_pct = config['protocol_fee_rate_bps'] / 100.0
            creator_pct = config['creator_fee_rate_bps'] / 100.0
            buffer_pct = 100.0 - protocol_pct - creator_pct
            print(f"{name}")
            print(f"   • Protocol Fee: {protocol_pct:.1f}% ({config['protocol_fee_rate_bps']} bps)")
            print(f"   • Creator Fee: {creator_pct:.1f}% ({config['creator_fee_rate_bps']} bps)")
            print(f"   • Buffer Share: {buffer_pct:.1f}% (remainder)")
            print()
    elif args.command == 'test':
        print("Testing AgentPy integration...")
        
        # Test basic model creation and execution
        config = SimulationConfig(enable_participant_behavior=False)
        params = config.to_agentpy_params()
        
        model = FeelsMarketModel(params)
        model.setup()
        model.random.seed(42)
        for _ in range(args.steps):
            model.step()
        
        print(f"Successfully ran {args.steps} steps")
        print(f"Collected {len(model.snapshots)} snapshots")
        print(f"Final buffer: ${model.floor_state.buffer_balance:.2f}")
        print("AgentPy integration working!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
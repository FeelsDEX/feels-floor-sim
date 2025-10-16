"""
CLI module for running parameter sweeps and comparative analysis.

This module provides command-line interfaces for systematic exploration of fee ranges,
funding strategies, and market conditions to identify optimal protocol parameters.

Usage:
    python -m feels_sim.cli sweep --sweep-type full --hours 168
    python -m feels_sim.cli sweep --sweep-type fee_range
    python -m feels_sim.cli sweep --sweep-type market_conditions
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import sys

from .config import SimulationConfig
from .core import FeelsSimulation
from .metrics import analyze_results, generate_summary_report, MetricsCollector


def run_single_scenario(scenario_name: str, config_overrides: Dict[str, Any], 
                       hours: int = 168) -> Dict[str, Any]:
    """Run a single simulation scenario and return results."""
    try:
        # Create configuration
        if scenario_name in ["current_default", "protocol_sustainable", "creator_incentive", 
                           "balanced_growth", "maximum_protocol"]:
            config = SimulationConfig.create_fee_scenario(scenario_name, **config_overrides)
        else:
            config = SimulationConfig(**config_overrides)
        
        config.validate()
        
        # Run simulation
        sim = FeelsSimulation(config)
        results = sim.run(hours=hours)
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Return comprehensive results
        return {
            "scenario_name": scenario_name,
            "config": {
                "treasury_share_pct": config.treasury_share_pct,
                "creator_share_pct": config.creator_share_pct,
                "buffer_share_pct": config.buffer_share_pct,
                "base_fee_bps": config.base_fee_bps,
                "sol_volatility_daily": config.sol_volatility_daily,
                "jitosol_yield_apy": config.jitosol_yield_apy,
            },
            "analysis": analysis,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "scenario_name": scenario_name,
            "config": config_overrides,
            "analysis": {},
            "success": False,
            "error": str(e)
        }


def run_fee_range_sweep(base_fees: List[int], scenarios: List[str], 
                       hours: int = 168) -> List[Dict[str, Any]]:
    """Run parameter sweep across different fee ranges and scenarios."""
    results = []
    total_runs = len(base_fees) * len(scenarios)
    current_run = 0
    
    print(f"Running fee range sweep: {len(base_fees)} fee levels × {len(scenarios)} scenarios = {total_runs} total runs")
    
    for base_fee in base_fees:
        for scenario in scenarios:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Running {scenario} with {base_fee} bps fee...")
            
            start_time = time.time()
            result = run_single_scenario(
                scenario_name=scenario,
                config_overrides={"base_fee_bps": base_fee},
                hours=hours
            )
            elapsed = time.time() - start_time
            
            result["run_time_seconds"] = elapsed
            result["base_fee_bps"] = base_fee
            results.append(result)
            
            if result["success"]:
                analysis = result["analysis"]
                print(f"  ✓ Completed in {elapsed:.1f}s - Floor growth: {analysis.get('floor_growth_rate_annual', 0):.2%}, "
                      f"POMM deployments: {analysis.get('pomm_deployments', 0)}")
            else:
                print(f"  ✗ Failed: {result['error']}")
    
    return results


def run_market_condition_sweep(scenarios: List[str], hours: int = 168) -> List[Dict[str, Any]]:
    """Run parameter sweep across different market conditions."""
    market_conditions = [
        {"name": "bull_market", "sol_trend_bias": 0.2, "sol_volatility_daily": 0.04},
        {"name": "bear_market", "sol_trend_bias": -0.2, "sol_volatility_daily": 0.07},
        {"name": "sideways_market", "sol_trend_bias": 0.0, "sol_volatility_daily": 0.05},
        {"name": "high_volatility", "sol_trend_bias": 0.0, "sol_volatility_daily": 0.10},
    ]
    
    results = []
    total_runs = len(market_conditions) * len(scenarios)
    current_run = 0
    
    print(f"Running market condition sweep: {len(market_conditions)} conditions × {len(scenarios)} scenarios = {total_runs} total runs")
    
    for condition in market_conditions:
        for scenario in scenarios:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Running {scenario} in {condition['name']}...")
            
            start_time = time.time()
            result = run_single_scenario(
                scenario_name=scenario,
                config_overrides={
                    "sol_trend_bias": condition["sol_trend_bias"],
                    "sol_volatility_daily": condition["sol_volatility_daily"]
                },
                hours=hours
            )
            elapsed = time.time() - start_time
            
            result["run_time_seconds"] = elapsed
            result["market_condition"] = condition["name"]
            results.append(result)
            
            if result["success"]:
                analysis = result["analysis"]
                print(f"  ✓ Completed in {elapsed:.1f}s - Floor growth: {analysis.get('floor_growth_rate_annual', 0):.2%}")
            else:
                print(f"  ✗ Failed: {result['error']}")
    
    return results


def generate_comparative_summary(results: List[Dict[str, Any]], output_file: str):
    """Generate comparative analysis summary."""
    # Filter successful results
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        print("No successful results to analyze")
        return
    
    # Create summary report
    summary = {
        "sweep_metadata": {
            "total_runs": len(results),
            "successful_runs": len(successful_results),
            "failed_runs": len(results) - len(successful_results),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "scenarios": {}
    }
    
    # Group by scenario
    scenario_groups = {}
    for result in successful_results:
        scenario = result["scenario_name"]
        if scenario not in scenario_groups:
            scenario_groups[scenario] = []
        scenario_groups[scenario].append(result)
    
    # Analyze each scenario group
    for scenario, scenario_results in scenario_groups.items():
        analyses = [r["analysis"] for r in scenario_results]
        
        scenario_summary = {
            "run_count": len(scenario_results),
            "config_variations": [r["config"] for r in scenario_results],
            "metrics": {
                "floor_growth_rate_annual": {
                    "mean": sum(a.get("floor_growth_rate_annual", 0) for a in analyses) / len(analyses),
                    "min": min(a.get("floor_growth_rate_annual", 0) for a in analyses),
                    "max": max(a.get("floor_growth_rate_annual", 0) for a in analyses),
                },
                "avg_floor_to_market_ratio": {
                    "mean": sum(a.get("avg_floor_to_market_ratio", 0) for a in analyses) / len(analyses),
                    "min": min(a.get("avg_floor_to_market_ratio", 0) for a in analyses),
                    "max": max(a.get("avg_floor_to_market_ratio", 0) for a in analyses),
                },
                "pomm_deployments": {
                    "mean": sum(a.get("pomm_deployments", 0) for a in analyses) / len(analyses),
                    "min": min(a.get("pomm_deployments", 0) for a in analyses),
                    "max": max(a.get("pomm_deployments", 0) for a in analyses),
                },
                "total_volume": {
                    "mean": sum(a.get("total_volume", 0) for a in analyses) / len(analyses),
                    "min": min(a.get("total_volume", 0) for a in analyses),
                    "max": max(a.get("total_volume", 0) for a in analyses),
                },
                "protocol_efficiency": {
                    "mean": sum(a.get("protocol_efficiency", 0) for a in analyses) / len(analyses),
                    "min": min(a.get("protocol_efficiency", 0) for a in analyses),
                    "max": max(a.get("protocol_efficiency", 0) for a in analyses),
                }
            }
        }
        
        summary["scenarios"][scenario] = scenario_summary
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Comparative summary saved to {output_file}")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    
    best_floor_growth = max(summary["scenarios"].items(), 
                           key=lambda x: x[1]["metrics"]["floor_growth_rate_annual"]["mean"])
    best_efficiency = max(summary["scenarios"].items(), 
                         key=lambda x: x[1]["metrics"]["protocol_efficiency"]["mean"])
    
    print(f"Best floor growth: {best_floor_growth[0]} ({best_floor_growth[1]['metrics']['floor_growth_rate_annual']['mean']:.2%} annual)")
    print(f"Best efficiency: {best_efficiency[0]} ({best_efficiency[1]['metrics']['protocol_efficiency']['mean']:.4f} USD/FeelsSOL)")


def sweep_command(args):
    """Handle the sweep subcommand."""
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting parameter sweep: {args.sweep_type}")
    print(f"Duration: {args.hours} hours")
    print(f"Scenarios: {args.scenarios}")
    print(f"Output directory: {output_dir}")
    
    results = []
    
    if args.sweep_type in ["fee_range", "full"]:
        print("\n=== RUNNING FEE RANGE SWEEP ===")
        fee_results = run_fee_range_sweep(args.fee_range, args.scenarios, args.hours)
        results.extend(fee_results)
        
        # Save fee range results
        fee_output = output_dir / f"fee_sweep_{timestamp}.json"
        with open(fee_output, 'w') as f:
            json.dump(fee_results, f, indent=2)
        print(f"Fee range results saved to {fee_output}")
    
    if args.sweep_type in ["market_conditions", "full"]:
        print("\n=== RUNNING MARKET CONDITIONS SWEEP ===")
        market_results = run_market_condition_sweep(args.scenarios, args.hours)
        results.extend(market_results)
        
        # Save market conditions results
        market_output = output_dir / f"market_sweep_{timestamp}.json"
        with open(market_output, 'w') as f:
            json.dump(market_results, f, indent=2)
        print(f"Market conditions results saved to {market_output}")
    
    # Generate comparative summary
    if results:
        summary_output = output_dir / f"comparative_summary_{timestamp}.json"
        generate_comparative_summary(results, str(summary_output))
        
        # Save all results
        all_results_output = output_dir / f"all_results_{timestamp}.json"
        with open(all_results_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"All results saved to {all_results_output}")
    
    print(f"\nParameter sweep completed. Results saved in {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Feels Protocol CLI Tools",
        prog="python -m feels_sim.cli"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Sweep subcommand
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweeps")
    sweep_parser.add_argument("--sweep-type", choices=["fee_range", "market_conditions", "full"], 
                             default="fee_range", help="Type of parameter sweep to run")
    sweep_parser.add_argument("--hours", type=int, default=168, 
                             help="Simulation duration in hours (default: 168 = 1 week)")
    sweep_parser.add_argument("--output-dir", type=str, default="experiments/runs", 
                             help="Output directory for results")
    sweep_parser.add_argument("--scenarios", nargs="+", 
                             default=["current_default", "protocol_sustainable", "creator_incentive", "balanced_growth"],
                             help="Fee scenarios to test")
    sweep_parser.add_argument("--fee-range", nargs="+", type=int, 
                             default=[10, 20, 30, 40, 50, 75, 100],
                             help="Base fee range in basis points")
    sweep_parser.set_defaults(func=sweep_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Call the appropriate function
    args.func(args)


if __name__ == "__main__":
    main()
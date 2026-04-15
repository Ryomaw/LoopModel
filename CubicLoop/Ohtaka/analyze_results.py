#!/usr/bin/env python
"""
Post-processing and analysis tools for Monte Carlo simulation results.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results_csv(csv_path: str) -> List[Dict]:
    """Load simulation results from CSV file."""
    float_fields = {
        "n", "x", "thermalization", "observation",
        "average_length", "length_jackknife_error",
        "average_length_squared", "length_squared_jackknife_error",
        "average_length_cubed", "length_cubed_jackknife_error",
        "average_length_fourth", "length_fourth_jackknife_error",
        "average_length_per_cluster", "length_per_cluster_jackknife_error",
        "average_plaquette_order_x", "plaquette_order_x_jackknife_error",
        "average_plaquette_order_y", "plaquette_order_y_jackknife_error",
        "heat_capacity_like", "heat_capacity_like_jackknife_error",
        "average_cluster_count", "cluster_jackknife_error",
        "average_cluster_squared", "cluster_squared_jackknife_error",
        "average_cluster_cubed", "cluster_cubed_jackknife_error",
        "average_cluster_fourth", "cluster_fourth_jackknife_error",
        "cluster_heat_capacity_like", "cluster_heat_capacity_like_jackknife_error",
        "acceptance_rate",
        "winding_sector_even_even_fraction", "winding_sector_even_odd_fraction",
        "winding_sector_odd_even_fraction", "winding_sector_odd_odd_fraction",
    }
    int_fields = {
        "lattice_size", "thermalization", "observation",
        "jackknife_block_size", "jackknife_block_count", "effective_samples",
        "winding_sector_even_even_count", "winding_sector_even_odd_count",
        "winding_sector_odd_even_count", "winding_sector_odd_odd_count",
    }

    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {}
            for key, value in row.items():
                if key in int_fields:
                    result[key] = int(value)
                elif key in float_fields:
                    result[key] = float(value)
                else:
                    result[key] = value
            results.append(result)
    return results


def print_summary(results: List[Dict]) -> None:
    """Print summary statistics of simulation results."""
    if not results:
        print("No results to summarize.")
        return

    print("=" * 80)
    print("Simulation Results Summary")
    print("=" * 80)
    print()

    # Basic statistics
    print(f"Total simulations: {len(results)}")
    print(f"Lattice size: {results[0]['lattice_size']}")
    print()

    # n values
    n_values = sorted(set(r["n"] for r in results))
    print(f"n values: {n_values}")
    print(f"Number of n: {len(n_values)}")
    print()

    # x values
    x_values = sorted(set(r["x"] for r in results))
    print(f"x range: {min(x_values):.2f} to {max(x_values):.2f}")
    print(f"Number of x: {len(x_values)}")
    print()

    # ts/os configurations
    ts_os_configs = sorted(set((r["thermalization"], r["observation"]) for r in results))
    print(f"ts/os configurations:")
    for ts, os in ts_os_configs:
        count = sum(1 for r in results if r["thermalization"] == ts and r["observation"] == os)
        print(f"  (ts={ts:5d}, os={os:6d}): {count} simulations")
    print()

    # Quality metrics
    acceptance_rates = [r["acceptance_rate"] for r in results]
    print(f"Acceptance rate:")
    print(f"  Mean: {np.mean(acceptance_rates):.4f}")
    print(f"  Min:  {np.min(acceptance_rates):.4f}")
    print(f"  Max:  {np.max(acceptance_rates):.4f}")
    print()

    # Observable ranges
    avg_lengths = [r["average_length"] for r in results]
    print(f"Average length <l>:")
    print(f"  Range: {np.min(avg_lengths):.2f} to {np.max(avg_lengths):.2f}")
    print(f"  Mean:  {np.mean(avg_lengths):.2f}")
    print(f"  Std:   {np.std(avg_lengths):.2f}")
    print()

    print("=" * 80)


def export_by_n(results: List[Dict], output_dir: str) -> None:
    """Export results separated by n value."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_values = sorted(set(r["n"] for r in results))
    
    for n in n_values:
        n_results = [r for r in results if r["n"] == n]
        
        csv_file = output_path / f"results_n_{n:.0f}.csv"
        fieldnames = list(n_results[0].keys())
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(n_results)
        
        print(f"Exported: {csv_file} ({len(n_results)} rows)")


def export_by_ts_os(results: List[Dict], output_dir: str) -> None:
    """Export results separated by (ts, os) configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ts_os_configs = sorted(set((r["thermalization"], r["observation"]) for r in results))
    
    for ts, os in ts_os_configs:
        config_results = [r for r in results if r["thermalization"] == ts and r["observation"] == os]
        
        csv_file = output_path / f"results_ts{ts}_os{os}.csv"
        fieldnames = list(config_results[0].keys())
        
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(config_results)
        
        print(f"Exported: {csv_file} ({len(config_results)} rows)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process Monte Carlo simulation results"
    )
    parser.add_argument(
        "csv_file",
        help="Path to results CSV file"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary statistics"
    )
    parser.add_argument(
        "--export-by-n", type=str, metavar="DIR",
        help="Export results separated by n value to directory"
    )
    parser.add_argument(
        "--export-by-ts-os", type=str, metavar="DIR",
        help="Export results separated by (ts, os) configuration to directory"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Loading results from {csv_path}...")
    results = load_results_csv(str(csv_path))
    print(f"Loaded {len(results)} results.\n")

    if args.summary:
        print_summary(results)
    
    if args.export_by_n:
        print(f"\nExporting by n to {args.export_by_n}...")
        export_by_n(results, args.export_by_n)
    
    if args.export_by_ts_os:
        print(f"\nExporting by (ts, os) to {args.export_by_ts_os}...")
        export_by_ts_os(results, args.export_by_ts_os)

    if not any([args.summary, args.export_by_n, args.export_by_ts_os]):
        print("No actions specified. Use --summary, --export-by-n, or --export-by-ts-os")
        print("\nExample usage:")
        print(f"  python {sys.argv[0]} results.csv --summary")
        print(f"  python {sys.argv[0]} results.csv --export-by-n ./by_n")
        print(f"  python {sys.argv[0]} results.csv --export-by-ts-os ./by_config")


if __name__ == "__main__":
    main()

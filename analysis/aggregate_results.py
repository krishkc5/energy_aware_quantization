#!/usr/bin/env python3
"""
Aggregate and analyze results from multiple experimental trials.

This script loads all CSV results from the results/ directory,
computes summary statistics, and generates comparison tables.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def load_all_results(results_dir: Path) -> dict:
    """
    Load all result CSV files.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary mapping precision to list of DataFrames
    """
    results = {
        "fp32": [],
        "fp16": [],
        "int8": [],
    }

    for precision in ["fp32", "fp16", "int8"]:
        precision_dir = results_dir / precision
        if not precision_dir.exists():
            print(f"Warning: {precision_dir} does not exist")
            continue

        csv_files = list(precision_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files for {precision.upper()}")

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            results[precision].append(df)

    return results


def aggregate_trials(dfs: list) -> pd.DataFrame:
    """
    Aggregate multiple trial DataFrames into summary statistics.

    Args:
        dfs: List of DataFrames (one per trial)

    Returns:
        DataFrame with mean, std, min, max for each metric
    """
    if len(dfs) == 0:
        return pd.DataFrame()

    # Concatenate all trials
    all_trials = pd.concat(dfs, ignore_index=True)

    # Metrics to aggregate
    numeric_cols = all_trials.select_dtypes(include=[np.number]).columns

    # Compute statistics
    stats = pd.DataFrame({
        "mean": all_trials[numeric_cols].mean(),
        "std": all_trials[numeric_cols].std(),
        "min": all_trials[numeric_cols].min(),
        "max": all_trials[numeric_cols].max(),
        "n_trials": len(dfs),
    })

    return stats


def create_comparison_table(results_dict: dict) -> pd.DataFrame:
    """
    Create comparison table across precision modes.

    Args:
        results_dict: Dictionary mapping precision to list of DataFrames

    Returns:
        Comparison DataFrame
    """
    comparison_metrics = [
        "mean_latency",
        "throughput",
        "mean_power_w",
        "energy_per_inference_mj",
        "accuracy",
        "model_size_mb",
        "peak_allocated_gb",
    ]

    comparison_data = []

    for precision in ["fp32", "fp16", "int8"]:
        dfs = results_dict[precision]
        if len(dfs) == 0:
            continue

        all_trials = pd.concat(dfs, ignore_index=True)

        row = {"precision": precision.upper()}
        for metric in comparison_metrics:
            if metric in all_trials.columns:
                mean_val = all_trials[metric].mean()
                std_val = all_trials[metric].std()
                row[f"{metric}_mean"] = mean_val
                row[f"{metric}_std"] = std_val

        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def print_summary_table(comparison_df: pd.DataFrame):
    """Print formatted summary table."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY - COMPARISON ACROSS PRECISION MODES")
    print("="*80)

    if len(comparison_df) == 0:
        print("No results found.")
        return

    # Key metrics to display
    metrics = {
        "Mean Latency (ms)": "mean_latency_mean",
        "Throughput (samples/s)": "throughput_mean",
        "Mean Power (W)": "mean_power_w_mean",
        "Energy/Inference (mJ)": "energy_per_inference_mj_mean",
        "Accuracy (%)": "accuracy_mean",
        "Model Size (MB)": "model_size_mb_mean",
        "Peak Memory (GB)": "peak_allocated_gb_mean",
    }

    print(f"\n{'Precision':<12}", end="")
    for label in metrics.keys():
        print(f"{label:>20}", end="")
    print()
    print("-" * 80)

    for _, row in comparison_df.iterrows():
        precision = row["precision"]
        print(f"{precision:<12}", end="")

        for metric_col in metrics.values():
            if metric_col in row:
                val = row[metric_col]

                # Format based on metric
                if "latency" in metric_col:
                    val = val * 1000  # Convert to ms
                    print(f"{val:>20.2f}", end="")
                elif "accuracy" in metric_col:
                    val = val * 100  # Convert to percentage
                    print(f"{val:>20.2f}", end="")
                else:
                    print(f"{val:>20.2f}", end="")
            else:
                print(f"{'N/A':>20}", end="")
        print()

    print("="*80)


def compute_improvements(comparison_df: pd.DataFrame):
    """Compute improvements relative to FP32 baseline."""
    print("\n" + "="*80)
    print("IMPROVEMENTS RELATIVE TO FP32 BASELINE")
    print("="*80)

    if len(comparison_df) < 2:
        print("Need at least 2 precision modes for comparison")
        return

    # Find FP32 baseline
    fp32_row = comparison_df[comparison_df["precision"] == "FP32"]
    if len(fp32_row) == 0:
        print("No FP32 baseline found")
        return

    fp32_row = fp32_row.iloc[0]

    print(f"\n{'Precision':<12}{'Speedup':>15}{'Energy Reduction':>20}{'Accuracy Loss':>20}")
    print("-" * 80)

    for _, row in comparison_df.iterrows():
        if row["precision"] == "FP32":
            continue

        precision = row["precision"]

        # Speedup = baseline latency / current latency
        if "mean_latency_mean" in row and "mean_latency_mean" in fp32_row:
            speedup = fp32_row["mean_latency_mean"] / row["mean_latency_mean"]
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        # Energy reduction
        if "energy_per_inference_mj_mean" in row and "energy_per_inference_mj_mean" in fp32_row:
            energy_reduction = (
                (fp32_row["energy_per_inference_mj_mean"] - row["energy_per_inference_mj_mean"])
                / fp32_row["energy_per_inference_mj_mean"] * 100
            )
            energy_str = f"{energy_reduction:.1f}%"
        else:
            energy_str = "N/A"

        # Accuracy loss
        if "accuracy_mean" in row and "accuracy_mean" in fp32_row:
            accuracy_loss = (fp32_row["accuracy_mean"] - row["accuracy_mean"]) * 100
            accuracy_str = f"{accuracy_loss:.2f}%"
        else:
            accuracy_str = "N/A"

        print(f"{precision:<12}{speedup_str:>15}{energy_str:>20}{accuracy_str:>20}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze experimental results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for aggregated results"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    print("="*80)
    print("AGGREGATING EXPERIMENTAL RESULTS")
    print("="*80)
    print(f"Results directory: {results_dir}")

    # Load all results
    results = load_all_results(results_dir)

    # Create comparison table
    comparison_df = create_comparison_table(results)

    # Print summary
    print_summary_table(comparison_df)

    # Print improvements
    compute_improvements(comparison_df)

    # Save aggregated results
    if args.output:
        output_path = Path(args.output)
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Aggregated results saved to: {output_path}")
    else:
        # Default output
        output_path = results_dir / "aggregated_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Aggregated results saved to: {output_path}")

    # Print individual precision statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS BY PRECISION")
    print("="*80)

    for precision in ["fp32", "fp16", "int8"]:
        dfs = results[precision]
        if len(dfs) == 0:
            continue

        print(f"\n{precision.upper()}:")
        print("-" * 40)

        stats = aggregate_trials(dfs)
        key_metrics = [
            "mean_latency",
            "energy_per_inference_mj",
            "accuracy",
            "mean_power_w",
        ]

        for metric in key_metrics:
            if metric in stats.index:
                mean = stats.loc[metric, "mean"]
                std = stats.loc[metric, "std"]

                # Format based on metric
                if metric == "mean_latency":
                    mean *= 1000
                    std *= 1000
                    unit = "ms"
                elif metric == "accuracy":
                    mean *= 100
                    std *= 100
                    unit = "%"
                elif metric == "mean_power_w":
                    unit = "W"
                else:
                    unit = "mJ"

                print(f"  {metric:30s}: {mean:8.2f} ± {std:6.2f} {unit}")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

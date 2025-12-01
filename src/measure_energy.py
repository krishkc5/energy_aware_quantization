#!/usr/bin/env python3
"""
Main orchestrator for energy-aware quantization experiments.

This script coordinates the complete measurement pipeline:
1. Load pre-tokenized dataset
2. Load model in specified precision
3. Warmup GPU
4. Start power logging
5. Run timed inference
6. Stop power logging
7. Compute energy metrics
8. Save results

Usage:
    python src/measure_energy.py --precision fp32 --dataset datasets/tokenized_data --num_iters 1000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import load_model, get_model_info, validate_model
from src.dataset_loader import load_pre_tokenized, validate_dataset
from src.warmup import warmup, check_gpu_ready, reset_peak_memory_stats, get_memory_stats
from src.power_logger import PowerLogger, validate_power_samples
from src.inference_runner import run_steady_state_benchmark, validate_timing_results
from src.energy_utils import (
    compute_energy_with_timing,
    compute_energy_efficiency,
    format_energy_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Measure energy consumption for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FP32 benchmark
  python src/measure_energy.py --precision fp32 --dataset datasets/tokenized_data --num_iters 1000

  # Run FP16 benchmark with more warmup
  python src/measure_energy.py --precision fp16 --dataset datasets/tokenized_data --warmup_steps 200

  # Run INT8 benchmark with custom output
  python src/measure_energy.py --precision int8 --dataset datasets/tokenized_data --output results/int8/trial_1.csv

  # Run multiple trials
  for i in {1..5}; do
    python src/measure_energy.py --precision fp32 --dataset datasets/tokenized_data --trial $i
  done
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HuggingFace model name or path (default: distilbert-base-uncased-finetuned-sst-2-english)"
    )

    parser.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["fp32", "fp16", "int8"],
        help="Model precision mode"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to pre-tokenized dataset directory"
    )

    parser.add_argument(
        "--num_iters",
        type=int,
        default=1000,
        help="Number of inference iterations (default: 1000)"
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100)"
    )

    parser.add_argument(
        "--power_sample_interval",
        type=int,
        default=100,
        help="Power sampling interval in milliseconds (default: 100)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: results/<precision>/results_<timestamp>.csv)"
    )

    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Trial number (for multiple runs, default: 1)"
    )

    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Skip warmup phase (not recommended)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def setup_output_path(args) -> Path:
    """
    Setup output path for results.

    Args:
        args: Parsed command line arguments

    Returns:
        Path object for output file
    """
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: results/<precision>/trial_<N>_<timestamp>.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / args.precision
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"trial_{args.trial}_{timestamp}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results(results: Dict, output_path: Path, args) -> None:
    """
    Save results to CSV and JSON.

    Args:
        results: Dictionary with all metrics
        output_path: Path to save CSV file
        args: Command line arguments
    """
    # Add metadata
    results["model_name"] = args.model
    results["precision"] = args.precision
    results["dataset_path"] = args.dataset
    results["num_iters"] = args.num_iters
    results["warmup_steps"] = args.warmup_steps
    results["trial_number"] = args.trial
    results["timestamp"] = datetime.now().isoformat()

    # Save as CSV (one row)
    df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    print(f"\n Results saved to: {output_path}")

    # Also save as JSON for easier reading
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Results saved to: {json_path}")


def print_summary(results: Dict, args) -> None:
    """
    Print comprehensive results summary.

    Args:
        results: Dictionary with all metrics
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Iterations: {args.num_iters}")
    print(f"  Trial: {args.trial}")

    print(f"\nPerformance Metrics:")
    print(f"  Mean Latency: {results['mean_latency']*1000:.2f} ms")
    print(f"  Std Latency: {results['std_latency']*1000:.2f} ms")
    print(f"  Throughput: {results['throughput']:.2f} samples/s")

    print(f"\nPower Metrics:")
    print(f"  Mean Power: {results['mean_power_w']:.2f} W")
    print(f"  Std Power: {results['std_power_w']:.2f} W")
    print(f"  Min Power: {results['min_power_w']:.2f} W")
    print(f"  Max Power: {results['max_power_w']:.2f} W")

    print(f"\nEnergy Metrics:")
    print(f"  Total Energy: {results['total_energy_j']:.3f} J")
    print(f"  Energy/Inference: {results['energy_per_inference_j']:.6f} J")
    print(f"  Energy/Inference: {results['energy_per_inference_mj']:.3f} mJ")
    print(f"  Inferences/Joule: {results['inferences_per_joule']:.2f}")

    if "accuracy" in results:
        print(f"\nAccuracy Metrics:")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Correct: {results['num_correct']}/{results['num_samples_accuracy']}")

    if "peak_allocated_gb" in results:
        print(f"\nMemory Usage:")
        print(f"  Peak Allocated: {results['peak_allocated_gb']:.2f} GB")
        print(f"  Peak Reserved: {results['peak_reserved_gb']:.2f} GB")

    print("="*70)


def main():
    """Main orchestrator function."""
    args = parse_args()

    print("="*70)
    print("ENERGY-AWARE QUANTIZATION EXPERIMENT")
    print("="*70)
    print(f"Precision: {args.precision}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Trial: {args.trial}")
    print("="*70)

    # Setup output path
    output_path = setup_output_path(args)
    print(f"\nOutput will be saved to: {output_path}")

    # Step 1: Check GPU readiness
    print("\n" + "="*70)
    print("STEP 1: Checking GPU")
    print("="*70)
    check_gpu_ready(verbose=True)

    # Step 2: Load dataset
    print("\n" + "="*70)
    print("STEP 2: Loading Dataset")
    print("="*70)
    input_ids, attention_mask, labels, metadata = load_pre_tokenized(
        args.dataset,
        device=args.device
    )
    validate_dataset(input_ids, attention_mask, labels, expected_num_labels=2)

    # Step 3: Load model
    print("\n" + "="*70)
    print("STEP 3: Loading Model")
    print("="*70)
    model = load_model(
        model_name=args.model,
        precision=args.precision,
        device=args.device,
        num_labels=2,
        verbose=True
    )

    # Validate model
    validate_model(model, input_ids[:1], attention_mask[:1], expected_device=args.device)
    model_info = get_model_info(model)

    # Step 4: Warmup
    if not args.no_warmup:
        print("\n" + "="*70)
        print("STEP 4: Warming Up GPU")
        print("="*70)
        reset_peak_memory_stats()
        warmup(
            model,
            input_ids,
            attention_mask,
            num_steps=args.warmup_steps,
            verbose=True
        )
    else:
        print("\n  Skipping warmup (not recommended)")

    # Step 5: Setup power logger
    print("\n" + "="*70)
    print("STEP 5: Setting Up Power Logger")
    print("="*70)
    power_logger = PowerLogger(
        sample_interval_ms=args.power_sample_interval,
        gpu_id=args.gpu_id,
        verbose=args.verbose
    )

    # Step 6: Run benchmark with power logging
    print("\n" + "="*70)
    print("STEP 6: Running Benchmark")
    print("="*70)

    # Reset memory stats before measurement
    reset_peak_memory_stats()

    # Start power logging
    power_logger.start()
    time.sleep(0.5)  # Let power logger stabilize

    # Run benchmark
    benchmark_results = run_steady_state_benchmark(
        model,
        input_ids,
        attention_mask,
        labels,
        num_iters=args.num_iters,
        compute_accuracy=True,
        verbose=True
    )

    # Stop power logging
    time.sleep(0.5)  # Capture trailing power samples
    power_logger.stop()

    # Get power samples
    power_samples = power_logger.read()
    print(f"\n Collected {len(power_samples)} power samples")

    # Validate power samples
    try:
        validate_power_samples(power_samples, min_samples=10)
        print(" Power samples validated")
    except ValueError as e:
        print(f"  Power validation warning: {e}")

    # Step 7: Compute energy metrics
    print("\n" + "="*70)
    print("STEP 7: Computing Energy Metrics")
    print("="*70)

    energy_results = compute_energy_with_timing(power_samples, benchmark_results)
    print(format_energy_results(energy_results))

    # Compute energy efficiency
    if "accuracy" in benchmark_results:
        efficiency_results = compute_energy_efficiency(
            energy_results["energy_per_inference_j"],
            benchmark_results["accuracy"]
        )
        energy_results.update(efficiency_results)

    # Get memory stats
    memory_stats = get_memory_stats()

    # Step 8: Combine all results
    results = {}
    results.update(benchmark_results)
    results.update(energy_results)
    results.update(memory_stats)
    results.update(model_info)

    # Step 9: Save results
    print("\n" + "="*70)
    print("STEP 8: Saving Results")
    print("="*70)
    save_results(results, output_path, args)

    # Step 10: Print final summary
    print_summary(results, args)

    print("\n Experiment completed successfully!")

    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nL Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Timed inference execution for latency and throughput measurements.

This module provides zero-overhead inference loops with precise timing
for performance benchmarking.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


def run_inference(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    num_iters: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run timed inference loop.

    This is the core measurement function that records end-to-end
    inference latency with CUDA synchronization.

    Args:
        model: PyTorch model in eval mode
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        num_iters: Number of inference iterations
        verbose: Whether to show progress bar

    Returns:
        Dictionary with timing statistics:
        - total_time: Total time for all iterations (seconds)
        - mean_latency: Mean latency per iteration (seconds)
        - std_latency: Standard deviation of latency (seconds)
        - min_latency: Minimum latency (seconds)
        - max_latency: Maximum latency (seconds)
        - median_latency: Median latency (seconds)
        - throughput: Samples per second
        - batch_size: Batch size used
        - num_iters: Number of iterations

    Example:
        >>> results = run_inference(model, input_ids, mask, num_iters=1000)
        >>> print(f"Latency: {results['mean_latency']*1000:.2f} ms")
    """
    model.eval()

    if input_ids.device.type == "cpu":
        raise ValueError("input_ids must be on CUDA device")

    if attention_mask.device.type == "cpu":
        raise ValueError("attention_mask must be on CUDA device")

    batch_size = input_ids.shape[0]
    latencies = []

    # Ensure clean state before measurement
    torch.cuda.synchronize()

    iterator = range(num_iters)
    if verbose:
        iterator = tqdm(iterator, desc="Running inference", unit="iter")

    # Main measurement loop
    with torch.no_grad():
        start_time = time.perf_counter()

        for _ in iterator:
            iter_start = time.perf_counter()

            # Forward pass
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Synchronize to ensure kernel completion
            torch.cuda.synchronize()

            iter_end = time.perf_counter()
            latencies.append(iter_end - iter_start)

        end_time = time.perf_counter()

    total_time = end_time - start_time
    latencies = np.array(latencies)

    results = {
        "total_time": float(total_time),
        "mean_latency": float(np.mean(latencies)),
        "std_latency": float(np.std(latencies)),
        "min_latency": float(np.min(latencies)),
        "max_latency": float(np.max(latencies)),
        "median_latency": float(np.median(latencies)),
        "throughput": float(batch_size * num_iters / total_time),
        "batch_size": batch_size,
        "num_iters": num_iters,
    }

    if verbose:
        print(f"\n Inference completed: {num_iters} iterations")
        print(f"  - Total time: {total_time:.3f} s")
        print(f"  - Mean latency: {results['mean_latency']*1000:.2f} ms")
        print(f"  - Std latency: {results['std_latency']*1000:.2f} ms")
        print(f"  - Min latency: {results['min_latency']*1000:.2f} ms")
        print(f"  - Max latency: {results['max_latency']*1000:.2f} ms")
        print(f"  - Throughput: {results['throughput']:.2f} samples/s")

    return results


def run_inference_with_accuracy(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    num_iters: int = 1,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run inference and compute accuracy.

    This function runs the model and computes accuracy metrics.
    Note: For timing-only measurements, use run_inference() instead.

    Args:
        model: PyTorch model in eval mode
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Ground truth labels [batch_size]
        num_iters: Number of times to repeat (for stability, usually 1)
        verbose: Whether to print results

    Returns:
        Dictionary with accuracy metrics:
        - accuracy: Classification accuracy (0-1)
        - num_correct: Number of correct predictions
        - num_samples: Total number of samples
        - mean_latency: Mean latency per sample (seconds)

    Example:
        >>> results = run_inference_with_accuracy(model, ids, mask, labels)
        >>> print(f"Accuracy: {results['accuracy']*100:.2f}%")
    """
    model.eval()

    if input_ids.device.type == "cpu":
        raise ValueError("input_ids must be on CUDA device")

    batch_size = input_ids.shape[0]
    all_predictions = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for _ in range(num_iters):
            start = time.perf_counter()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append(end - start)
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute accuracy
    correct = (all_predictions == all_labels).sum().item()
    total = len(all_labels)
    accuracy = correct / total

    results = {
        "accuracy": float(accuracy),
        "num_correct": int(correct),
        "num_samples": int(total),
        "mean_latency": float(np.mean(latencies)),
    }

    if verbose:
        print(f"\n Inference with accuracy completed")
        print(f"  - Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        print(f"  - Mean latency: {results['mean_latency']*1000:.2f} ms")

    return results


def run_steady_state_benchmark(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    num_iters: int = 1000,
    compute_accuracy: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete steady-state benchmark with timing and accuracy.

    This is the main benchmarking function that combines:
    1. Timed inference for latency/throughput
    2. Accuracy computation (optional)

    Args:
        model: PyTorch model in eval mode
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Ground truth labels [batch_size]
        num_iters: Number of timed iterations
        compute_accuracy: Whether to compute accuracy
        verbose: Whether to print results

    Returns:
        Dictionary with all metrics combined

    Example:
        >>> results = run_steady_state_benchmark(
        ...     model, ids, mask, labels, num_iters=1000
        ... )
    """
    results = {}

    # Run timed inference
    if verbose:
        print("\n" + "="*60)
        print("RUNNING TIMED INFERENCE")
        print("="*60)

    timing_results = run_inference(
        model, input_ids, attention_mask, num_iters, verbose=verbose
    )
    results.update(timing_results)

    # Compute accuracy separately
    if compute_accuracy:
        if verbose:
            print("\n" + "="*60)
            print("COMPUTING ACCURACY")
            print("="*60)

        accuracy_results = run_inference_with_accuracy(
            model, input_ids, attention_mask, labels, num_iters=1, verbose=verbose
        )

        # Add accuracy metrics (avoid overwriting latency from timing run)
        results["accuracy"] = accuracy_results["accuracy"]
        results["num_correct"] = accuracy_results["num_correct"]
        results["num_samples_accuracy"] = accuracy_results["num_samples"]

    if verbose:
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Latency (mean): {results['mean_latency']*1000:.2f} ms")
        print(f"Throughput: {results['throughput']:.2f} samples/s")
        if compute_accuracy:
            print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print("="*60)

    return results


def validate_timing_results(results: Dict, min_iters: int = 100) -> bool:
    """
    Validate that timing results are reasonable.

    Args:
        results: Results dictionary from run_inference()
        min_iters: Minimum number of iterations expected

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if results["num_iters"] < min_iters:
        raise ValueError(
            f"Insufficient iterations: got {results['num_iters']}, need {min_iters}"
        )

    if results["mean_latency"] <= 0:
        raise ValueError(f"Invalid mean latency: {results['mean_latency']}")

    if results["std_latency"] < 0:
        raise ValueError(f"Invalid std latency: {results['std_latency']}")

    # Check coefficient of variation (CV = std/mean)
    cv = results["std_latency"] / results["mean_latency"]
    if cv > 0.5:
        print(f"Warning: High variance in latency (CV={cv:.2%})")

    return True

"""
Energy computation utilities.

This module provides functions to compute energy consumption from
power measurements and timing data.
"""

from typing import Dict, List

import numpy as np


def compute_energy(
    power_samples: List[float],
    total_time: float,
    num_inferences: int
) -> Dict[str, float]:
    """
    Compute energy consumption from power samples.

    Energy is computed as:
        Total Energy (J) = Mean Power (W) � Total Time (s)
        Per-Inference Energy (J) = Total Energy / Number of Inferences

    Args:
        power_samples: List of power measurements in Watts
        total_time: Total time of inference run in seconds
        num_inferences: Number of inference iterations performed

    Returns:
        Dictionary containing:
        - mean_power_w: Mean power draw in Watts
        - std_power_w: Standard deviation of power in Watts
        - min_power_w: Minimum power draw in Watts
        - max_power_w: Maximum power draw in Watts
        - median_power_w: Median power draw in Watts
        - total_energy_j: Total energy consumed in Joules
        - energy_per_inference_j: Energy per inference in Joules
        - energy_per_inference_mj: Energy per inference in millijoules
        - num_power_samples: Number of power samples collected

    Example:
        >>> energy = compute_energy(power_samples, total_time=10.0, num_inferences=1000)
        >>> print(f"Energy per inference: {energy['energy_per_inference_mj']:.2f} mJ")
    """
    if len(power_samples) == 0:
        raise ValueError("No power samples provided")

    if total_time <= 0:
        raise ValueError(f"Invalid total_time: {total_time}")

    if num_inferences <= 0:
        raise ValueError(f"Invalid num_inferences: {num_inferences}")

    power_array = np.array(power_samples)

    # Compute power statistics
    mean_power = float(np.mean(power_array))
    std_power = float(np.std(power_array))
    min_power = float(np.min(power_array))
    max_power = float(np.max(power_array))
    median_power = float(np.median(power_array))

    # Compute energy
    # Energy (J) = Power (W) � Time (s)
    total_energy = mean_power * total_time

    # Per-inference energy
    energy_per_inference = total_energy / num_inferences
    energy_per_inference_mj = energy_per_inference * 1000  # Convert to millijoules

    return {
        "mean_power_w": mean_power,
        "std_power_w": std_power,
        "min_power_w": min_power,
        "max_power_w": max_power,
        "median_power_w": median_power,
        "total_energy_j": total_energy,
        "energy_per_inference_j": energy_per_inference,
        "energy_per_inference_mj": energy_per_inference_mj,
        "num_power_samples": len(power_samples),
    }


def compute_energy_with_timing(
    power_samples: List[float],
    timing_results: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute energy using timing results from inference runner.

    This is a convenience wrapper around compute_energy() that extracts
    the necessary fields from timing_results.

    Args:
        power_samples: List of power measurements in Watts
        timing_results: Dictionary from run_inference() containing
                       total_time and num_iters

    Returns:
        Dictionary with energy metrics (same as compute_energy)

    Example:
        >>> timing = run_inference(model, input_ids, mask, num_iters=1000)
        >>> power_samples = logger.read()
        >>> energy = compute_energy_with_timing(power_samples, timing)
    """
    total_time = timing_results["total_time"]
    num_inferences = int(timing_results["num_iters"])

    return compute_energy(power_samples, total_time, num_inferences)


def compute_energy_efficiency(
    energy_per_inference_j: float,
    accuracy: float
) -> Dict[str, float]:
    """
    Compute energy efficiency metrics.

    Energy efficiency considers both energy consumption and model quality.

    Args:
        energy_per_inference_j: Energy per inference in Joules
        accuracy: Model accuracy (0-1)

    Returns:
        Dictionary containing:
        - energy_per_inference_j: Energy per inference
        - accuracy: Model accuracy
        - energy_accuracy_product: Energy � (1 - accuracy)
        - inferences_per_joule: Number of inferences per Joule
        - accuracy_per_millijoule: Accuracy points per millijoule

    Example:
        >>> eff = compute_energy_efficiency(energy_per_inference_j=0.05, accuracy=0.92)
        >>> print(f"Inferences per joule: {eff['inferences_per_joule']:.2f}")
    """
    if energy_per_inference_j <= 0:
        raise ValueError(f"Invalid energy: {energy_per_inference_j}")

    if not 0 <= accuracy <= 1:
        raise ValueError(f"Invalid accuracy: {accuracy}, must be in [0, 1]")

    inferences_per_joule = 1.0 / energy_per_inference_j
    energy_per_inference_mj = energy_per_inference_j * 1000

    # Lower is better: energy cost for lost accuracy
    energy_accuracy_product = energy_per_inference_j * (1 - accuracy)

    # Higher is better: accuracy achieved per unit energy
    accuracy_per_millijoule = accuracy / energy_per_inference_mj

    return {
        "energy_per_inference_j": energy_per_inference_j,
        "accuracy": accuracy,
        "energy_accuracy_product": energy_accuracy_product,
        "inferences_per_joule": inferences_per_joule,
        "accuracy_per_millijoule": accuracy_per_millijoule,
    }


def compare_configurations(
    results_list: List[Dict[str, float]],
    config_names: List[str]
) -> Dict:
    """
    Compare energy metrics across different configurations.

    Args:
        results_list: List of result dictionaries with energy metrics
        config_names: List of configuration names (e.g., ["FP32", "FP16", "INT8"])

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> results = [fp32_results, fp16_results, int8_results]
        >>> names = ["FP32", "FP16", "INT8"]
        >>> comparison = compare_configurations(results, names)
    """
    if len(results_list) != len(config_names):
        raise ValueError("Number of results must match number of config names")

    if len(results_list) == 0:
        raise ValueError("No results provided")

    comparison = {
        "configurations": config_names,
        "metrics": {},
    }

    # Extract common metrics
    metric_keys = [
        "energy_per_inference_j",
        "mean_power_w",
        "mean_latency",
        "throughput",
        "accuracy",
    ]

    for key in metric_keys:
        if key in results_list[0]:
            values = [r.get(key, None) for r in results_list]
            comparison["metrics"][key] = {
                "values": values,
                "min": min(v for v in values if v is not None),
                "max": max(v for v in values if v is not None),
                "range": max(v for v in values if v is not None) - min(v for v in values if v is not None),
            }

    # Compute relative improvements
    if len(results_list) >= 2:
        baseline = results_list[0]
        comparison["relative_to_baseline"] = {}

        for result, name in zip(results_list[1:], config_names[1:]):
            relative = {}

            if "energy_per_inference_j" in baseline and "energy_per_inference_j" in result:
                energy_reduction = (
                    (baseline["energy_per_inference_j"] - result["energy_per_inference_j"])
                    / baseline["energy_per_inference_j"]
                )
                relative["energy_reduction_pct"] = energy_reduction * 100

            if "mean_latency" in baseline and "mean_latency" in result:
                speedup = baseline["mean_latency"] / result["mean_latency"]
                relative["speedup"] = speedup

            if "accuracy" in baseline and "accuracy" in result:
                accuracy_delta = result["accuracy"] - baseline["accuracy"]
                relative["accuracy_delta_pct"] = accuracy_delta * 100

            comparison["relative_to_baseline"][name] = relative

    return comparison


def format_energy_results(results: Dict[str, float], indent: int = 2) -> str:
    """
    Format energy results as a readable string.

    Args:
        results: Dictionary with energy metrics
        indent: Number of spaces for indentation

    Returns:
        Formatted string

    Example:
        >>> print(format_energy_results(energy_results))
    """
    lines = []
    prefix = " " * indent

    if "mean_power_w" in results:
        lines.append(f"{prefix}Mean Power: {results['mean_power_w']:.2f} W")
        lines.append(f"{prefix}Power Std: {results['std_power_w']:.2f} W")

    if "total_energy_j" in results:
        lines.append(f"{prefix}Total Energy: {results['total_energy_j']:.3f} J")

    if "energy_per_inference_j" in results:
        lines.append(f"{prefix}Energy/Inference: {results['energy_per_inference_j']:.6f} J")
        lines.append(f"{prefix}Energy/Inference: {results['energy_per_inference_mj']:.3f} mJ")

    if "inferences_per_joule" in results:
        lines.append(f"{prefix}Inferences/Joule: {results['inferences_per_joule']:.2f}")

    if "mean_latency" in results:
        lines.append(f"{prefix}Mean Latency: {results['mean_latency']*1000:.2f} ms")

    if "throughput" in results:
        lines.append(f"{prefix}Throughput: {results['throughput']:.2f} samples/s")

    if "accuracy" in results:
        lines.append(f"{prefix}Accuracy: {results['accuracy']*100:.2f}%")

    return "\n".join(lines)

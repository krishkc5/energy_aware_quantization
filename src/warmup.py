"""
GPU warmup utilities for stabilizing performance measurements.

This module provides warmup functions to ensure GPU clocks are stable,
CUDA kernels are compiled, and memory allocation patterns are established
before actual measurements begin.
"""

import time
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


def warmup(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    num_steps: int = 100,
    verbose: bool = True
) -> float:
    """
    Warmup GPU by running inference without timing.

    This function stabilizes:
    - GPU clock frequencies
    - CUDA kernel compilation (JIT)
    - Memory allocator behavior
    - TensorCore activation (for Ampere/Hopper GPUs)

    Args:
        model: PyTorch model in eval mode
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        num_steps: Number of warmup iterations (default: 100)
        verbose: Whether to show progress bar

    Returns:
        Average warmup iteration time in seconds

    Example:
        >>> model.eval()
        >>> warmup_time = warmup(model, input_ids, mask, num_steps=100)
        >>> print(f"Warmup completed: {warmup_time:.4f}s per iteration")
    """
    model.eval()

    if input_ids.device.type == "cpu":
        raise ValueError("input_ids must be on CUDA device for warmup")

    if attention_mask.device.type == "cpu":
        raise ValueError("attention_mask must be on CUDA device for warmup")

    # Ensure model is on same device as inputs
    device = input_ids.device
    model.to(device)

    times = []

    iterator = range(num_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Warming up GPU", unit="iter")

    with torch.no_grad():
        for _ in iterator:
            start = time.perf_counter()

            # Run forward pass
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Synchronize to ensure kernel completion
            torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed)

    avg_time = sum(times) / len(times)

    if verbose:
        print(f" Warmup completed: {num_steps} iterations")
        print(f"  - Average time: {avg_time*1000:.2f} ms/iter")
        print(f"  - Min time: {min(times)*1000:.2f} ms")
        print(f"  - Max time: {max(times)*1000:.2f} ms")

    return avg_time


def warmup_with_different_sizes(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    num_steps: int = 50,
    verbose: bool = True
) -> None:
    """
    Warmup with varying batch sizes to cover different kernel paths.

    Some CUDA kernels are optimized differently for different batch sizes.
    This ensures all code paths are warmed up.

    Args:
        model: PyTorch model in eval mode
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        num_steps: Number of warmup iterations per size
        verbose: Whether to print progress

    Example:
        >>> warmup_with_different_sizes(model, input_ids, mask, num_steps=50)
    """
    model.eval()
    original_batch_size = input_ids.shape[0]

    # Test with single sample, half batch, and full batch
    batch_sizes = [1, max(1, original_batch_size // 2), original_batch_size]
    batch_sizes = sorted(set(batch_sizes))  # Remove duplicates

    if verbose:
        print(f"Warming up with batch sizes: {batch_sizes}")

    with torch.no_grad():
        for bs in batch_sizes:
            ids_subset = input_ids[:bs]
            mask_subset = attention_mask[:bs]

            iterator = range(num_steps)
            if verbose:
                iterator = tqdm(
                    iterator,
                    desc=f"Warmup batch_size={bs}",
                    unit="iter",
                    leave=False
                )

            for _ in iterator:
                _ = model(input_ids=ids_subset, attention_mask=mask_subset)
                torch.cuda.synchronize()

    if verbose:
        print(f" Multi-size warmup completed")


def check_gpu_ready(verbose: bool = True) -> bool:
    """
    Check if GPU is ready for measurements.

    Verifies:
    - CUDA is available
    - GPU is accessible
    - No pending operations

    Args:
        verbose: Whether to print GPU information

    Returns:
        True if GPU is ready

    Raises:
        RuntimeError: If GPU is not available or not ready
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices found")

    # Get current device
    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    device_props = torch.cuda.get_device_properties(device_id)

    # Synchronize all streams
    torch.cuda.synchronize()

    # Clear cache
    torch.cuda.empty_cache()

    if verbose:
        print(f" GPU ready for measurements")
        print(f"  - Device: {device_name}")
        print(f"  - Compute capability: {device_props.major}.{device_props.minor}")
        print(f"  - Total memory: {device_props.total_memory / 1e9:.2f} GB")
        print(f"  - Multi-processors: {device_props.multi_processor_count}")

        # Check memory usage
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  - Memory allocated: {allocated:.2f} GB")
        print(f"  - Memory reserved: {reserved:.2f} GB")

    return True


def reset_peak_memory_stats() -> None:
    """
    Reset peak memory statistics.

    Call this before measurements to get accurate peak memory usage.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def get_memory_stats() -> dict:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory statistics in GB
    """
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }

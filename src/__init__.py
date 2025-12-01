"""
Energy-aware quantization measurement harness.

This package provides tools for measuring accuracy, latency, and energy
consumption of quantized LLM inference.
"""

from .dataset_loader import load_pre_tokenized, validate_dataset
from .warmup import warmup, check_gpu_ready, get_memory_stats
from .power_logger import PowerLogger
from .inference_runner import run_inference, run_steady_state_benchmark
from .energy_utils import compute_energy, compute_energy_efficiency

__version__ = "0.1.0"

__all__ = [
    "load_pre_tokenized",
    "validate_dataset",
    "warmup",
    "check_gpu_ready",
    "get_memory_stats",
    "PowerLogger",
    "run_inference",
    "run_steady_state_benchmark",
    "compute_energy",
    "compute_energy_efficiency",
]

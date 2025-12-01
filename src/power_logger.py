"""
GPU power monitoring using nvidia-smi.

This module provides asynchronous power logging to measure GPU power draw
during inference without interfering with the measurement loop.
"""

import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


class PowerLogger:
    """
    Asynchronous GPU power logger using nvidia-smi.

    This class runs nvidia-smi in a background subprocess to continuously
    monitor GPU power draw without blocking the main thread.

    Example:
        >>> logger = PowerLogger(sample_interval_ms=100)
        >>> logger.start()
        >>> # Run your workload here
        >>> time.sleep(5)
        >>> logger.stop()
        >>> samples = logger.read()
        >>> print(f"Mean power: {np.mean(samples):.2f} W")
    """

    def __init__(
        self,
        sample_interval_ms: int = 100,
        gpu_id: int = 0,
        verbose: bool = False
    ):
        """
        Initialize power logger.

        Args:
            sample_interval_ms: Sampling interval in milliseconds (default: 100)
            gpu_id: GPU device ID to monitor (default: 0)
            verbose: Whether to print debug information
        """
        self.sample_interval_ms = sample_interval_ms
        self.gpu_id = gpu_id
        self.verbose = verbose

        self.process: Optional[subprocess.Popen] = None
        self.samples: List[float] = []
        self.is_running = False
        self._lock = threading.Lock()

        # Check if nvidia-smi is available
        self._check_nvidia_smi()

    def _check_nvidia_smi(self) -> None:
        """
        Check if nvidia-smi is available and GPU is accessible.

        Raises:
            RuntimeError: If nvidia-smi is not available or GPU is not accessible
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

            # Try to parse the output
            power = float(result.stdout.strip())

            if self.verbose:
                print(f" nvidia-smi available, current power: {power:.2f} W")

        except FileNotFoundError:
            raise RuntimeError("nvidia-smi not found. Is NVIDIA driver installed?")
        except ValueError as e:
            raise RuntimeError(f"Failed to parse nvidia-smi output: {e}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("nvidia-smi timed out")

    def start(self) -> None:
        """
        Start power logging in background.

        Uses manual polling approach to avoid buffering issues.

        Raises:
            RuntimeError: If logger is already running
        """
        with self._lock:
            if self.is_running:
                raise RuntimeError("PowerLogger is already running")

            self.samples = []
            self.is_running = True

        # We don't start a continuous nvidia-smi process anymore
        # Instead, the reader thread will poll nvidia-smi directly
        self.process = None  # Not used with manual polling

        if self.verbose:
            print(f"Starting power logger with manual polling (interval: {self.sample_interval_ms} ms)")

        # Start reader thread (will poll nvidia-smi)
        self._reader_thread = threading.Thread(target=self._read_samples, daemon=True)
        self._reader_thread.start()

        if self.verbose:
            print(f" Power logger started (interval: {self.sample_interval_ms} ms)")

    def _read_samples(self) -> None:
        """
        Background thread that polls nvidia-smi periodically.

        This runs continuously until stop() is called.
        Uses manual polling to avoid buffering issues with continuous mode.
        """
        poll_interval = self.sample_interval_ms / 1000.0  # Convert to seconds

        try:
            while self.is_running:
                try:
                    # Poll nvidia-smi directly (no continuous mode)
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits",
                            f"--id={self.gpu_id}"
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )

                    if result.returncode == 0:
                        line = result.stdout.strip()
                        try:
                            power = float(line)
                            with self._lock:
                                self.samples.append(power)

                            if self.verbose and len(self.samples) % 100 == 0:
                                print(f"  Power samples collected: {len(self.samples)}")
                        except ValueError:
                            if self.verbose:
                                print(f"  Warning: Could not parse power value: {line}")

                except subprocess.TimeoutExpired:
                    if self.verbose:
                        print("  Warning: nvidia-smi polling timeout")
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: nvidia-smi poll error: {e}")

                # Sleep until next sample
                time.sleep(poll_interval)

        except Exception as e:
            if self.verbose:
                print(f"  Error in power reader thread: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()

    def stop(self) -> None:
        """
        Stop power logging and wait for reader thread to finish.

        Raises:
            RuntimeError: If logger is not running
        """
        with self._lock:
            if not self.is_running:
                raise RuntimeError("PowerLogger is not running")

            self.is_running = False

        # Wait for reader thread to finish (no subprocess to terminate with polling)
        if hasattr(self, '_reader_thread'):
            self._reader_thread.join(timeout=2)

        if self.verbose:
            print(f" Power logger stopped ({len(self.samples)} samples collected)")

    def read(self) -> List[float]:
        """
        Get all collected power samples.

        Returns:
            List of power samples in Watts

        Note:
            Can be called while logger is running or after it's stopped.
        """
        with self._lock:
            return self.samples.copy()

    def get_statistics(self) -> dict:
        """
        Get statistics on collected power samples.

        Returns:
            Dictionary with mean, std, min, max, median power in Watts

        Raises:
            ValueError: If no samples have been collected
        """
        samples = self.read()

        if len(samples) == 0:
            raise ValueError("No power samples collected")

        return {
            "mean_power_w": float(np.mean(samples)),
            "std_power_w": float(np.std(samples)),
            "min_power_w": float(np.min(samples)),
            "max_power_w": float(np.max(samples)),
            "median_power_w": float(np.median(samples)),
            "num_samples": len(samples),
            "sample_interval_ms": self.sample_interval_ms,
        }

    def clear(self) -> None:
        """
        Clear all collected samples.

        Useful for running multiple measurement trials.
        """
        with self._lock:
            self.samples = []

        if self.verbose:
            print(" Power samples cleared")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_running:
            self.stop()
        return False


def estimate_power_from_single_sample(
    gpu_id: int = 0,
    num_samples: int = 10,
    interval_ms: int = 100
) -> List[float]:
    """
    Estimate power by taking discrete samples (fallback for continuous monitoring).

    This is a fallback when continuous nvidia-smi monitoring fails (e.g., on Kaggle).
    Takes multiple discrete samples over a short period.

    Args:
        gpu_id: GPU device ID
        num_samples: Number of samples to take
        interval_ms: Interval between samples in milliseconds

    Returns:
        List of power samples in Watts
    """
    import subprocess
    import time

    samples = []
    interval_sec = interval_ms / 1000.0

    print(f"  Taking {num_samples} discrete power samples (fallback mode)...")

    for i in range(num_samples):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                    f"--id={gpu_id}"
                ],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                power = float(result.stdout.strip())
                samples.append(power)

                if (i + 1) % 3 == 0:
                    print(f"    Sampled {i+1}/{num_samples}: {power:.2f} W")

            time.sleep(interval_sec)

        except Exception as e:
            print(f"  Warning: Sample {i+1} failed: {e}")
            continue

    if len(samples) > 0:
        print(f"  ✓ Collected {len(samples)} power samples")
        print(f"    Mean power: {np.mean(samples):.2f} W")
    else:
        print(f"  ⚠️  Failed to collect any power samples")

    return samples


def validate_power_samples(
    samples: List[float],
    min_samples: int = 10,
    min_power: float = 0.0,
    max_power: float = 1000.0
) -> bool:
    """
    Validate that power samples are reasonable.

    Args:
        samples: List of power measurements in Watts
        min_samples: Minimum number of samples required
        min_power: Minimum expected power draw in Watts
        max_power: Maximum expected power draw in Watts

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if len(samples) < min_samples:
        raise ValueError(
            f"Insufficient power samples: got {len(samples)}, need {min_samples}"
        )

    if any(p < min_power or p > max_power for p in samples):
        raise ValueError(
            f"Power samples out of range [{min_power}, {max_power}]: "
            f"min={min(samples):.2f}, max={max(samples):.2f}"
        )

    # Check for variance (if all samples are identical, something is wrong)
    if len(set(samples)) == 1:
        raise ValueError("All power samples are identical, logger may not be working")

    return True

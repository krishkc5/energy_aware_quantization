"""
Dataset loader for pre-tokenized SST-2 data.

This module provides zero-I/O loading of pre-tokenized datasets
that have been saved as .pt files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor


def load_pre_tokenized(
    path: str,
    device: str = "cuda"
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """
    Load pre-tokenized dataset from .pt files.

    This function loads tokenized data that was prepared offline,
    ensuring zero I/O overhead during inference measurements.

    Args:
        path: Directory containing input_ids.pt, attention_mask.pt,
              labels.pt, and metadata.json
        device: Device to load tensors to (default: "cuda")

    Returns:
        Tuple of (input_ids, attention_mask, labels, metadata)
        where all tensors are already on the specified device.

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If device is invalid

    Example:
        >>> input_ids, mask, labels, meta = load_pre_tokenized("datasets/tokenized_data")
        >>> print(f"Loaded {meta['num_samples']} samples")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")

    # Load metadata
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load tensors
    required_files = ["input_ids.pt", "attention_mask.pt", "labels.pt"]
    tensors = {}

    for filename in required_files:
        file_path = path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {filename} in {path}")

        # Load tensor directly to target device to avoid CPU-GPU transfer later
        tensor = torch.load(file_path, map_location=device)
        tensors[filename.replace(".pt", "")] = tensor

    input_ids = tensors["input_ids"]
    attention_mask = tensors["attention_mask"]
    labels = tensors["labels"]

    # Validate shapes
    if input_ids.shape[0] != attention_mask.shape[0]:
        raise ValueError(
            f"Shape mismatch: input_ids has {input_ids.shape[0]} samples "
            f"but attention_mask has {attention_mask.shape[0]}"
        )

    if input_ids.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Shape mismatch: input_ids has {input_ids.shape[0]} samples "
            f"but labels has {labels.shape[0]}"
        )

    print(f" Loaded {input_ids.shape[0]} samples from {path}")
    print(f"  - Input shape: {input_ids.shape}")
    print(f"  - Mask shape: {attention_mask.shape}")
    print(f"  - Labels shape: {labels.shape}")
    print(f"  - Device: {input_ids.device}")
    print(f"  - Max sequence length: {metadata.get('max_length', 'N/A')}")

    return input_ids, attention_mask, labels, metadata


def validate_dataset(
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    expected_num_labels: int = 2
) -> bool:
    """
    Validate that dataset tensors are properly formatted.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Classification labels [batch_size]
        expected_num_labels: Expected number of unique labels

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    # Check dimensions
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 2D, got {input_ids.ndim}D")

    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be 2D, got {attention_mask.ndim}D")

    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.ndim}D")

    # Check shapes match
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"input_ids shape {input_ids.shape} != "
            f"attention_mask shape {attention_mask.shape}"
        )

    if input_ids.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Batch size mismatch: input_ids has {input_ids.shape[0]} "
            f"but labels has {labels.shape[0]}"
        )

    # Check label range
    unique_labels = labels.unique()
    if len(unique_labels) > expected_num_labels:
        raise ValueError(
            f"Found {len(unique_labels)} unique labels, "
            f"expected at most {expected_num_labels}"
        )

    if labels.min() < 0 or labels.max() >= expected_num_labels:
        raise ValueError(
            f"Labels out of range [0, {expected_num_labels}): "
            f"min={labels.min()}, max={labels.max()}"
        )

    print(f" Dataset validation passed")
    print(f"  - Batch size: {input_ids.shape[0]}")
    print(f"  - Sequence length: {input_ids.shape[1]}")
    print(f"  - Unique labels: {unique_labels.tolist()}")

    return True

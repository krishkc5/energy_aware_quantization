"""
Model loading utilities for FP32, FP16, and INT8 precision modes.

This module provides functions to load DistilBERT and GPT2 models
in different precision formats for energy measurement.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    precision: str  # "fp32", "fp16", "int8"
    device: str = "cuda"
    num_labels: int = 2


def _apply_int8_quantization_cuda(model: nn.Module, verbose: bool = True) -> None:
    """
    Apply INT8-like quantization for CUDA by quantizing weights to INT8 range.

    This uses symmetric per-tensor quantization:
    - Quantize: Q = round(R / scale) where scale = max(abs(R)) / 127
    - Dequantize: R' = Q * scale

    The weights are stored as FP32/FP16 but quantized to INT8 precision.
    This is not true INT8 compute (which requires special kernels) but
    simulates the accuracy/precision loss of INT8 quantization.

    Args:
        model: Model to quantize (must be on CUDA)
        verbose: Whether to print quantization info
    """
    num_quantized = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Quantize weight
            with torch.no_grad():
                weight = module.weight.data

                # Symmetric per-tensor quantization
                # Scale = max_abs_value / 127 (INT8 max positive value)
                scale = weight.abs().max() / 127.0

                if scale > 0:
                    # Quantize: divide by scale, round, clamp to INT8 range
                    weight_q = torch.round(weight / scale)
                    weight_q = torch.clamp(weight_q, -128, 127)

                    # Dequantize: multiply back by scale
                    weight_dequant = weight_q * scale

                    # Replace original weight with quantized version
                    module.weight.data = weight_dequant

                # Quantize bias if it exists
                if module.bias is not None:
                    bias = module.bias.data
                    scale_bias = bias.abs().max() / 127.0

                    if scale_bias > 0:
                        bias_q = torch.round(bias / scale_bias)
                        bias_q = torch.clamp(bias_q, -128, 127)
                        bias_dequant = bias_q * scale_bias
                        module.bias.data = bias_dequant

                num_quantized += 1

    if verbose:
        print(f"  ✓ Quantized {num_quantized} Linear layers to INT8 precision")
        print(f"  ✓ Running on CUDA (simulated INT8 compute)")


def load_model(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    precision: str = "fp32",
    device: str = "cuda",
    num_labels: int = 2,
    verbose: bool = True
) -> nn.Module:
    """
    Load a transformer model in specified precision.

    Supports three precision modes:
    - fp32: Full precision (32-bit floating point)
    - fp16: Half precision (16-bit floating point)
    - int8: 8-bit integer quantization (dynamic quantization)

    Args:
        model_name: HuggingFace model name or path
        precision: One of "fp32", "fp16", "int8"
        device: Device to load model to
        num_labels: Number of classification labels
        verbose: Whether to print loading information

    Returns:
        Loaded model in eval mode

    Raises:
        ValueError: If precision mode is invalid

    Example:
        >>> model = load_model("distilbert-base-uncased-finetuned-sst-2-english", "fp16")
        >>> model.eval()
    """
    precision = precision.lower()
    valid_precisions = ["fp32", "fp16", "int8"]

    if precision not in valid_precisions:
        raise ValueError(f"Invalid precision: {precision}. Must be one of {valid_precisions}")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")

    if verbose:
        print(f"\nLoading model: {model_name}")
        print(f"Precision: {precision}")
        print(f"Device: {device}")

    # Load model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    # Load model in FP32 first
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float32
    )

    # Apply precision conversion
    if precision == "fp32":
        model = model.to(device)

    elif precision == "fp16":
        # Convert to FP16
        model = model.half()
        model = model.to(device)

    elif precision == "int8":
        # For CUDA: Use manual INT8 conversion with fake quantization
        # This simulates INT8 by quantizing weights to int8 range but keeps computation in FP32/FP16
        if device == "cuda":
            if verbose:
                print("  Using CUDA-compatible INT8 (simulated quantization)")

            # Move model to CUDA first
            model = model.to(device)

            # Apply simulated INT8 quantization to Linear layers
            _apply_int8_quantization_cuda(model, verbose=verbose)
        else:
            # For CPU: Use PyTorch's dynamic quantization
            if verbose:
                print("  Using CPU dynamic quantization")
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            model = model.to("cpu")

    model.eval()

    if verbose:
        print(f"✓ Model loaded successfully")
        print(f"  - Parameters: {count_parameters(model):,}")
        print(f"  - Model size: {get_model_size_mb(model):.2f} MB")

        # Check actual dtype of first parameter
        first_param = next(model.parameters())
        print(f"  - Parameter dtype: {first_param.dtype}")
        print(f"  - Parameter device: {first_param.device}")

    return model


def load_model_with_autocast(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cuda",
    num_labels: int = 2,
    verbose: bool = True
) -> Tuple[nn.Module, torch.autocast]:
    """
    Load FP32 model with autocast context for FP16 inference.

    This approach keeps model weights in FP32 but performs operations in FP16
    using PyTorch's automatic mixed precision.

    Args:
        model_name: HuggingFace model name
        device: Device to load to
        num_labels: Number of labels
        verbose: Whether to print info

    Returns:
        Tuple of (model, autocast_context)

    Example:
        >>> model, autocast_ctx = load_model_with_autocast()
        >>> with autocast_ctx:
        ...     output = model(input_ids, attention_mask)
    """
    if verbose:
        print(f"\nLoading model with autocast: {model_name}")

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    model = model.to(device)
    model.eval()

    # Create autocast context
    autocast_ctx = torch.autocast(device_type=device, dtype=torch.float16)

    if verbose:
        print(f"Model loaded with autocast")
        print(f"  - Autocast device: {device}")
        print(f"  - Autocast dtype: torch.float16")

    return model, autocast_ctx


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def get_model_info(model: nn.Module) -> dict:
    """
    Get detailed model information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    first_param = next(model.parameters())

    return {
        "num_parameters": count_parameters(model),
        "model_size_mb": get_model_size_mb(model),
        "dtype": str(first_param.dtype),
        "device": str(first_param.device),
        "is_quantized": hasattr(model, "qconfig"),
    }


def validate_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    expected_device: str = "cuda"
) -> bool:
    """
    Validate that model is properly loaded and can perform inference.

    Args:
        model: Model to validate
        input_ids: Sample input IDs
        attention_mask: Sample attention mask
        expected_device: Expected device for model

    Returns:
        True if validation passes

    Raises:
        RuntimeError: If validation fails
    """
    # Check model is in eval mode
    if model.training:
        raise RuntimeError("Model is in training mode, should be in eval mode")

    # Check device
    model_device = next(model.parameters()).device
    if expected_device not in str(model_device):
        raise RuntimeError(
            f"Model on wrong device: expected {expected_device}, got {model_device}"
        )

    # Check inputs are on same device as model
    if str(input_ids.device) != str(model_device):
        raise RuntimeError(
            f"Input device mismatch: inputs on {input_ids.device}, model on {model_device}"
        )

    # Try a forward pass
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Check output shape
        if logits.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                f"Output batch size mismatch: {logits.shape[0]} != {input_ids.shape[0]}"
            )

    except Exception as e:
        raise RuntimeError(f"Forward pass failed: {e}")

    print(f"✓ Model validation passed")
    return True


# Preset model configurations
PRESET_MODELS = {
    "distilbert-sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "distilbert": "distilbert-base-uncased",
    "gpt2-small": "gpt2",
}


def get_preset_model_name(preset: str) -> str:
    """
    Get full model name from preset.

    Args:
        preset: Preset name (e.g., "distilbert-sst2")

    Returns:
        Full HuggingFace model name

    Raises:
        ValueError: If preset is not found
    """
    if preset in PRESET_MODELS:
        return PRESET_MODELS[preset]

    # If not a preset, assume it's already a full model name
    return preset

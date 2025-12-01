#!/usr/bin/env python3
"""
Test script to verify installation and dataset availability.

This script checks:
1. Python dependencies are installed
2. CUDA is available
3. Pre-tokenized datasets exist
4. All modules can be imported
"""

import sys
from pathlib import Path

print("="*70)
print("TESTING INSTALLATION")
print("="*70)

# Test 1: Check Python version
print("\n[1/7] Checking Python version...")
print(f"  Python version: {sys.version}")
if sys.version_info < (3, 8):
    print("  ❌ Python 3.8+ required")
    sys.exit(1)
print("  ✓ Python version OK")

# Test 2: Check dependencies
print("\n[2/7] Checking dependencies...")
required_packages = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "tqdm",
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ❌ {package} not found")
        missing_packages.append(package)

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check CUDA
print("\n[3/7] Checking CUDA availability...")
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    print("  ⚠️  CUDA not available (CPU-only mode)")
    print("  Note: Energy measurements require CUDA GPU")

# Test 4: Check datasets
print("\n[4/7] Checking datasets...")
dataset_dir = Path("datasets")
if not dataset_dir.exists():
    print("  ❌ datasets/ directory not found")
    sys.exit(1)

dataset_variants = ["tokenized_data", "tokenized_data_small", "tokenized_data_large", "tokenized_data_standard"]
found_datasets = []

for variant in dataset_variants:
    variant_dir = dataset_dir / variant
    if variant_dir.exists():
        required_files = ["input_ids.pt", "attention_mask.pt", "labels.pt", "metadata.json"]
        all_exist = all((variant_dir / f).exists() for f in required_files)
        if all_exist:
            print(f"  ✓ {variant}")
            found_datasets.append(variant)
        else:
            print(f"  ⚠️  {variant} (incomplete)")

if not found_datasets:
    print("  ❌ No complete datasets found")
    sys.exit(1)

# Test 5: Test imports
print("\n[5/7] Testing module imports...")
try:
    from src import (
        load_pre_tokenized,
        warmup,
        PowerLogger,
        run_inference,
        compute_energy
    )
    print("  ✓ src module imports OK")
except ImportError as e:
    print(f"  ❌ Failed to import src modules: {e}")
    sys.exit(1)

try:
    from models import load_model
    print("  ✓ models module imports OK")
except ImportError as e:
    print(f"  ❌ Failed to import models module: {e}")
    sys.exit(1)

# Test 6: Test dataset loading
print("\n[6/7] Testing dataset loading...")
try:
    input_ids, mask, labels, metadata = load_pre_tokenized(
        f"datasets/{found_datasets[0]}",
        device="cpu"  # Use CPU for testing
    )
    print(f"  ✓ Dataset loaded: {input_ids.shape[0]} samples")
except Exception as e:
    print(f"  ❌ Dataset loading failed: {e}")
    sys.exit(1)

# Test 7: Test model loading
print("\n[7/7] Testing model loading...")
try:
    from models import load_model
    print("  Loading DistilBERT (this may take a moment)...")
    model = load_model(
        "distilbert-base-uncased-finetuned-sst-2-english",
        precision="fp32",
        device="cpu",
        verbose=False
    )
    print("  ✓ Model loaded successfully")

    # Test inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids[:1], attention_mask=mask[:1])
        logits = outputs.logits
    print(f"  ✓ Model inference works (output shape: {logits.shape})")

except Exception as e:
    print(f"  ❌ Model loading/inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "="*70)
print("✓ ALL TESTS PASSED")
print("="*70)
print("\nYour installation is ready!")
print("\nTo run experiments:")
print("  python src/measure_energy.py --precision fp32 --dataset datasets/tokenized_data --num_iters 100")
print("\nOr run all experiments:")
print("  ./run_all_experiments.sh")
print("="*70)

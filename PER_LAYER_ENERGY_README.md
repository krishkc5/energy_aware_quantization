# Per-Layer Energy Measurement Guide

## Overview

I've added per-layer energy profiling capabilities to your energy measurement harness. This allows you to measure the energy consumption of each individual layer in the DistilBERT model.

## What Was Added

### 1. **LayerEnergyProfiler Class**
A new class that:
- Registers all layers in DistilBERT (embeddings, 6 transformer blocks, pre-classifier, classifier)
- Measures energy consumption for each layer individually
- Profiles latency and power draw per layer
- Supports batch processing

### 2. **Key Features**

#### Layers Profiled:
1. **Embeddings** - Word + position embeddings
2. **Transformer Block 0-5** - Six transformer layers (attention + FFN)
3. **Pre-classifier** - Dense layer before classification
4. **Classifier** - Final classification head

#### Metrics Per Layer:
- Energy per inference (mJ)
- Average power draw (W)
- Latency per loop (ms)
- Energy percentage of total

### 3. **Visualization**
Automatic generation of:
- Per-layer energy bar chart
- Per-layer latency bar chart
- Side-by-side comparison

## How to Use

### Step 1: Run the PowerLogger Test
```python
# Test PowerLogger first (already in notebook)
logger = PowerLogger(gpu_id=0, poll_interval_ms=100)
logger.start()
time.sleep(5)
samples = logger.stop()
```

### Step 2: Run Per-Layer Profiling
The notebook now includes a dedicated cell for per-layer profiling. Simply run:

```python
# This cell is already in the notebook
# It will:
# 1. Load the FP32 model
# 2. Create the profiler
# 3. Warm up the GPU
# 4. Profile all 9 layers
# 5. Generate visualizations
# 6. Save results
```

### Step 3: Analyze Results
The profiler will output:
- CSV file: `fp32_per_layer_energy.csv`
- Visualization: `fp32_per_layer_energy.png`
- Console summary with energy breakdown

## Expected Output

```
PER-LAYER ENERGY PROFILING
======================================================================
Configuration:
  Batch size: 16
  Loops per layer: 100
  Total layers: 9

  Measuring layer: embeddings
    ✓ embeddings                 : 2.345 mJ, 0.234 ms, 45.2 W

  Measuring layer: transformer_block_0
    ✓ transformer_block_0        : 8.123 mJ, 0.812 ms, 48.5 W

  ... (continues for all layers)

ENERGY BREAKDOWN BY LAYER
======================================================================
  embeddings               :  2.345 mJ ( 5.23%)
  transformer_block_0      :  8.123 mJ (18.12%)
  transformer_block_1      :  8.034 mJ (17.92%)
  transformer_block_2      :  7.989 mJ (17.82%)
  transformer_block_3      :  7.876 mJ (17.56%)
  transformer_block_4      :  7.923 mJ (17.67%)
  transformer_block_5      :  7.812 mJ (17.42%)
  pre_classifier           :  1.234 mJ ( 2.75%)
  classifier               :  0.876 mJ ( 1.95%)

  TOTAL                    : 44.812 mJ (100.00%)
======================================================================
```

## Use Cases

### 1. **Identify Energy Hotspots**
Find which layers consume the most energy:
```python
# Sort by energy consumption
df_layers.sort_values('energy_per_inference_mj', ascending=False)
```

### 2. **Target Quantization**
Use results to decide which layers to quantize:
- High-energy layers → Prioritize for quantization
- Low-energy layers → May not benefit as much

### 3. **Compare Precision Levels**
Run the same profiling for FP16 and INT8 (when available):
```python
# For FP16
fp16_model = load_model("fp16", config.model_name, config.device)
profiler_fp16 = LayerEnergyProfiler(fp16_model, device=config.device)
layer_results_fp16 = profiler_fp16.profile_all_layers(...)
```

### 4. **Energy-Aware Architecture Design**
Understand which components (attention vs FFN) dominate energy:
- Transformer blocks typically consume 80-90% of total energy
- Embeddings: 5-10%
- Classification head: 2-5%

## Important Notes

### ⚠️ Measurement Accuracy
- Per-layer measurements include overhead from previous layers
- The profiler runs each layer in sequence, not in isolation
- Results show the **incremental energy** of running through each layer
- For most accurate results, use `num_loops >= 100`

### ⚠️ Dataset Requirements
The current configuration uses 100 samples and will wraparound. For better results:
```bash
# Generate more samples
python create_large_dataset.py
```
This creates 500-872 unique samples to reduce data reuse.

### ⚠️ GPU Stabilization
- Each layer measurement includes a 0.3s stabilization period
- Total profiling time: ~9 layers × (100 loops + 0.3s) ≈ 3-5 minutes
- Warmup is crucial for stable results

## Configuration Options

### Adjust Number of Loops
```python
# More loops = more accurate, but slower
layer_results = profiler.profile_all_layers(
    input_ids,
    attention_mask,
    batch_size=16,
    num_loops=200  # Increase for better accuracy
)
```

### Change Batch Size
```python
# Larger batch = better GPU utilization
layer_results = profiler.profile_all_layers(
    input_ids,
    attention_mask,
    batch_size=32,  # Increase if you have enough samples
    num_loops=100
)
```

## Troubleshooting

### Issue: "No power samples collected"
**Solution:** PowerLogger not working. Test nvidia-smi manually:
```bash
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -l 1
```

### Issue: "FileNotFoundError: Dataset not found"
**Solution:** Update the dataset path in the config:
```python
dataset_path: str = r"c:\Users\taara\UPENN JR FALL\ESE 5390\energy_aware_quantization\datasets\tokenized_data_large"
```

### Issue: Out of memory
**Solution:** Reduce batch size:
```python
batch_size=8  # Reduce from 16 to 8
```

## Next Steps

1. **Run the profiler** on your FP32 baseline
2. **Analyze the results** to identify energy hotspots
3. **Compare with FP16/INT8** when quantized models are ready
4. **Use insights** to guide quantization strategy

## Files Modified

- `notebooks/energy_harness_kaggle (1).ipynb`
  - Added `LayerEnergyProfiler` class
  - Added `visualize_layer_energy()` function
  - Added per-layer profiling experiment cell

## Example Results Structure

```python
layer_results = [
    {
        'layer_name': 'embeddings',
        'layer_idx': 0,
        'avg_power_w': 45.2,
        'total_time_s': 2.34,
        'energy_total_j': 105.8,
        'energy_per_inference_mj': 2.345,
        'num_loops': 100,
        'latency_per_loop_ms': 0.234,
        'num_power_samples': 23
    },
    # ... more layers
]
```

## Questions?

If you encounter issues or need to adjust the profiling methodology, check:
1. PowerLogger is working (run the test cell first)
2. Dataset path is correct
3. GPU has enough memory for the batch size
4. nvidia-smi is accessible

Good luck with your energy profiling! 🚀

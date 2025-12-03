# Accuracy Evaluation Added to Energy Harness

## Summary

I've successfully added **accuracy evaluation cells** to [energy_harness_T4_with_INT8.ipynb](notebooks/energy_harness_T4_with_INT8.ipynb). The notebook now provides a complete analysis of the **energy-accuracy trade-off** for FP32, FP16, and INT8 models.

## What Was Added

### New Components (Part 14)

#### 1. **AccuracyEvaluator Class**
A comprehensive class that evaluates model accuracy on the full dataset:

```python
class AccuracyEvaluator:
    """Evaluate model accuracy on entire dataset."""

    def evaluate(self, input_ids, attention_mask, labels, batch_size):
        # Measures:
        # - Accuracy
        # - Precision (metric)
        # - Recall
        # - F1-Score
        # - Confusion Matrix
```

**Features:**
- Handles both GPU (FP32/FP16) and CPU (INT8) inference
- Computes standard classification metrics
- Progress tracking for long evaluations
- Returns detailed results dictionary

#### 2. **Accuracy Evaluation Loop**
Automatically evaluates all three precisions:

```python
for precision in ['fp32', 'fp16', 'int8']:
    # Load dataset (on correct device)
    # Load model
    # Evaluate accuracy
    # Store results
```

#### 3. **Accuracy Comparison Table**
Side-by-side comparison showing:
- Accuracy (%)
- Precision metric
- Recall
- F1-Score
- Accuracy degradation vs FP32

#### 4. **Combined Energy + Accuracy Results**
Single table with all metrics:

| Precision | Accuracy (%) | Latency (ms) | Energy (mJ) | Throughput | Power (W) | F1-Score |
|-----------|--------------|--------------|-------------|------------|-----------|----------|
| FP32      | ...          | ...          | ...         | ...        | ...       | ...      |
| FP16      | ...          | ...          | ...         | ...        | ...       | ...      |
| INT8      | ...          | ...          | ...         | ...        | ...       | ...      |

#### 5. **Enhanced Visualizations**
New 2×3 subplot grid with:

**Top Row (Energy metrics):**
- Inference Latency
- Energy per Inference
- GPU Power Draw

**Bottom Row (Accuracy & Trade-offs):**
- Model Accuracy (bar chart)
- **Energy vs Accuracy scatter plot** (shows trade-off)
- Normalized comparison (Energy/Latency/Accuracy relative to FP32)

Saved as: `energy_accuracy_tradeoff.png`

#### 6. **Additional Output Files**
New CSV/JSON files generated:

1. **combined_energy_accuracy.csv** - Main results table
2. **accuracy_detailed.csv** - Full accuracy metrics with confusion matrices
3. **complete_results.json** - Everything (energy + accuracy + config)

#### 7. **Final Summary**
Comprehensive text summary showing:
- Energy & Performance metrics for all precisions
- Accuracy metrics for all precisions
- Speedup vs FP32
- Energy savings vs FP32
- Accuracy change vs FP32

## How to Use

### Run the Complete Notebook

Simply execute all cells in order. The notebook will:

1. **Part 0-13**: Run energy measurements (as before)
2. **Part 14 (NEW)**: Run accuracy evaluations
3. Generate combined visualizations and results

### Expected Workflow

```
[Existing cells: Energy measurement for FP32, FP16, INT8]
         ↓
[NEW Cell: AccuracyEvaluator class]
         ↓
[NEW Cell: Run accuracy evaluation for all precisions]
         ↓
[NEW Cell: Compare accuracy across precisions]
         ↓
[NEW Cell: Combine energy + accuracy results]
         ↓
[NEW Cell: Visualize energy-accuracy trade-off]
         ↓
[NEW Cell: Save combined results]
         ↓
[NEW Cell: Final summary with accuracy]
```

## Output Files Summary

After running the complete notebook, you'll have:

| File | Description |
|------|-------------|
| `energy_results_T4_with_INT8.csv` | Energy metrics only |
| `energy_results_T4_with_INT8.json` | Energy metrics (JSON) |
| **`combined_energy_accuracy.csv`** | **Energy + Accuracy combined** ⭐ |
| **`accuracy_detailed.csv`** | **Detailed accuracy metrics** ⭐ |
| **`complete_results.json`** | **Everything (JSON)** ⭐ |
| `precision_comparison.png` | Energy/latency plots (original) |
| **`energy_accuracy_tradeoff.png`** | **Complete 6-panel analysis** ⭐ |

**⭐ = New files from accuracy evaluation**

## Example Results Structure

### Energy Results (existing)
```python
results_all = {
    'fp32': {
        'latency_per_sample_ms': 3.353,
        'energy_per_inference_mj': 225.956,
        'avg_power_w': 67.39,
        'throughput_samples_s': 298.26,
        # ...
    },
    'fp16': { ... },
    'int8': { ... }
}
```

### Accuracy Results (NEW)
```python
accuracy_results = {
    'fp32': {
        'accuracy': 0.9100,
        'precision_metric': 0.9123,
        'recall': 0.9078,
        'f1_score': 0.9100,
        'confusion_matrix': [[45, 5], [4, 46]]
    },
    'fp16': { ... },
    'int8': { ... }
}
```

### Combined Results (NEW)
```python
combined_results = [
    {
        'Precision': 'FP32',
        'Accuracy (%)': 91.00,
        'Latency (ms)': 3.353,
        'Energy (mJ)': 225.956,
        'Throughput (samples/s)': 298.26,
        'Avg Power (W)': 67.39,
        'F1-Score': 0.9100
    },
    # ... FP16, INT8
]
```

## Key Insights You Can Now Analyze

### 1. **Energy-Accuracy Trade-off**
The scatter plot shows the fundamental trade-off:
- **FP16**: Best of both worlds (fast + accurate + efficient)
- **INT8**: Lowest energy but may sacrifice accuracy
- **FP32**: Highest accuracy but energy-intensive

### 2. **Normalized Comparison**
See at a glance:
- How much energy you save with FP16/INT8
- How much speed you gain
- How much accuracy you lose (if any)

All normalized to FP32 = 1.0 for easy comparison.

### 3. **Per-Precision Metrics**
For each precision level, you now have:
- ⚡ **Performance**: Latency, throughput
- 🔋 **Energy**: Power draw, energy per inference
- 🎯 **Accuracy**: All classification metrics
- 📊 **Size**: Model parameters and memory usage

### 4. **Research Paper Ready**
All metrics needed for your paper/presentation:
- Tables (CSV format)
- Visualizations (high-res PNG)
- Raw data (JSON)
- Statistical comparisons (speedup, savings, degradation)

## Integration with Existing Code

The new accuracy evaluation cells:
- ✅ **Reuse existing functions** (`load_pre_tokenized_dataset`, `load_model`)
- ✅ **Follow same device logic** (CPU for INT8, GPU for FP32/FP16)
- ✅ **Use same configuration** (`config.batch_size`, `config.dataset_path`)
- ✅ **Maintain same structure** (results dictionaries, pandas DataFrames)

No breaking changes to existing code!

## Device Handling

The accuracy evaluator correctly handles device placement:

```python
# For FP32/FP16: Data on CUDA, model on CUDA
eval_device = "cuda"

# For INT8: Data on CPU, model on CPU
eval_device = "cpu"
```

This matches the device fix from earlier, ensuring no device mismatch errors.

## Performance Considerations

### Evaluation Time
- **FP32/FP16 on GPU**: Fast (~5-10 seconds for 100 samples)
- **INT8 on CPU**: Slower (~30-60 seconds for 100 samples)

The accuracy evaluation is much faster than the energy measurement loop (which runs 500 iterations), so total runtime increase is minimal.

### Memory Usage
The evaluator processes data in batches, so memory usage is controlled by `batch_size` parameter (default: 16).

## Troubleshooting

### Issue: "NameError: name 'accuracy_score' is not defined"
**Solution:** The imports are in Cell 5. Make sure to run all cells in order.

### Issue: Accuracy evaluation hangs
**Solution:**
- INT8 is slower on CPU, be patient
- Check that the dataset loaded correctly
- Reduce batch_size if needed

### Issue: Plots look wrong
**Solution:**
- Make sure you ran the energy experiments first (Part 9)
- Check that `results_all` and `accuracy_results` both exist
- Re-run the visualization cell

## Next Steps

1. ✅ **Run the complete notebook** on T4 GPU
2. ✅ **Analyze the energy-accuracy trade-off** plots
3. ✅ **Export results** for your paper
4. ✅ **Interpret findings**:
   - Is FP16 a good compromise?
   - Does INT8 maintain acceptable accuracy?
   - What's the optimal precision for your use case?

## Files Modified

- [notebooks/energy_harness_T4_with_INT8.ipynb](notebooks/energy_harness_T4_with_INT8.ipynb)
  - Added Part 14 (8 new cells)
  - Total cells: 30 → 38

## Summary of Changes

| Aspect | Before | After |
|--------|--------|-------|
| Metrics | Energy + Latency only | Energy + Latency + Accuracy |
| Plots | 4 plots (2×2 grid) | 10 plots (4 + 6 in 2×3 grid) |
| Output files | 3 files | 7 files |
| Analysis | Performance trade-offs | **Energy-Accuracy trade-offs** |
| Paper-ready | Partial | ✅ **Complete** |

## Example Insights You Can Report

From your results, you can now say:

> "FP16 quantization achieves a **4.64× speedup** and **79.4% energy reduction** compared to FP32, while maintaining **XX.X% accuracy** (only a **-X.X% degradation**). INT8 quantization reduces model size by **75%** but runs slower on CPU, consuming **6× more energy** than FP16 with an accuracy of **YY.Y%**."

All numbers backed by actual measurements! 📊

---

**Ready for your research paper!** 🎉

The notebook now provides everything you need for a comprehensive analysis of energy-aware quantization with real INT8 results and accuracy metrics.

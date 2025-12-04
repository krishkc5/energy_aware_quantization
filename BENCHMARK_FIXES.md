# Benchmark Fixes and Justifications

## Issues Identified and Fixed

### 1. BF16 Performance Issue (CRITICAL FIX)

**Problem:** BF16 was showing worse performance than FP32 (higher latency, lower throughput, higher energy).

**Root Cause:** Tesla T4 has compute capability 7.5, which does NOT natively support BF16. BF16 requires Ampere or newer (compute capability >= 8.0). PyTorch's `torch.cuda.is_bf16_supported()` can return `True` even when BF16 is emulated (slow) rather than natively supported (fast).

**Fix:** Added compute capability check to exclude BF16 on GPUs with compute capability < 8.0. BF16 will now be excluded from benchmarks on Tesla T4.

**Expected Result After Fix:**
- BF16 will not be tested on T4
- Remaining formats: FP32, FP16, MIXED
- Order should be: FP32 (slowest) > MIXED > FP16 (fastest)

### 2. Input Dtype Mismatch

**Problem:** Input tensors might not match model precision, potentially affecting accuracy and performance measurements.

**Root Cause:** `attention_mask` was not being cast to match the model's dtype for FP16/BF16 models.

**Fix:** Added `model_dtype` parameter to `benchmark_model()` and ensure `attention_mask` matches model dtype for non-mixed precision models.

**Expected Result:** More accurate performance and accuracy measurements.

### 3. Accuracy Measurements

**Current Observation:** All models show 91.00% accuracy.

**Possible Explanations:**
1. **This might be correct:** FP16 quantization on well-trained models (like DistilBERT fine-tuned on SST-2) often doesn't cause accuracy loss. The model was likely trained to be robust to quantization.
2. **Model quantization is working:** The dtype conversions are being applied correctly, but the model maintains accuracy.

**Note:** If you expect accuracy loss, you might need to:
- Use a different model that's more sensitive to quantization
- Use INT8 quantization instead of FP16/BF16
- Check if the model is actually being quantized (verify dtypes in model parameters)

### 4. Model Size for MIXED Precision

**Current Observation:** MIXED shows same size as FP32 (255.41 MB).

**Explanation:** This is **CORRECT**. Mixed precision (torch.autocast) uses:
- **FP32 weights** (stored in memory) - same size as FP32
- **FP16 computation** (during forward pass) - faster execution

Mixed precision does NOT reduce model size, only computation precision during inference. The weights remain in FP32 format.

**Expected Model Sizes:**
- FP32: ~255 MB (baseline)
- FP16: ~128 MB (half the size)
- BF16: ~128 MB (half the size) 
- MIXED: ~255 MB (same as FP32, uses FP32 weights)

## Expected Results After Fixes

### Latency (Lower is Better)
**Expected Order:** FP32 > MIXED > FP16
- FP32: Highest (baseline)
- MIXED: Medium (FP32 weights, FP16 computation)
- FP16: Lowest (full FP16 model)

### Speedup vs FP32 (Higher is Better)
**Expected Order:** FP32 (1.0x) < MIXED < FP16
- FP32: 1.0x (baseline)
- MIXED: ~4-5x (depending on hardware)
- FP16: ~4-5x (similar to MIXED, sometimes slightly faster)

### Energy per Sample (Lower is Better)
**Expected Order:** FP32 > MIXED > FP16
- FP32: Highest energy
- MIXED: Medium energy
- FP16: Lowest energy

### Throughput (Higher is Better)
**Expected Order:** FP32 < MIXED < FP16
- FP32: Lowest throughput
- MIXED: Medium throughput
- FP16: Highest throughput

### Accuracy (Higher is Better)
**Expected:** Some small loss possible, but FP16 often maintains accuracy
- FP32: Baseline accuracy
- FP16: May show slight decrease (0-1%), but often maintains accuracy
- MIXED: Should match FP32 (uses FP32 weights)

### Model Size (Lower is Better)
**Expected Order:** FP32 = MIXED > FP16
- FP32: ~255 MB
- MIXED: ~255 MB (same as FP32 - uses FP32 weights)
- FP16: ~128 MB (half size)

## Summary of Changes Made

1. **Cell 4 (Format Support Check):**
   - Added compute capability check for BF16
   - BF16 now excluded on GPUs with compute capability < 8.0
   - Added warning message when BF16 is emulated vs native

2. **Cell 12 (Benchmark Function):**
   - Added `model_dtype` parameter
   - Added input dtype matching for `attention_mask` to match model precision

3. **Cell 14 (Run Benchmarks):**
   - Pass `model_dtype` to benchmark function
   - Ensures proper dtype matching for accurate measurements

## Next Steps

1. **Run the notebook again** - BF16 should now be excluded on T4
2. **Verify results match expectations:**
   - FP32 should be slowest
   - FP16 should be fastest
   - MIXED should be in between
3. **If accuracy is still identical:** This might be correct behavior for FP16 quantization on this model. Consider testing with INT8 if you want to see accuracy loss.

## Technical Notes

- **Tesla T4:** Compute capability 7.5, supports FP32 and FP16 natively
- **BF16 Native Support:** Requires Ampere (A100, RTX 30xx, etc.) or newer
- **Mixed Precision:** Uses FP32 weights with FP16 computation - best of both worlds (accuracy + speed)
- **FP16 Quantization:** Often maintains accuracy on well-trained models, especially for inference tasks


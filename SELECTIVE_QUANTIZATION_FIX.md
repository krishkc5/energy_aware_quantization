# Fix for Selective Quantization Dtype Mismatch Error

## Problem

When selectively quantizing layers (converting some to FP16 while keeping others in FP32), you get:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Half
```

This happens because FP32 activations from earlier layers flow into FP16 layers, causing dtype mismatches.

## Root Cause

The current `quantize_layer_to_fp16()` function converts layer parameters to FP16 but doesn't handle input dtype conversion:

```python
def quantize_layer_to_fp16(model, layer_name):
    # ... (get the layer module)

    if isinstance(module, (nn.Linear, nn.LayerNorm)):
        module.to(torch.float16)  # ← This only converts parameters, not inputs!
```

When FP32 inputs hit this FP16 layer → **dtype mismatch error**.

## Solution: Add Forward Hook for Automatic Dtype Conversion

Replace the `quantize_layer_to_fp16()` function in **Cell 26** with this fixed version:

```python
def quantize_layer_to_fp16(model, layer_name):
    """
    Quantize a specific layer to FP16 while keeping the rest in FP32.
    Adds a forward hook to automatically convert inputs to FP16.

    Args:
        model: The model to modify
        layer_name: Full name of the layer (e.g., "distilbert.transformer.layer.0.ffn.lin1")

    Returns:
        Modified model (in-place modification)
    """
    # Get the layer module using getattr recursively
    parts = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Get the actual layer module
    layer_part = parts[-1]
    module = getattr(parent, layer_part)

    # Convert layer to FP16
    if isinstance(module, (nn.Linear, nn.LayerNorm)):
        # Convert all parameters and buffers to FP16
        module.to(torch.float16)

        # Add forward hook to convert inputs to FP16
        def fp16_input_hook(module, input_tuple):
            """Convert inputs to FP16 to match layer dtype"""
            if isinstance(input_tuple[0], torch.Tensor):
                # Convert first input (typically the activation tensor)
                return (input_tuple[0].to(torch.float16),) + input_tuple[1:]
            return input_tuple

        # Register the hook
        module.register_forward_pre_hook(fp16_input_hook)

    elif isinstance(module, nn.Embedding):
        # Embeddings typically stay in FP32 for stability
        pass

    return model
```

## How the Fix Works

1. **Layer parameters** → Converted to FP16 (as before)
2. **Forward pre-hook** → Added to layer
3. **During forward pass**:
   - Input arrives (may be FP32 or FP16)
   - Hook converts it to FP16
   - Layer processes FP16 inputs with FP16 parameters ✓
   - No dtype mismatch!

## What Changed

### Before (Broken)
```
FP32 Layer → [FP32 output] → FP16 Layer (expects FP16 input) → ERROR!
                                    ↑
                                Parameters are FP16,
                                but input is FP32
```

### After (Fixed)
```
FP32 Layer → [FP32 output] → [Hook converts to FP16] → FP16 Layer → Success!
                                          ↑
                                    Automatic conversion
```

## Alternative Solutions

If the hook approach doesn't work for some reason, here are alternatives:

### Option 1: Convert Entire Model to FP16

Instead of selective quantization, convert the whole model:

```python
# Don't use selective quantization
model = model.half()  # Convert everything to FP16
```

**Pros:** No dtype mismatches
**Cons:** Loses the benefit of selective quantization

### Option 2: Use torch.autocast

Use mixed precision with autocast instead:

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
```

**Pros:** Automatic dtype management
**Cons:** Different from manual selective quantization

### Option 3: Manual Activation Conversion

Convert activations between layers manually (complex, not recommended):

```python
# Would need to modify the forward pass extensively
# Not practical for transformer models
```

## Testing the Fix

After applying the fix, run these cells in order:

1. **Cell 26** (with updated `quantize_layer_to_fp16`)
2. **Cell 27** (baseline accuracy measurement)
3. **Cell 28** (accuracy impact analysis) ← Should work now!

Expected output:
```
[1/36] Testing: lin2... ✓ Accuracy: 86.00% (drop: 0.00%)
[2/36] Testing: lin2... ✓ Accuracy: 86.00% (drop: 0.00%)
...
```

No more dtype mismatch errors!

## Additional Notes

- The hook is lightweight and adds minimal overhead
- Each FP16 layer gets its own hook
- The hook only converts inputs, outputs remain FP16
- LayerNorm layers also get the hook (they're often FP16-quantized)
- Embeddings are skipped (they stay in FP32)

## If You Still Get Errors

If you still see dtype mismatches after applying this fix:

1. **Check that the fix was applied** - Re-run Cell 26 after editing
2. **Restart the kernel** - Old function definitions may be cached
3. **Verify the layer type** - Some custom layers may need special handling
4. **Check PyTorch version** - Hooks work slightly differently in older versions

## Summary

**Problem:** Selective FP16 quantization causes dtype mismatches
**Solution:** Add forward pre-hook to auto-convert inputs to FP16
**Result:** Selective quantization works without errors ✓

Now you can run the per-layer energy profiling with selective quantization successfully!

# Critical Issue: Power Logger Only Collecting ~2% of Samples

## The Problem

All three experiments (FP32, FP16, INT8) successfully ran on CUDA, but the power logger is only collecting **17-18 samples** instead of the expected **~800+ samples** for 80-second runs.

### Evidence from Kaggle Runs

| Experiment | Runtime | Power Samples | Expected Samples | Collection Rate |
|------------|---------|---------------|------------------|-----------------|
| FP32 | 88.5s | 18 | ~885 | **2.0%** |
| FP16 | 83.6s | 17 | ~836 | **2.0%** |
| INT8 | 88.5s | 18 | ~885 | **2.0%** |

**Sampling interval**: 100ms (0.1s) → Should collect ~10 samples/second

### Why This Matters

1. ❌ **Energy measurements are unreliable** - based on only 2% of actual power data
2. ❌ **High variance** - Power Std of 42-45W indicates incomplete sampling (should be <10W)
3. ❌ **Can't draw conclusions** - Can't compare energy efficiency across precisions

### Current Results (Unreliable Energy Data)

| Metric | FP32 | FP16 | INT8 |
|--------|------|------|------|
| Latency | 88.39 ms | 83.54 ms | 88.42 ms |
| Throughput | 564.97 samples/s | 597.75 samples/s | 564.81 samples/s |
| Accuracy | 86.00% | 86.00% | 88.00% |
| Mean Power | 225.83 W | 228.59 W | 224.43 W |
| **Power Std** | **43.89 W** ⚠️ | **45.01 W** ⚠️ | **42.83 W** ⚠️ |
| Energy/Inf | 19.99 J | 19.12 J | 19.87 J |

**Performance metrics (latency, throughput, accuracy) are valid ✅**
**Energy metrics are unreliable ❌**

---

## Root Cause Analysis

The power logger uses `subprocess.Popen` to run `nvidia-smi` in loop mode:

```bash
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id=0 -l 0.1
```

This command outputs power readings every 0.1 seconds (100ms).

### The Buffering Issue

Python's subprocess stdout is **buffered** by default. Even with `bufsize=0`, the issue persists because:

1. **Output buffering in nvidia-smi itself** - nvidia-smi may buffer output before writing to stdout
2. **OS-level buffering** - The operating system may buffer pipe writes
3. **readline() blocking** - `readline()` waits for a newline, which may not arrive immediately

### Why We Only Get ~18 Samples

- nvidia-smi outputs ~10 samples/second (every 100ms)
- But Python's `readline()` blocks waiting for buffered output
- Only when the buffer fills (~8KB) or process ends does output flush
- Result: We only see output sporadically (~18 times during 80s run)

---

## Attempted Fixes

### Fix #1: Binary Unbuffered Mode (Tried)
```python
self.process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    text=False,  # Binary mode
    bufsize=0,   # Unbuffered
)
```
**Result**: Still only 17-18 samples ❌

### Fix #2: Byte-by-Byte Reading (Tried)
```python
buffer = b""
while self.is_running:
    chunk = self.process.stdout.read(1)  # Read 1 byte at a time
    buffer += chunk
    if chunk == b'\n':
        # Process line
```
**Result**: Too slow, still buffering issues ❌

### Fix #3: iter() with readline (Current)
```python
for line_bytes in iter(self.process.stdout.readline, b''):
    # Process line
```
**Result**: Not yet tested on Kaggle

---

## Potential Solutions

### Solution A: Use `pexpect` Library (Recommended)

`pexpect` provides better control over subprocess I/O without buffering issues.

**Install**:
```bash
pip install pexpect
```

**Modified power_logger.py**:
```python
import pexpect

def start(self) -> None:
    cmd = f"nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id={self.gpu_id} -l {loop_interval_seconds}"

    # Use pexpect for unbuffered output
    self.process = pexpect.spawn(cmd, encoding='utf-8', timeout=None)

    # Start reader thread
    self._reader_thread = threading.Thread(target=self._read_samples, daemon=True)
    self._reader_thread.start()

def _read_samples(self) -> None:
    try:
        while self.is_running:
            line = self.process.readline().strip()
            if line:
                try:
                    power = float(line)
                    with self._lock:
                        self.samples.append(power)
                except ValueError:
                    pass
    except (pexpect.EOF, pexpect.TIMEOUT):
        pass
```

### Solution B: Use `stdbuf` to Disable nvidia-smi Buffering

Force nvidia-smi to flush output immediately:

```python
cmd = [
    "stdbuf", "-oL",  # Line-buffered output
    "nvidia-smi",
    "--query-gpu=power.draw",
    "--format=csv,noheader,nounits",
    f"--id={self.gpu_id}",
    "-l", str(loop_interval_seconds)
]
```

### Solution C: Poll nvidia-smi Manually

Instead of continuous mode, poll nvidia-smi every 100ms:

```python
def _read_samples(self) -> None:
    while self.is_running:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", f"--id={self.gpu_id}"],
                capture_output=True,
                text=True,
                timeout=1
            )
            power = float(result.stdout.strip())
            with self._lock:
                self.samples.append(power)
        except:
            pass

        time.sleep(self.sample_interval_ms / 1000.0)
```

**Pros**: Simple, guaranteed to work
**Cons**: Higher overhead from spawning processes repeatedly

---

## Recommended Action Plan

### Option 1: Try Solution B (stdbuf) - No Dependencies

1. Update power_logger.py to use `stdbuf -oL`
2. Commit and push changes
3. Pull on Kaggle and re-run experiments
4. If still doesn't work, proceed to Option 2

### Option 2: Use Solution C (Manual Polling) - Guaranteed to Work

1. Implement manual polling approach
2. Test locally first
3. Commit and push
4. Run on Kaggle

### Option 3: Use Solution A (pexpect) - Best but Requires Install

1. Add pexpect to requirements.txt
2. Modify power_logger.py to use pexpect
3. Install on Kaggle: `!pip install pexpect`
4. Run experiments

---

## Testing the Fix

After implementing any solution, verify:

```python
# Should see this output:
Power samples collected: 10
Power samples collected: 20
Power samples collected: 30
...
Power samples collected: 880

✓ Collected 880 power samples  ← Should see ~800+, not 17-18!
```

**Expected power variance**: Std < 10W (currently 42-45W)

---

## Current Status

✅ **Performance measurements**: Valid (latency, throughput, accuracy all measured correctly)
✅ **INT8 on CUDA**: Working perfectly (simulated quantization approach validated)
✅ **All three precisions run**: FP32, FP16, INT8 all complete successfully
❌ **Energy measurements**: Unreliable due to insufficient power samples

---

## Next Steps

1. Choose a solution (recommend starting with Solution B - stdbuf)
2. Implement the fix
3. Test locally if possible
4. Commit and push
5. Pull on Kaggle
6. Re-run all three experiments
7. Verify power sample collection (~800+ samples)
8. Check power variance (should be <10W)
9. Use corrected energy data for final analysis

---

## Alternative: Manual Power Measurement

If automated power logging continues to fail, you can manually measure power:

1. Open Kaggle in two windows
2. In window 1: Run `watch -n 0.1 nvidia-smi` to monitor power
3. In window 2: Run experiment
4. Manually record power readings during experiment
5. Calculate average power manually

This is less ideal but would give you valid energy data.

---

## Performance Results (Valid)

These results are reliable and can be used for analysis:

### Latency Comparison
- FP16 is **5.5% faster** than FP32 (83.54ms vs 88.39ms)
- INT8 is **same as FP32** (88.42ms vs 88.39ms)

### Throughput Comparison
- FP16: **597.75 samples/s** (5.8% higher than FP32)
- FP32: **564.97 samples/s** (baseline)
- INT8: **564.81 samples/s** (same as FP32)

### Accuracy Comparison
- All three maintain high accuracy: 86-88%
- INT8 actually shows **2% higher accuracy** (88% vs 86%) - likely statistical variance

### Model Size Comparison
- FP32: **255.42 MB** (baseline)
- FP16: **127.71 MB** (50% smaller) ✅
- INT8: **255.42 MB** (same as FP32 - weights stored as FP32 after quantization)

---

## Conclusion

The experiments **ran successfully**, but energy measurements are **unreliable** due to insufficient power sampling. The power logger needs to be fixed before final energy analysis can be performed.

However, the **performance and accuracy results are valid** and demonstrate that:
- FP16 provides ~6% latency improvement and 50% memory reduction with no accuracy loss
- INT8 maintains accuracy but doesn't provide latency benefits (expected for simulated quantization)

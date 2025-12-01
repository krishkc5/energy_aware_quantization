#!/usr/bin/env python3
"""
Test nvidia-smi continuous monitoring on Kaggle.

This script tests different nvidia-smi command formats to find what works.
"""

import subprocess
import time
import threading

print("="*70)
print("NVIDIA-SMI CONTINUOUS MONITORING TEST")
print("="*70)

# Test 1: Basic nvidia-smi
print("\n[Test 1] Basic nvidia-smi query...")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=2
    )
    if result.returncode == 0:
        print(f"  ✓ Works! Power: {result.stdout.strip()} W")
    else:
        print(f"  ✗ Failed: {result.stderr}")
except Exception as e:
    print(f"  ✗ Exception: {e}")

# Test 2: Loop mode with space (nvidia-smi -lms 100)
print("\n[Test 2] Loop mode: -lms 100 (with space)...")
print("  Starting subprocess...")
samples = []
try:
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    print("  Collecting samples for 2 seconds...")
    start_time = time.time()
    while time.time() - start_time < 2.0:
        line = process.stdout.readline()
        if line:
            try:
                power = float(line.strip())
                samples.append(power)
                print(f"    Sample {len(samples)}: {power:.2f} W")
            except:
                print(f"    Parse error: {line.strip()}")

    process.terminate()
    process.wait()

    if len(samples) > 0:
        print(f"  ✓ Works! Collected {len(samples)} samples")
        print(f"    Mean power: {sum(samples)/len(samples):.2f} W")
    else:
        print(f"  ✗ No samples collected")
        stderr = process.stderr.read()
        if stderr:
            print(f"    stderr: {stderr}")

except Exception as e:
    print(f"  ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Loop mode without space (nvidia-smi -lms100)
print("\n[Test 3] Loop mode: -lms100 (no space)...")
print("  Starting subprocess...")
samples = []
try:
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    print("  Collecting samples for 2 seconds...")
    start_time = time.time()
    while time.time() - start_time < 2.0:
        line = process.stdout.readline()
        if line:
            try:
                power = float(line.strip())
                samples.append(power)
                print(f"    Sample {len(samples)}: {power:.2f} W")
            except:
                print(f"    Parse error: {line.strip()}")

    process.terminate()
    process.wait()

    if len(samples) > 0:
        print(f"  ✓ Works! Collected {len(samples)} samples")
        print(f"    Mean power: {sum(samples)/len(samples):.2f} W")
    else:
        print(f"  ✗ No samples collected")
        stderr = process.stderr.read()
        if stderr:
            print(f"    stderr: {stderr}")

except Exception as e:
    print(f"  ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Loop mode with -l flag and separate ms value
print("\n[Test 4] Loop mode: -l -ms 100 (separate flags)...")
print("  Starting subprocess...")
samples = []
try:
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-l", "0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    print("  Collecting samples for 2 seconds...")
    start_time = time.time()
    while time.time() - start_time < 2.0:
        line = process.stdout.readline()
        if line:
            try:
                power = float(line.strip())
                samples.append(power)
                print(f"    Sample {len(samples)}: {power:.2f} W")
            except:
                print(f"    Parse error: {line.strip()}")

    process.terminate()
    process.wait()

    if len(samples) > 0:
        print(f"  ✓ Works! Collected {len(samples)} samples")
        print(f"    Mean power: {sum(samples)/len(samples):.2f} W")
    else:
        print(f"  ✗ No samples collected")
        stderr = process.stderr.read()
        if stderr:
            print(f"    stderr: {stderr}")

except Exception as e:
    print(f"  ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Threading approach (like PowerLogger)
print("\n[Test 5] Threading approach (like PowerLogger)...")
print("  Starting subprocess with background thread...")

samples_threaded = []
is_running = True

def read_samples(process):
    global samples_threaded, is_running
    try:
        for line in iter(process.stdout.readline, ''):
            if not is_running:
                break
            line = line.strip()
            if line:
                try:
                    power = float(line)
                    samples_threaded.append(power)
                    if len(samples_threaded) % 5 == 0:
                        print(f"    Samples collected: {len(samples_threaded)}")
                except ValueError:
                    print(f"    Parse error: {line}")
    except Exception as e:
        print(f"    Thread exception: {e}")

try:
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    reader_thread = threading.Thread(target=read_samples, args=(process,), daemon=True)
    reader_thread.start()

    print("  Collecting samples for 3 seconds...")
    time.sleep(3.0)

    is_running = False
    process.terminate()
    process.wait()
    reader_thread.join(timeout=1)

    if len(samples_threaded) > 0:
        print(f"  ✓ Works! Collected {len(samples_threaded)} samples")
        print(f"    Mean power: {sum(samples_threaded)/len(samples_threaded):.2f} W")
        print(f"    Min power: {min(samples_threaded):.2f} W")
        print(f"    Max power: {max(samples_threaded):.2f} W")
    else:
        print(f"  ✗ No samples collected")
        stderr = process.stderr.read() if process.stderr else ""
        if stderr:
            print(f"    stderr: {stderr}")

except Exception as e:
    print(f"  ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nRecommendation:")
print("  If Test 2 or 3 worked: Use that format in PowerLogger")
print("  If Test 5 worked: Current PowerLogger implementation should work")
print("  If none worked: Kaggle may not support continuous nvidia-smi")
print("="*70)

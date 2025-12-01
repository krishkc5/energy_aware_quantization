# Energy-Aware Quantization for LLM Inference

**A production-ready framework for measuring accuracy, latency, and per-inference energy consumption of quantized transformer models.**

This project provides a comprehensive, zero-I/O measurement harness for comparing FP32, FP16, and INT8 versions of transformer models (DistilBERT, GPT2) on Kaggle GPUs.

---

## Features

- **Zero-I/O Design**: All data pre-tokenized and GPU-resident before measurement
- **Comprehensive Metrics**: Accuracy, latency, throughput, power, and energy per inference
- **GPU Power Logging**: Asynchronous nvidia-smi monitoring with configurable sampling
- **Multiple Precision Modes**: FP32, FP16 (native and autocast), and INT8 (dynamic quantization)
- **Reproducible**: Warmup routines, synchronized timing, and controlled measurement
- **Production-Ready**: Type hints, error handling, validation, and comprehensive logging

---

## Repository Structure

```
energy_aware_quantization/
│
├── datasets/                   # Pre-tokenized SST-2 datasets
│   ├── tokenized_data/         # Standard dataset (50 samples)
│   ├── tokenized_data_small/   # Small dataset for testing
│   ├── tokenized_data_large/   # Large dataset for benchmarks
│   └── tokenized_data_standard/
│
├── models/                     # Model loading utilities
│   ├── __init__.py
│   └── model_loader.py         # FP32/FP16/INT8 model loaders
│
├── src/                        # Measurement harness
│   ├── __init__.py
│   ├── dataset_loader.py       # Pre-tokenized data loading
│   ├── warmup.py               # GPU warmup routines
│   ├── power_logger.py         # Async power monitoring
│   ├── inference_runner.py     # Timed inference loops
│   ├── energy_utils.py         # Energy computation
│   └── measure_energy.py       # Main orchestrator script
│
├── notebooks/                  # Jupyter notebooks
│   └── krishna.ipynb          # Kaggle notebook
│
├── results/                    # Output directory
│   ├── fp32/
│   ├── fp16/
│   └── int8/
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Installation

### On Kaggle

1. Clone the repository:

```bash
cd /kaggle/working
git clone https://github.com/YOUR_USERNAME/energy_aware_quantization.git
cd energy_aware_quantization
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/energy_aware_quantization.git
cd energy_aware_quantization
pip install -r requirements.txt
```

---

## Quick Start

### Basic Usage

Run a single experiment:

```bash
python src/measure_energy.py \
    --precision fp32 \
    --dataset datasets/tokenized_data \
    --num_iters 1000
```

### Run All Precision Modes

```bash
# FP32 baseline
python src/measure_energy.py --precision fp32 --dataset datasets/tokenized_data --num_iters 1000

# FP16 (half precision)
python src/measure_energy.py --precision fp16 --dataset datasets/tokenized_data --num_iters 1000

# INT8 (quantized)
python src/measure_energy.py --precision int8 --dataset datasets/tokenized_data --num_iters 1000
```

### Run Multiple Trials

```bash
for i in {1..5}; do
    python src/measure_energy.py \
        --precision fp32 \
        --dataset datasets/tokenized_data \
        --num_iters 1000 \
        --trial $i
done
```

---

## Command Line Arguments

| Argument                    | Type | Default                                             | Description                                    |
| --------------------------- | ---- | --------------------------------------------------- | ---------------------------------------------- |
| `--model`                 | str  | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model name                         |
| `--precision`             | str  | **required**                                  | Precision mode:`fp32`, `fp16`, or `int8` |
| `--dataset`               | str  | **required**                                  | Path to pre-tokenized dataset                  |
| `--num_iters`             | int  | 1000                                                | Number of inference iterations                 |
| `--warmup_steps`          | int  | 100                                                 | Number of warmup iterations                    |
| `--power_sample_interval` | int  | 100                                                 | Power sampling interval (ms)                   |
| `--device`                | str  | `cuda`                                            | Device to use                                  |
| `--gpu_id`                | int  | 0                                                   | GPU device ID                                  |
| `--output`                | str  | auto                                                | Output CSV path                                |
| `--trial`                 | int  | 1                                                   | Trial number                                   |
| `--no_warmup`             | flag | False                                               | Skip warmup (not recommended)                  |
| `--verbose`               | flag | False                                               | Enable verbose output                          |

---

## Output

Each experiment produces two files:

1. **CSV file**: `results/<precision>/trial_<N>_<timestamp>.csv`

   - Single-row dataframe with all metrics
   - Easy to concatenate for analysis
2. **JSON file**: `results/<precision>/trial_<N>_<timestamp>.json`

   - Human-readable format with all metrics
   - Includes metadata and configuration

### Metrics Included

**Performance:**

- `mean_latency`: Mean inference latency (seconds)
- `std_latency`: Standard deviation of latency
- `throughput`: Samples per second
- `min_latency`, `max_latency`, `median_latency`

**Power:**

- `mean_power_w`: Mean GPU power draw (Watts)
- `std_power_w`: Power standard deviation
- `min_power_w`, `max_power_w`, `median_power_w`
- `num_power_samples`: Number of samples collected

**Energy:**

- `total_energy_j`: Total energy consumed (Joules)
- `energy_per_inference_j`: Energy per inference (Joules)
- `energy_per_inference_mj`: Energy per inference (millijoules)
- `inferences_per_joule`: Efficiency metric

**Accuracy:**

- `accuracy`: Classification accuracy (0-1)
- `num_correct`: Number of correct predictions
- `num_samples_accuracy`: Total samples

**Memory:**

- `allocated_gb`: GPU memory allocated
- `peak_allocated_gb`: Peak memory usage
- `reserved_gb`: GPU memory reserved

**Model Info:**

- `num_parameters`: Total model parameters
- `model_size_mb`: Model size in MB
- `dtype`: Parameter data type

---

## Workflow

The measurement harness follows this pipeline:

1. **GPU Check**: Verify CUDA availability and GPU readiness
2. **Load Dataset**: Load pre-tokenized tensors directly to GPU
3. **Load Model**: Initialize model in specified precision mode
4. **Warmup**: Stabilize GPU clocks and CUDA kernels
5. **Start Power Logging**: Launch nvidia-smi in background
6. **Run Inference**: Execute timed measurement loop
7. **Stop Power Logging**: Collect power samples
8. **Compute Energy**: Calculate energy metrics
9. **Save Results**: Export CSV and JSON files

---

## Design Principles

### Zero-I/O Measurement

All I/O operations (dataset loading, model loading) happen **before** the measurement loop:

- Dataset tensors loaded to GPU once at startup
- Model loaded and moved to GPU before warmup
- No CPU-GPU transfers inside measurement loop
- No disk access during measurement

### Precise Timing

- Uses `time.perf_counter()` for high-resolution timing
- CUDA synchronization after each forward pass
- Warmup phase to stabilize GPU clocks
- Multiple iterations for statistical significance

### Asynchronous Power Logging

- nvidia-smi runs in separate subprocess
- Does not interfere with inference timing
- Configurable sampling interval (default: 100ms)
- Thread-safe sample collection

---

## API Usage

You can also use the harness components programmatically:

```python
from src import load_pre_tokenized, warmup, PowerLogger, run_inference, compute_energy
from models import load_model

# Load dataset
input_ids, mask, labels, metadata = load_pre_tokenized("datasets/tokenized_data")

# Load model
model = load_model("distilbert-base-uncased-finetuned-sst-2-english", precision="fp16")

# Warmup
warmup(model, input_ids, mask, num_steps=100)

# Measure with power logging
logger = PowerLogger(sample_interval_ms=100)
logger.start()

results = run_inference(model, input_ids, mask, num_iters=1000)

logger.stop()
power_samples = logger.read()

# Compute energy
energy = compute_energy(power_samples, results["total_time"], results["num_iters"])
```

---

## Kaggle Notebook Integration

In your Kaggle notebook:

```python
# Clone or pull latest code
!cd /kaggle/working/energy_aware_quantization && git pull

# Run experiments
!python /kaggle/working/energy_aware_quantization/src/measure_energy.py \
    --precision fp32 \
    --dataset /kaggle/working/energy_aware_quantization/datasets/tokenized_data \
    --num_iters 1000

# Load results
import pandas as pd
df = pd.read_csv("/kaggle/working/energy_aware_quantization/results/fp32/trial_1_*.csv")
print(df)
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in dataset or use smaller dataset variant:

```bash
python src/measure_energy.py --dataset datasets/tokenized_data_small ...
```

### Power Logging Fails

Verify nvidia-smi is available:

```bash
nvidia-smi --query-gpu=power.draw --format=csv
```

### Import Errors

Ensure you're running from repository root:

```bash
cd /path/to/energy_aware_quantization
python src/measure_energy.py ...
```

---

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for more models (BERT, GPT-2, etc.)
- [ ] Implement static INT8 quantization
- [ ] Add support for batch size sweeps
- [ ] Create visualization notebooks
- [ ] Add CPU fallback mode

---

## Citation

If you use this framework in your research, please cite:

```
@misc{energy_aware_quantization,
  title={Energy-Aware Quantization for LLM Inference},
  author={Krishna Karthikeya Chemudupati, Taarana Jammula, and Thomas Ngulube},
  year={2025},
  course={ESE5390},
  institution={University of Pennsylvania}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Acknowledgments

- Course: ESE5390
- Institution: University of Pennsylvania
- Platform: Kaggle GPU infrastructure

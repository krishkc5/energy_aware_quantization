# Energy Aware Quantization

End to end framework for measuring accuracy, latency, and energy consumption
for FP32, FP16, and INT8 LLM inference using zero-IO pre tokenized datasets
and GPU power logging.

Structure:
- datasets/      (pre tokenized SST-2 samples)
- models/        (FP32, FP16, INT8 versions)
- src/           (measurement harness and utilities)
- notebooks/     (development notebooks)
- results/       (latency and energy logs)


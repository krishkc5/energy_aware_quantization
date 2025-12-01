#!/bin/bash

# Run all experiments for FP32, FP16, and INT8
# This script runs multiple trials for each precision mode

set -e  # Exit on error

DATASET="datasets/tokenized_data"
NUM_ITERS=1000
NUM_TRIALS=5

echo "======================================================================"
echo "RUNNING ENERGY-AWARE QUANTIZATION EXPERIMENTS"
echo "======================================================================"
echo "Dataset: $DATASET"
echo "Iterations per trial: $NUM_ITERS"
echo "Number of trials: $NUM_TRIALS"
echo "======================================================================"

# Create results directories
mkdir -p results/fp32 results/fp16 results/int8

# Run FP32 experiments
echo ""
echo "======================================================================"
echo "RUNNING FP32 EXPERIMENTS"
echo "======================================================================"
for i in $(seq 1 $NUM_TRIALS); do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "FP32 Trial $i/$NUM_TRIALS"
    echo "----------------------------------------------------------------------"
    python src/measure_energy.py \
        --precision fp32 \
        --dataset $DATASET \
        --num_iters $NUM_ITERS \
        --trial $i
done

# Run FP16 experiments
echo ""
echo "======================================================================"
echo "RUNNING FP16 EXPERIMENTS"
echo "======================================================================"
for i in $(seq 1 $NUM_TRIALS); do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "FP16 Trial $i/$NUM_TRIALS"
    echo "----------------------------------------------------------------------"
    python src/measure_energy.py \
        --precision fp16 \
        --dataset $DATASET \
        --num_iters $NUM_ITERS \
        --trial $i
done

# Run INT8 experiments
echo ""
echo "======================================================================"
echo "RUNNING INT8 EXPERIMENTS"
echo "======================================================================"
for i in $(seq 1 $NUM_TRIALS); do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "INT8 Trial $i/$NUM_TRIALS"
    echo "----------------------------------------------------------------------"
    python src/measure_energy.py \
        --precision int8 \
        --dataset $DATASET \
        --num_iters $NUM_ITERS \
        --trial $i
done

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "======================================================================"
echo "Results saved in results/ directory"
echo ""
echo "To analyze results, run:"
echo "  python analysis/aggregate_results.py"

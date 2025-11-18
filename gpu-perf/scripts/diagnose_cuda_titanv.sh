#!/bin/bash
# Diagnostic script to check CUDA environment for TITAN V

echo "=== CUDA Version ==="
nvcc --version

echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

echo ""
echo "=== Checking which compute capabilities are supported ==="
# Try to compile a simple test for different architectures
echo "Testing sm_70 (Volta - TITAN V)..."
echo "__global__ void test() {}" > /tmp/test_sm70.cu
if nvcc -arch=sm_70 /tmp/test_sm70.cu -o /tmp/test_sm70 2>&1; then
    echo "✓ sm_70 is supported"
else
    echo "✗ sm_70 is NOT supported"
fi

echo ""
echo "Testing compute_70..."
if nvcc -arch=compute_70 /tmp/test_sm70.cu -o /tmp/test_compute70 2>&1; then
    echo "✓ compute_70 is supported"
else
    echo "✗ compute_70 is NOT supported"
fi

echo ""
echo "=== Available CUDA architectures in this nvcc ==="
nvcc --help | grep -A 20 "gpu-architecture" || echo "Could not find architecture help"

rm -f /tmp/test_sm70.cu /tmp/test_sm70 /tmp/test_compute70

echo ""
echo "=== Checking calibration tool compilation ==="
cd calibration
echo "Compiling stream_like..."
nvcc stream_like.cu -o /tmp/test_stream 2>&1 | head -20
echo ""
echo "Compiling gemm_cublas..."
nvcc gemm_cublas.cu -lcublas -o /tmp/test_gemm 2>&1 | head -20
cd ..
rm -f /tmp/test_stream /tmp/test_gemm

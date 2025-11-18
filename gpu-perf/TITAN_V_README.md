# NVIDIA TITAN V Setup Guide

## System Requirements

- NVIDIA TITAN V GPU (Volta architecture, compute capability 7.0)
- CUDA Toolkit 9.0 or later (for Volta support)
- Python 3.6+
- ~12 GB GPU memory

## Quick Start

### Option 1: Run Complete Pipeline
```bash
cd gpu-perf

# Use GPU 0 (default)
./scripts/run_titanv_full.sh

# Or select specific GPU if you have multiple
GPU_ID=0 ./scripts/run_titanv_full.sh
GPU_ID=1 ./scripts/run_titanv_full.sh
```

### Option 2: Step-by-Step

1. **Build calibration tools and measure GPU performance:**
```bash
cd gpu-perf
mkdir -p bin data
cd calibration

# Compile tools
nvcc -o ../bin/props_titanv props.cu
nvcc -o ../bin/stream_like_titanv stream_like.cu
nvcc -o ../bin/gemm_cublas_titanv gemm_cublas.cu -lcublas

cd ..

# Run calibration
bin/props_titanv > data/props_titanv.out
bin/stream_like_titanv > data/stream_like_titanv.out
bin/gemm_cublas_titanv > data/gemm_cublas_titanv.out
```

2. **Generate benchmark trials:**
```bash
./scripts/gen_trials_titanv.sh
```

3. **Build final dataset:**
```bash
python3 scripts/make_final_titanv.py
```

## Expected Performance

TITAN V typical calibration values:
- **Memory Bandwidth:** ~550-650 GB/s (theoretical: 653 GB/s)
- **Compute Throughput:** ~13,000-15,000 GFLOPS FP32 (theoretical: 15 TFLOPS)
- **SM Count:** 80
- **Memory:** 12 GB HBM2

## Troubleshooting

### Issue: "Unsupported gpu architecture 'sm_70'"

Your CUDA toolkit might not support Volta. The script will automatically try fallback options:
1. `-arch=sm_70` (standard)
2. `-arch=compute_70` (fallback)
3. `-gencode arch=compute_70,code=sm_70` (alternative)

If all fail, upgrade your CUDA toolkit:
```bash
nvcc --version  # Check current version
# You need CUDA 9.0+ for Volta support
```

### Issue: Suspicious calibration values

If you see extremely high or low values:
1. **Check which GPU is being used:**
   ```bash
   nvidia-smi
   ```

2. **Verify GPU selection:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 ./scripts/run_titanv_full.sh
   ```

3. **Check calibration output files:**
   ```bash
   cat data/stream_like_titanv.out
   cat data/gemm_cublas_titanv.out
   ```

### Issue: Compilation errors

Check the compilation log:
```bash
cat data/ptxas_titanv.log
```

Common fixes:
- Ensure CUDA toolkit is loaded: `module load cuda` (on HPC systems)
- Check nvcc version: `nvcc --version`
- Verify GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`

### Issue: Out of Memory (OOM)

The scripts use conservative memory limits (8M for XLarge tests) for 12GB VRAM. If you still encounter OOM:

1. Check current GPU memory usage:
   ```bash
   nvidia-smi
   ```

2. Ensure no other processes are using the GPU

3. Select an idle GPU if you have multiple:
   ```bash
   GPU_ID=1 ./scripts/run_titanv_full.sh
   ```

## Multiple GPU Setup

If you have 2 TITAN V GPUs (like shown in your nvidia-smi):

```bash
# Run on GPU 0
GPU_ID=0 ./scripts/run_titanv_full.sh &

# Run on GPU 1 with different output
GPU_ID=1 ./scripts/run_titanv_full.sh
```

## Profiling Individual Kernels

Profile a single kernel with NCU:
```bash
./scripts/profile_titanv.sh vector_add --N 1048576
./scripts/profile_titanv.sh matmul_tiled --matN 1024
```

Profile all kernels:
```bash
./scripts/profile_all_titanv.sh
```

## Output Files

After successful run:
- `data/props_titanv.out` - GPU properties
- `data/stream_like_titanv.out` - Memory bandwidth measurements
- `data/gemm_cublas_titanv.out` - Compute throughput measurements
- `data/device_calibration_titanv.json` - Calibration data (JSON)
- `data/trials_*__titanv.csv` - Individual trial files (~16 files)
- `data/runs_titanv_final.csv` - **Final dataset ready for analysis**

## Architecture Details

- **Architecture:** Volta
- **Compute Capability:** 7.0 (sm_70)
- **NVCC Flag:** `-arch=sm_70` or `-arch=compute_70`
- **Warp Size:** 32
- **Max Threads/SM:** 2048
- **Max Blocks/SM:** 32
- **Shared Memory/SM:** 96 KB
- **L2 Cache:** 4.5 MB

## Benchmarked Kernels

The suite includes 16 different kernel types:
- Vector operations (add, saxpy, strided_copy)
- Matrix operations (naive/tiled matmul, transpose)
- Reductions (sum, dot_product)
- Convolutions (3x3, 7x7)
- Memory patterns (random_access, divergent branches)
- Atomics (histogram, atomic_hotspot)
- Shared memory (bank conflicts)

Each kernel is tested with multiple problem sizes for comprehensive performance characterization.

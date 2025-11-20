# GPU Performance Data Collection Suite

A comprehensive benchmark suite for collecting GPU performance data across multiple NVIDIA GPU architectures. This repository contains CUDA kernels, calibration tools, and data collection pipelines to generate detailed performance datasets.

## Overview

This project systematically collects GPU performance metrics by running 16 different CUDA kernel benchmarks across multiple problem sizes on various NVIDIA GPUs. The generated datasets include execution times, hardware specifications, and performance characteristics.

## Folder Structure

```
gpu-perf/
├── bin/                    # Compiled CUDA executables
├── calibration/            # GPU calibration source code
│   ├── props.cu           # Device properties extraction
│   ├── stream_like.cu     # Memory bandwidth calibration
│   └── gemm_cublas.cu     # Compute throughput calibration
├── data/                   # Generated datasets and calibration outputs
├── kernels/                # CUDA kernel implementations (16 kernels)
├── runner/                 # Main benchmark orchestrator
│   └── main.cu            # Kernel dispatcher and timing harness
└── scripts/                # Data collection pipeline scripts
    ├── generate_dataset_full.sh   # Main pipeline orchestrator
    ├── gen_trials_2080ti.sh       # RTX 2080 Ti trial generation
    ├── gen_trials_4070.sh         # RTX 4070 trial generation
    ├── gen_trials_titanv.sh       # Titan V trial generation
    ├── gen_trials_titanx.sh       # Titan X trial generation
    ├── run_trials.sh              # Individual kernel execution
    └── build_final_dataset.py     # Data aggregation and processing
```

## Supported GPUs

- **RTX 2080 Ti** (Turing, sm_75)
- **RTX 4070** (Ada Lovelace, sm_89)
- **Titan V** (Volta, sm_70)
- **Titan X** (Turing, sm_75)

## Benchmark Kernels

### Memory-Bound Kernels
- **vector_add**: Simple element-wise addition (coalesced memory access)
- **saxpy**: Scaled vector addition (2 FLOPs per element)
- **strided_copy_8**: Strided memory access pattern (stride=8)
- **random_access**: Random gather operations
- **naive_transpose**: Matrix transpose with uncoalesced access
- **shared_transpose**: Optimized transpose using shared memory

### Compute-Bound Kernels
- **reduce_sum**: Parallel reduction with shared memory
- **dot_product**: Vector dot product with reduction
- **matmul_naive**: Basic matrix multiplication
- **matmul_tiled**: Tiled matrix multiplication with shared memory
- **conv2d_3x3**: 2D convolution with 3x3 kernel
- **conv2d_7x7**: 2D convolution with 7x7 kernel

### Specialized Kernels
- **histogram**: Atomic operations for histogram computation
- **atomic_hotspot**: Extreme atomic contention benchmark
- **vector_add_divergent**: Branch divergence test
- **shared_bank_conflict**: Shared memory bank conflict test

## Data Collection Pipeline

### Step 1: GPU Calibration
Before collecting data, calibrate the GPU to determine hardware limits:

```bash
cd gpu-perf/calibration
nvcc -O3 -o ../bin/props props.cu && ../bin/props > ../data/props_2080ti.out
nvcc -O3 -o ../bin/stream_like stream_like.cu && ../bin/stream_like > ../data/stream_like_2080ti.out
nvcc -O3 -lcublas -o ../bin/gemm_cublas gemm_cublas.cu && ../bin/gemm_cublas > ../data/gemm_cublas_2080ti.out
```

Calibration outputs:
- `props_*.out`: Device specifications (SM count, memory, cache sizes)
- `stream_like_*.out`: Sustained memory bandwidth (GB/s)
- `gemm_cublas_*.out`: Sustained compute throughput (GFLOPS)

### Step 2: Generate Dataset

**Option A: Automated Pipeline (RTX 2080 Ti Only)**

For RTX 2080 Ti, use the automated script:

```bash
cd gpu-perf
./scripts/generate_dataset_full.sh
```

This will automatically run both steps below and output `data/runs_2080ti_final.csv`.

**Option B: Manual Two-Step Process (All GPUs)**

For **any GPU** (RTX 2080 Ti, RTX 4070, Titan V, or Titan X), run the two-step process:

```bash
cd gpu-perf

# Step 2a: Generate trial data (choose your GPU)
./scripts/gen_trials_2080ti.sh   # For RTX 2080 Ti
# OR
./scripts/gen_trials_4070.sh     # For RTX 4070
# OR
./scripts/gen_trials_titanv.sh   # For Titan V
# OR
./scripts/gen_trials_titanx.sh   # For Titan X

# Step 2b: Process trials into final dataset
# Replace "2080ti" with your GPU: 4070, titanv, or titanx
python3 scripts/build_final_dataset.py \
  "data/trials_*__2080ti.csv" \
  data/props_2080ti.out \
  data/stream_like_2080ti.out \
  data/gemm_cublas_2080ti.out \
  data/runs_2080ti_final.csv
```

Both approaches will:
1. Build the CUDA runner binary for the specific GPU architecture
2. Execute 16 kernels × 4 problem sizes = ~65 configurations
3. Run 10 trials per configuration with warmup
4. Aggregate results and calculate all metrics
5. Output: `data/runs_<gpu>_final.csv`

**Important**: You can only generate data for **one GPU at a time** since the pipeline uses the GPU currently available in your system.

### Pipeline Details

**generate_dataset_full.sh** orchestrates:
```
gen_trials_2080ti.sh
├── Builds CUDA runner (nvcc with sm_75)
├── Runs run_trials.sh 65+ times (different kernels/sizes)
└── Outputs: data/trials_*__2080ti.csv

build_final_dataset.py
├── Aggregates all trial files
├── Calculates kernel metrics
├── Adds GPU hardware specs
└── Outputs: data/runs_2080ti_final.csv
```

## Collected Metrics

The final dataset contains 40 columns organized into 9 categories:

### 1. Kernel Identification (3 metrics)
- `kernel`: Kernel name
- `regs`: Registers per thread
- `shmem`: Shared memory per block (bytes)

### 2. Launch Configuration (2 metrics)
- `block`: Threads per block
- `grid_blocks`: Total blocks in grid

### 3. Performance Results (2 metrics)
- `mean_ms`: Mean execution time (milliseconds)
- `std_ms`: Standard deviation of execution time

### 4. Problem Sizes (7 metrics)
- `N`: Number of elements (1D kernels)
- `rows`: Matrix rows (2D kernels)
- `cols`: Matrix columns (2D kernels)
- `H`: Height (convolution kernels)
- `W`: Width (convolution kernels)
- `matN`: Matrix dimension (matmul kernels)
- `size_kind`: Size representation type ("N" or "rows_cols")

### 5. Kernel Metrics (8 metrics)
- `FLOPs`: Floating-point operations executed
- `BYTES`: Total bytes accessed (read + write)
- `shared_bytes`: Shared memory allocated
- `working_set_bytes`: Active memory footprint
- `arithmetic_intensity`: FLOPs/BYTES ratio
- `mem_pattern`: Memory access pattern (coalesced, strided, random, etc.)
- `has_branch_divergence`: Branch divergence flag (0 or 1)
- `atomic_ops_count`: Number of atomic operations

### 6. GPU Hardware (10 metrics)
- `gpu_device_name`: Full device name
- `gpu_architecture`: Architecture (Turing, Ampere, Ada Lovelace, etc.)
- `gpu_compute_capability`: Compute capability (e.g., "7.5")
- `gpu_sms`: Number of streaming multiprocessors
- `gpu_max_threads_per_sm`: Maximum threads per SM
- `gpu_max_blocks_per_sm`: Maximum blocks per SM
- `gpu_regs_per_sm`: Registers per SM
- `gpu_shared_mem_per_sm`: Shared memory per SM (bytes)
- `gpu_l2_bytes`: L2 cache size (bytes)
- `gpu_warp_size`: Warp size (typically 32)

### 7. GPU Performance Limits (4 metrics)
- `peak_theoretical_gflops`: Theoretical peak compute (GFLOPS)
- `peak_theoretical_bandwidth_gbps`: Theoretical peak bandwidth (GB/s)
- `calibrated_mem_bandwidth_gbps`: Measured sustained bandwidth (GB/s)
- `calibrated_compute_gflops`: Measured sustained compute (GFLOPS)

### 8. Achieved Performance (2 metrics)
- `achieved_bandwidth_gbps`: Actual bandwidth achieved (GB/s)
- `achieved_compute_gflops`: Actual compute throughput (GFLOPS)

### 9. Performance Models (2 metrics)
- `T1_model_ms`: Single-thread execution time estimate
- `speedup_model`: Parallel speedup (T1 / actual_time)

## Metric Definitions

### Arithmetic Intensity
Ratio of computation to memory traffic: `FLOPs / BYTES`
- Low AI (< 0.5): Memory-bound kernels
- High AI (> 5): Compute-bound kernels

### Memory Patterns
- `coalesced`: Sequential aligned memory access
- `stride_N`: Strided access with stride N
- `transpose_naive`: Uncoalesced transpose
- `transpose_tiled`: Shared memory optimized transpose
- `random_gather`: Random memory access
- `shared_reduction`: Reduction using shared memory
- `matmul_naive`: Basic matrix multiply access pattern
- `matmul_tiled`: Tiled shared memory matrix multiply
- `stencil_NxN`: Stencil operations (convolution)
- `atomics_global_256`: Global atomic operations (256 bins)
- `atomics_hotspot`: Single-location atomic contention
- `smem_bank_conflict`: Shared memory bank conflicts

### Calibrated vs Theoretical Performance
- **Theoretical**: Maximum hardware capability from specifications
- **Calibrated**: Sustained performance measured on actual hardware
- **Achieved**: Performance of specific kernel workload

Calibrated values are typically 70-90% of theoretical peak due to real-world memory controller limits, instruction scheduling overhead, and power/thermal constraints.

### Speedup Model
Estimated parallel speedup using roofline model:
- T1 = max(compute_time_single_thread, memory_time_single_thread)
- Speedup = T1 / measured_parallel_time
- Typical values: 2-30× depending on kernel characteristics

## Usage Example

Generate data for RTX 2080 Ti:
```bash
cd gpu-perf

# Calibrate GPU (one-time setup)
cd calibration
nvcc -O3 -o ../bin/props props.cu && ../bin/props > ../data/props_2080ti.out
nvcc -O3 -o ../bin/stream_like stream_like.cu && ../bin/stream_like > ../data/stream_like_2080ti.out
nvcc -O3 -lcublas -o ../bin/gemm_cublas gemm_cublas.cu && ../bin/gemm_cublas > ../data/gemm_cublas_2080ti.out
cd ..

# Run full pipeline
./scripts/generate_dataset_full.sh

# Output: data/runs_2080ti_final.csv (65+ rows × 40 columns)
```

The resulting CSV file can be used for performance analysis, modeling, or machine learning applications.

## Data Format

All output CSV files use standard format:
- Header row with column names
- One row per kernel configuration
- Numeric values with appropriate precision (6 decimals for times, 2 for throughput)
- Empty strings for inapplicable fields (e.g., N for 2D kernels)

Example row (abbreviated):
```
kernel,block,mean_ms,N,FLOPs,BYTES,arithmetic_intensity,achieved_bandwidth_gbps
vector_add,256,0.123456,1048576,1048576,12582912,0.083333,97.45
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)
- cuBLAS library
- Python 3.6+
- Bash shell

## Notes

- Each trial includes warmup iterations to ensure GPU is at stable clock speeds
- Multiple trials (10) per configuration provide statistical reliability
- Problem sizes chosen to span different cache/memory hierarchies
- Register and shared memory usage extracted from compiler output

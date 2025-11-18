# GPU Performance Benchmark - Complete Step-by-Step Guide

This guide provides complete instructions for running the GPU performance benchmark suite on any NVIDIA GPU, with specific examples for RTX 2080 Ti and RTX 4070.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start (5 minutes)](#quick-start)
3. [Detailed Step-by-Step Instructions](#detailed-step-by-step-instructions)
4. [Adapting to Different GPUs](#adapting-to-different-gpus)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **CUDA Toolkit:** 11.0 or higher
- **NVIDIA GPU:** Compute Capability 6.0+ (Pascal or newer)
- **GCC/G++:** Compatible with your CUDA version
- **Python 3:** 3.7 or higher
- **cuBLAS library:** Usually included with CUDA Toolkit

### Verify Installation
```bash
# Check CUDA installation
nvcc --version

# Check GPU
nvidia-smi

# Check Python
python3 --version
```

### Expected Output
```
nvcc: NVIDIA CUDA compiler driver
Cuda compilation tools, release 12.x

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx      Driver Version: 535.xxx    CUDA Version: 12.x     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce...  Off  | 00000000:01:00.0  On |                  N/A |
+-----------------------------------------------------------------------------+

Python 3.10.x
```

---

## Quick Start

### For RTX 2080 Ti (or any Turing GPU)

```bash
# Clone and navigate
cd gpu-perf

# Generate complete dataset in one command
./scripts/generate_dataset_full.sh
```

**Time:** ~10-20 minutes
**Output:** `data/runs_2080ti_final.csv` with ~16 rows × 40 columns

---

## Detailed Step-by-Step Instructions

### Step 1: Project Setup

```bash
# Navigate to the gpu-perf directory
cd /path/to/your/project/gpu-perf

# Verify directory structure
ls
```

**Expected output:**
```
kernels/  scripts/  data/  calibration/  CMakeLists.txt
```

### Step 2: Build the CUDA Runner

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Compile
make -j$(nproc)

# Verify executables were created
ls runner
```

**Expected output:**
```
runner  # Main benchmark executable
```

**Note:** If you get errors, see [Troubleshooting](#troubleshooting).

### Step 3: Run GPU Calibration Benchmarks

These benchmarks establish the performance baseline for your specific GPU.

```bash
cd ../calibration

# Build and run GPU properties capture
nvcc -o props props.cu
./props > ../data/props_2080ti.out

# Build and run memory bandwidth calibration
nvcc -o stream stream_like.cu
./stream > ../data/stream_like_2080ti.out

# Build and run compute calibration (requires cuBLAS)
nvcc -o gemm gemm_cublas.cu -lcublas
./gemm > ../data/gemm_cublas_2080ti.out

cd ..
```

**What these do:**
- **props:** Captures GPU specifications (SMs, registers, shared memory, etc.)
- **stream:** Measures sustained memory bandwidth using simple copy kernels
- **gemm:** Measures sustained compute throughput using optimized matrix multiply

**Time:** ~2-5 minutes total

### Step 4: Generate Benchmark Trial Data

```bash
# Run all 16 kernels with multiple problem sizes
./scripts/gen_trials_2080ti.sh
```

**What this does:**
- Runs each kernel with 4 different problem sizes (small, medium, large, xlarge)
- Each configuration: 10 trials with 50-100 repetitions
- Generates ~65 kernel configurations
- Outputs: `data/trials_*__2080ti.csv` (16 files, one per kernel)

**Time:** ~5-15 minutes depending on GPU

**Progress indicators:**
```
Running atomic_hotspot with N=262144...
Running atomic_hotspot with N=1048576...
Running vector_add with N=262144...
...
```

### Step 5: Process Trial Data into Final Dataset

```bash
# Aggregate trials and compute all metrics
python3 scripts/build_final_dataset.py \
    "data/trials_*__2080ti.csv" \
    data/props_2080ti.out \
    data/stream_like_2080ti.out \
    data/gemm_cublas_2080ti.out \
    data/runs_2080ti_final.csv
```

**What this does:**
- Aggregates trial data (computes mean, std)
- Calculates kernel metrics (FLOPs, BYTES, arithmetic intensity, etc.)
- Adds GPU hardware specs
- Computes achieved performance metrics
- Generates performance models (roofline, speedup)

**Output:** `data/runs_2080ti_final.csv`

**Time:** < 1 minute

### Step 6: Verify Output

```bash
# Check the final dataset
head -n 5 data/runs_2080ti_final.csv | cut -d',' -f1-10

# Count rows (should be ~16 kernels)
wc -l data/runs_2080ti_final.csv

# Check for any errors
grep -i error data/*.out
```

**Expected:**
- ~17 lines (1 header + 16 kernels)
- 40 columns
- No errors in output files

---

## Adapting to Different GPUs

### For RTX 4070 (Ada Lovelace Architecture)

The benchmark suite is GPU-agnostic, but you need to update specifications and filenames.

#### Step 1: Update GPU Specs in build_final_dataset.py

```bash
# Edit the GPU specs lookup table
nano scripts/build_final_dataset.py
```

**Add RTX 4070 specs to GPU_SPECS dictionary (around line 6):**

```python
GPU_SPECS = {
    "NVIDIA GeForce RTX 2080 Ti": {
        "architecture": "Turing",
        "peak_gflops_fp32": 13450,
        "peak_bandwidth_gbps": 616,
    },
    "NVIDIA GeForce RTX 3090": {
        "architecture": "Ampere",
        "peak_gflops_fp32": 35580,
        "peak_bandwidth_gbps": 936,
    },
    "NVIDIA GeForce RTX 4070": {
        "architecture": "Ada Lovelace",
        "peak_gflops_fp32": 29150,  # 46 SMs × 128 cores × 2.475 GHz × 2 ops
        "peak_bandwidth_gbps": 504,  # 192-bit × 21 Gbps / 8
    },
}
```

**RTX 4070 Specifications:**
- **SMs:** 46
- **CUDA Cores:** 5,888 (128 per SM)
- **Boost Clock:** 2.475 GHz
- **Memory:** 12 GB GDDR6X
- **Memory Bus:** 192-bit
- **Memory Clock:** 21 Gbps
- **Compute Capability:** 8.9

#### Step 2: Update Architecture Mapping (if needed)

The architecture map should already include Ada Lovelace:

```python
ARCHITECTURE_MAP = {
    (6, 0): "Pascal",
    (6, 1): "Pascal",
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",  # RTX 4070, 4080, 4090
    (9, 0): "Hopper",
}
```

#### Step 3: Update Filenames for RTX 4070

```bash
# Use a different tag for RTX 4070
export GPU_TAG="4070"
```

**OR** manually update scripts:

In `scripts/gen_trials_2080ti.sh`, change:
```bash
TAG="2080ti"
```
to:
```bash
TAG="4070"
```

#### Step 4: Run the Full Pipeline

```bash
# Run calibration
cd calibration
nvcc -o props props.cu && ./props > ../data/props_4070.out
nvcc -o stream stream_like.cu && ./stream > ../data/stream_like_4070.out
nvcc -o gemm gemm_cublas.cu -lcublas && ./gemm > ../data/gemm_cublas_4070.out
cd ..

# Generate trials (update script first with TAG="4070")
./scripts/gen_trials_2080ti.sh  # Will output trials_*__4070.csv files

# Process into final dataset
python3 scripts/build_final_dataset.py \
    "data/trials_*__4070.csv" \
    data/props_4070.out \
    data/stream_like_4070.out \
    data/gemm_cublas_4070.out \
    data/runs_4070_final.csv
```

**Output:** `data/runs_4070_final.csv`

---

### For Other GPUs

#### Supported Architectures
- **Pascal (sm_60, sm_61):** GTX 1080 Ti, Titan X Pascal
- **Volta (sm_70):** Titan V, Tesla V100
- **Turing (sm_75):** RTX 2060, 2070, 2080, 2080 Ti
- **Ampere (sm_80, sm_86):** RTX 3060, 3070, 3080, 3090, A100
- **Ada Lovelace (sm_89):** RTX 4060, 4070, 4080, 4090
- **Hopper (sm_90):** H100

#### Generic Steps

1. **Find GPU specifications:**
   - Visit https://www.techpowerup.com/gpu-specs/
   - Search for your GPU model
   - Note: CUDA cores, boost clock, memory bandwidth, bus width

2. **Calculate peak GFLOPS:**
   ```
   Peak FP32 GFLOPS = SMs × (Cores/SM) × Boost_Clock_GHz × 2
   ```

   Example for RTX 4070:
   ```
   = 46 SMs × 128 cores/SM × 2.475 GHz × 2 FLOPs/cycle
   = 29,150 GFLOPS
   ```

3. **Calculate peak bandwidth:**
   ```
   Peak Bandwidth (GB/s) = (Bus_Width_bits × Memory_Clock_Gbps) / 8
   ```

   Example for RTX 4070:
   ```
   = (192 bits × 21 Gbps) / 8
   = 504 GB/s
   ```

4. **Update GPU_SPECS in build_final_dataset.py:**
   ```python
   "NVIDIA GeForce RTX XXXX": {
       "architecture": "YourArchitecture",
       "peak_gflops_fp32": CALCULATED_GFLOPS,
       "peak_bandwidth_gbps": CALCULATED_BW,
   },
   ```

5. **Run the pipeline with appropriate tag:**
   - Update TAG in scripts or use environment variable
   - Follow steps 3-6 from detailed instructions

---

## Understanding the Output

### Output File: `runs_2080ti_final.csv`

**Format:** CSV with 40 columns

**Sample row (truncated):**
```csv
kernel,regs,shmem,block,grid_blocks,mean_ms,std_ms,...
vector_add,12,0,256,4096,0.025821,0.001111,...
```

### Key Metrics to Look For

#### 1. Execution Time
- **Column:** `mean_ms`
- **Lower is better**
- Example: `vector_add` = 0.026 ms (very fast)

#### 2. Memory Efficiency
- **Column:** `achieved_bandwidth_gbps`
- **Compare to:** `calibrated_mem_bandwidth_gbps` (541 GB/s for RTX 2080 Ti)
- **Good:** > 80% of calibrated
- Example: `vector_add` achieves 487 GB/s (90% efficiency)

#### 3. Compute Efficiency
- **Column:** `achieved_compute_gflops`
- **Compare to:** `calibrated_compute_gflops` (11,377 GFLOPS for RTX 2080 Ti)
- **Good:** > 70% for compute-bound kernels
- Example: `matmul_tiled` achieves 1,439 GFLOPS (limited by small matrix size)

#### 4. Performance Pattern
- **Column:** `mem_pattern`
- **Best:** `coalesced` (90% bandwidth efficiency)
- **Worst:** `stride_8`, `random_gather` (<10% efficiency)

#### 5. Speedup from Parallelism
- **Column:** `speedup_model`
- **Range:** 2-30× typical
- **High speedup:** Memory-bound kernels with good parallelism
- **Low speedup:** Kernels with contention or poor memory patterns

### Full Column Reference

See [METRICS_CHECKLIST.md](METRICS_CHECKLIST.md) for detailed explanation of all 40 columns.

---

## Troubleshooting

### Build Errors

#### "nvcc: command not found"
**Problem:** CUDA Toolkit not in PATH

**Solution:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Add to `~/.bashrc` for persistence.

#### "cannot find -lcublas"
**Problem:** cuBLAS library not found

**Solution:**
```bash
# Find cuBLAS location
find /usr -name "libcublas.so" 2>/dev/null

# Add to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or install CUDA libraries
sudo apt-get install cuda-cublas-dev-12-x  # Replace 12-x with your version
```

#### "unsupported GPU architecture"
**Problem:** GPU too old (compute capability < 6.0)

**Solution:**
Update CMakeLists.txt to include your architecture:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89)  # Add your sm_XX
```

### Runtime Errors

#### "CUDA error: out of memory"
**Problem:** GPU memory exhausted by large problem sizes

**Solution:**
Reduce problem sizes in `gen_trials_2080ti.sh`:
```bash
# Change xlarge sizes
SIZES_1D="262144 1048576 4194304"  # Remove 16777216
SIZES_2D="512 1024 2048"           # Remove 4096
```

#### "No trials found"
**Problem:** Trial generation didn't complete or files in wrong location

**Solution:**
```bash
# Check for trial files
ls data/trials_*

# Re-run trial generation
./scripts/gen_trials_2080ti.sh

# Verify runner built correctly
./build/runner --help
```

#### "Thermal throttling detected"
**Problem:** GPU overheating during benchmarks

**Solution:**
```bash
# Monitor temperature
nvidia-smi -l 1  # Updates every second

# Improve cooling or reduce load
# Add sleep between trials in gen_trials_2080ti.sh:
sleep 0.5  # After each kernel run
```

### Data Processing Errors

#### "KeyError: 'NVIDIA GeForce RTX XXXX'"
**Problem:** GPU not in GPU_SPECS lookup table

**Solution:**
Add your GPU to `scripts/build_final_dataset.py` as shown in [Adapting to Different GPUs](#adapting-to-different-gpus).

#### "Missing calibration file"
**Problem:** Calibration benchmarks didn't run successfully

**Solution:**
```bash
# Check for output files
ls data/*.out

# Re-run calibration
cd calibration
./run_all_calibration.sh  # Or run individually

# Check for errors
cat ../data/props_2080ti.out
```

---

## Advanced Usage

### Running Single Kernels

```bash
# Test one kernel manually
./build/runner vector_add --rows 1048576 --cols 1 --block 256 --warmup 10 --reps 50

# Output: CSV line with timing data
```

### Customizing Problem Sizes

Edit `scripts/gen_trials_2080ti.sh`:

```bash
# Add custom sizes
SIZES_1D="100000 500000 1000000 5000000"
SIZES_2D="256 512 1024 2048 4096"
SIZES_MATMUL="128 256 512 1024"
```

### Analyzing Results with Python

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data/runs_2080ti_final.csv')

# Find best performing kernels
df.nsmallest(5, 'mean_ms')[['kernel', 'mean_ms', 'achieved_bandwidth_gbps']]

# Compare memory patterns
df.groupby('mem_pattern')['achieved_bandwidth_gbps'].mean()

# Compute efficiency
df['bw_efficiency'] = df['achieved_bandwidth_gbps'] / df['calibrated_mem_bandwidth_gbps']
df['compute_efficiency'] = df['achieved_compute_gflops'] / df['calibrated_compute_gflops']
```

---

## Performance Expectations

### RTX 2080 Ti (Turing)
- **Peak Theoretical:** 13,450 GFLOPS, 616 GB/s
- **Calibrated Sustained:** ~11,400 GFLOPS, ~540 GB/s
- **Best Kernels:** 80-90% of calibrated bandwidth

### RTX 4070 (Ada Lovelace)
- **Peak Theoretical:** 29,150 GFLOPS, 504 GB/s
- **Expected Sustained:** ~24,000 GFLOPS, ~430 GB/s
- **Improvements over Turing:** 2.1× compute, similar bandwidth

### RTX 4090 (Ada Lovelace)
- **Peak Theoretical:** 82,580 GFLOPS, 1,008 GB/s
- **Expected Sustained:** ~70,000 GFLOPS, ~900 GB/s
- **Best for:** Large compute-bound workloads

---

## Summary Checklist

- [ ] CUDA Toolkit installed and in PATH
- [ ] Project built successfully (`make` in build/)
- [ ] GPU calibration completed (3 `.out` files in `data/`)
- [ ] Trial data generated (16 `trials_*.csv` files in `data/`)
- [ ] GPU specs added to `build_final_dataset.py`
- [ ] Final dataset created (`runs_<gpu>_final.csv`)
- [ ] Output verified (16 rows, 40 columns, no errors)

**Total Time:** 15-30 minutes for complete pipeline

**Questions?** See [METRICS_CHECKLIST.md](METRICS_CHECKLIST.md) for metric definitions or check the troubleshooting section above.

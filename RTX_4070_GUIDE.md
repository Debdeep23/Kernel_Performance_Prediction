# RTX 4070 Benchmark Guide

Complete guide for running GPU performance benchmarks on RTX 4070 (Ada Lovelace architecture).

---

## Quick Start (One Command)

If you just want to run everything end-to-end:

```bash
cd gpu-perf
./scripts/run_4070_full.sh
```

**Time:** ~15-25 minutes
**Output:** `data/runs_4070_final.csv` with all benchmark results

---

## What Gets Created

All files for 4070 are completely separate from 2080ti:

### Binary Files
- `bin/props_4070` - GPU properties capture tool
- `bin/stream_like_4070` - Memory bandwidth benchmark
- `bin/gemm_cublas_4070` - Compute throughput benchmark
- `bin/runner` - Main kernel benchmark runner (compiled for sm_89)

### Calibration Files
- `data/props_4070.out` - GPU hardware specifications
- `data/stream_like_4070.out` - Memory bandwidth measurements
- `data/gemm_cublas_4070.out` - Compute throughput measurements
- `data/device_calibration_4070.json` - JSON summary of GPU capabilities

### Trial Files (16 files, one per kernel)
- `data/trials_vector_add__4070.csv`
- `data/trials_saxpy__4070.csv`
- `data/trials_strided_copy_8__4070.csv`
- `data/trials_naive_transpose__4070.csv`
- `data/trials_shared_transpose__4070.csv`
- `data/trials_matmul_naive__4070.csv`
- `data/trials_matmul_tiled__4070.csv`
- `data/trials_reduce_sum__4070.csv`
- `data/trials_dot_product__4070.csv`
- `data/trials_histogram__4070.csv`
- `data/trials_conv2d_3x3__4070.csv`
- `data/trials_conv2d_7x7__4070.csv`
- `data/trials_random_access__4070.csv`
- `data/trials_vector_add_divergent__4070.csv`
- `data/trials_shared_bank_conflict__4070.csv`
- `data/trials_atomic_hotspot__4070.csv`

### Processed Data Files
- `data/runs_4070.csv` - Aggregated trials (raw)
- `data/runs_4070_with_counts.csv` - With static kernel metrics
- `data/runs_4070_enriched.csv` - With GPU metrics
- `data/runs_4070_final.csv` - **Final dataset with all metrics**

### Build Log
- `data/ptxas_4070.log` - CUDA compiler output for diagnostics

---

## Step-by-Step Manual Execution

If you want to run steps individually for debugging or understanding:

### Step 1: Build Calibration Tools

```bash
cd gpu-perf/calibration

# Build GPU properties tool
nvcc -o ../bin/props_4070 props.cu

# Build memory bandwidth benchmark
nvcc -o ../bin/stream_like_4070 stream_like.cu

# Build compute throughput benchmark
nvcc -o ../bin/gemm_cublas_4070 gemm_cublas.cu -lcublas

cd ..
```

### Step 2: Run Calibration Benchmarks

```bash
# Capture GPU specifications
bin/props_4070 > data/props_4070.out

# Measure sustained memory bandwidth
bin/stream_like_4070 > data/stream_like_4070.out

# Measure sustained compute throughput
bin/gemm_cublas_4070 > data/gemm_cublas_4070.out
```

**Verify outputs:**
```bash
# Check props
cat data/props_4070.out

# Check bandwidth (should see ~500 GB/s for RTX 4070)
tail data/stream_like_4070.out

# Check compute (should see ~20-30 TFLOPS for RTX 4070)
tail data/gemm_cublas_4070.out
```

### Step 3: Create Device Calibration JSON

```bash
python3 - <<'PYEOF'
import re, json

# Read props
props = {}
with open('data/props_4070.out') as f:
    for ln in f:
        if '=' in ln:
            k, v = ln.strip().split('=', 1)
            props[k] = v

# Read stream
stream_bw = 0.0
with open('data/stream_like_4070.out') as f:
    for ln in f:
        if ln.startswith('SUSTAINED_MEM_BW_GBPS='):
            stream_bw = float(ln.split('=')[1])

# Read gemm
gemm_gflops = 0.0
with open('data/gemm_cublas_4070.out') as f:
    for ln in f:
        if ln.startswith('SUSTAINED_COMPUTE_GFLOPS='):
            gemm_gflops = float(ln.split('=')[1])

# Create calibration JSON
calib = {
    "device_name": props.get("name", "RTX 4070"),
    "warp_size": int(props.get("warpSize", "32")),
    "sm_count": int(props.get("multiProcessorCount", "46")),
    "sustained_mem_bandwidth_gbps": stream_bw,
    "sustained_compute_gflops": gemm_gflops,
    "sm_limits": {
        "max_threads_per_sm": int(props.get("maxThreadsPerMultiProcessor", "1536")),
        "max_blocks_per_sm": int(props.get("maxBlocksPerMultiProcessor", "24")),
        "registers_per_sm": int(props.get("regsPerMultiprocessor", "65536")),
        "shared_mem_per_sm_bytes": int(props.get("sharedMemPerMultiprocessor", "102400"))
    }
}

with open('data/device_calibration_4070.json', 'w') as f:
    json.dump(calib, f, indent=2)

print("Created data/device_calibration_4070.json")
PYEOF
```

### Step 4: Generate Trial Data

```bash
# Run all kernels with multiple sizes (this takes the longest)
./scripts/gen_trials_4070.sh
```

**What this does:**
- Builds runner for sm_89 (Ada Lovelace architecture)
- Runs 16 kernels with 4 different problem sizes each
- Each configuration: 10 trials
- Total: ~70 benchmark configurations

**Time:** ~10-20 minutes

**Monitor progress:**
```bash
# In another terminal, watch files being created
watch -n 1 'ls -lh data/trials_*__4070.csv | wc -l'
```

### Step 5: Validate Trial Data

```bash
for f in data/trials_*__4070.csv; do
  python3 scripts/validate_csv.py "$f"
done
```

### Step 6: Aggregate Trials

```bash
python3 scripts/aggregate_trials.py data/trials_*__4070.csv > data/runs_4070.csv
```

**What this does:**
- Groups trials by kernel and configuration
- Calculates mean and standard deviation
- Produces one row per kernel configuration

### Step 7: Add Static Kernel Metrics

```bash
python3 scripts/static_counts.py data/runs_4070.csv data/runs_4070_with_counts.csv
```

**Adds:**
- FLOPs (floating point operations)
- BYTES (global memory traffic)
- Arithmetic intensity (FLOPs/BYTES)
- Working set size
- Memory access pattern classification

### Step 8: Enrich with GPU Metrics

```bash
python3 scripts/enrich_with_gpu_metrics.py \
  data/runs_4070_with_counts.csv \
  data/props_4070.out \
  data/stream_like_4070.out \
  data/gemm_cublas_4070.out \
  data/runs_4070_enriched.csv
```

**Adds:**
- GPU hardware specifications
- Calibrated bandwidth and compute throughput
- Achieved bandwidth and compute rates

### Step 9: Add Performance Models

```bash
python3 scripts/add_singlethread_baseline.py \
  data/runs_4070_enriched.csv \
  data/device_calibration_4070.json \
  data/runs_4070_final.csv \
  32
```

**Adds:**
- Single-thread roofline model prediction (T1_model_ms)
- Speedup vs single-thread baseline (speedup_model)

### Step 10: Inspect Final Output

```bash
# View first few rows
head -5 data/runs_4070_final.csv

# Count rows and columns
wc -l data/runs_4070_final.csv
head -1 data/runs_4070_final.csv | tr ',' '\n' | wc -l

# View specific kernel
grep "vector_add" data/runs_4070_final.csv | head -3
```

---

## Expected Results

### RTX 4070 Specifications
- **Architecture:** Ada Lovelace (sm_89)
- **SMs:** 46
- **CUDA Cores:** 5,888 (128 per SM)
- **Tensor Cores:** 184 (4th gen)
- **RT Cores:** 46 (3rd gen)
- **Base Clock:** ~2.0 GHz
- **Boost Clock:** ~2.5 GHz
- **Memory:** 12 GB GDDR6X
- **Memory Bus:** 192-bit
- **Memory Bandwidth:** ~504 GB/s (theoretical)
- **FP32 Performance:** ~29 TFLOPS (theoretical)

### Expected Measured Performance
- **Sustained Memory Bandwidth:** ~480-500 GB/s
- **Sustained Compute (FP32):** ~20-28 TFLOPS
- **L2 Cache:** 36 MB

### Final CSV Structure

**~70 rows** (header + kernels × sizes)
**~37 columns** organized as:

1. **Kernel Info** (3 cols): kernel name, register usage, shared memory
2. **Launch Config** (2 cols): block, grid_blocks
3. **Performance** (2 cols): mean_ms, std_ms
4. **Problem Size** (3 cols): N, rows, cols
5. **Kernel Metrics** (6 cols): FLOPs, BYTES, arithmetic_intensity, etc.
6. **GPU Hardware** (9 cols): SM count, compute capability, registers, etc.
7. **GPU Performance** (6 cols): calibrated bandwidth & compute
8. **Achieved Perf** (4 cols): achieved bandwidth & compute
9. **Models** (2 cols): T1_model_ms, speedup_model

---

## Comparing 2080Ti vs 4070

You now have two completely separate datasets:
- `data/runs_2080ti_final.csv`
- `data/runs_4070_final.csv`

### Quick Comparison

```bash
# Compare memory bandwidth
grep "SUSTAINED_MEM_BW" data/stream_like_2080ti.out
grep "SUSTAINED_MEM_BW" data/stream_like_4070.out

# Compare compute throughput
grep "SUSTAINED_COMPUTE" data/gemm_cublas_2080ti.out
grep "SUSTAINED_COMPUTE" data/gemm_cublas_4070.out

# Compare specific kernel (e.g., matmul_tiled)
grep "matmul_tiled" data/runs_2080ti_final.csv | cut -d',' -f1,15
grep "matmul_tiled" data/runs_4070_final.csv | cut -d',' -f1,15
```

### Analysis Script Example

```python
import pandas as pd

# Load both datasets
df_2080 = pd.read_csv('data/runs_2080ti_final.csv')
df_4070 = pd.read_csv('data/runs_4070_final.csv')

# Compare by kernel
comparison = pd.merge(
    df_2080[['kernel', 'N', 'mean_ms']],
    df_4070[['kernel', 'N', 'mean_ms']],
    on=['kernel', 'N'],
    suffixes=('_2080ti', '_4070')
)

comparison['speedup'] = comparison['mean_ms_2080ti'] / comparison['mean_ms_4070']
print(comparison.sort_values('speedup', ascending=False))
```

---

## Troubleshooting

### Issue: nvcc not found
```bash
# Check CUDA installation
which nvcc
nvcc --version

# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### Issue: sm_89 not supported
Your CUDA toolkit might be too old. RTX 4070 requires CUDA 11.8+.

```bash
# Check CUDA version
nvcc --version

# Update if needed (requires sudo)
# Or use -arch=sm_86 as fallback (Ampere)
```

### Issue: Out of memory during calibration
Reduce problem sizes in calibration benchmarks.

### Issue: Calibration shows unexpected values
- **Low bandwidth (<400 GB/s):** Check for background GPU usage
- **Low compute (<15 TFLOPS):** Ensure GPU isn't throttling
- **Run nvidia-smi to check utilization and temperature**

```bash
nvidia-smi dmon -s pucvmet
```

### Issue: Trials taking too long
Reduce the number of trials in `gen_trials_4070.sh`:
- Change last parameter from `10` to `5` or `3`
- This reduces statistical confidence but speeds up testing

---

## File Organization Summary

```
gpu-perf/
├── bin/
│   ├── runner               (compiled for current GPU)
│   ├── props_4070
│   ├── stream_like_4070
│   └── gemm_cublas_4070
├── data/
│   ├── props_4070.out
│   ├── stream_like_4070.out
│   ├── gemm_cublas_4070.out
│   ├── device_calibration_4070.json
│   ├── ptxas_4070.log
│   ├── trials_*__4070.csv   (16 files)
│   ├── runs_4070.csv
│   ├── runs_4070_with_counts.csv
│   ├── runs_4070_enriched.csv
│   └── runs_4070_final.csv  ← FINAL OUTPUT
└── scripts/
    ├── gen_trials_4070.sh
    └── run_4070_full.sh     ← ONE-COMMAND RUNNER

ALL 2080Ti files remain separate and unchanged!
```

---

## Next Steps

1. **Analyze Results:** Use pandas/numpy to analyze `runs_4070_final.csv`
2. **Compare GPUs:** Cross-reference with `runs_2080ti_final.csv`
3. **Visualize:** Create roofline plots, speedup charts, etc.
4. **Extend:** Add more kernels or problem sizes by modifying `gen_trials_4070.sh`

---

## Support

For issues or questions:
- Check build logs in `data/ptxas_4070.log`
- Validate CSVs with `scripts/validate_csv.py`
- Review this guide and STEP_BY_STEP_GUIDE.md

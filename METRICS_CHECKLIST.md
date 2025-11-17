# Core Metrics Checklist - COMPLETE! ✅

## Kernel Metrics (11/11) ✅

| # | Metric | Column Name | Source | Status |
|---|--------|-------------|--------|--------|
| 1 | Total FLOPs | `FLOPs` | Calculated from kernel logic | ✅ |
| 2 | Total bytes | `BYTES` | Calculated from memory accesses | ✅ |
| 3 | Arithmetic intensity | `arithmetic_intensity` | FLOPs / BYTES | ✅ |
| 4 | Memory access pattern | `mem_pattern` | Kernel classification | ✅ |
| 5 | Working set size | `working_set_bytes` | Estimated from problem size | ✅ |
| 6 | Threads per block | `block` | Launch configuration | ✅ |
| 7 | Grid size | `grid_blocks` | Launch configuration | ✅ |
| 8 | Registers per thread | `regs` | Kernel compilation info | ✅ |
| 9 | Shared memory per block | `shmem` | Kernel launch param | ✅ |
| 10 | Branch divergence | `has_branch_divergence` | **NEW** - Kernel classification | ✅ |
| 11 | Atomic ops | `atomic_ops_count` | **NEW** - Calculated from logic | ✅ |

## GPU Metrics (13/13) ✅

| # | Metric | Column Name | Source | Status |
|---|--------|-------------|--------|--------|
| 1 | Peak FP32 GFLOPS | `peak_theoretical_gflops` | **NEW** - GPU spec lookup | ✅ |
| 2 | Peak bandwidth | `peak_theoretical_bandwidth_gbps` | **NEW** - GPU spec lookup | ✅ |
| 3 | Sustained bandwidth (calibrated) | `calibrated_mem_bandwidth_gbps` | Measured calibration | ✅ |
| 4 | Sustained compute (calibrated) | `calibrated_compute_gflops` | Measured calibration | ✅ |
| 5 | Number of SMs | `gpu_sms` | Device properties | ✅ |
| 6 | Max threads/SM | `gpu_max_threads_per_sm` | Device properties | ✅ |
| 7 | Max blocks/SM | `gpu_max_blocks_per_sm` | Device properties | ✅ |
| 8 | Registers/SM | `gpu_regs_per_sm` | Device properties | ✅ |
| 9 | Shared memory/SM | `gpu_shared_mem_per_sm` | Device properties | ✅ |
| 10 | L2 cache size | `gpu_l2_bytes` | Device properties | ✅ |
| 11 | Architecture name | `gpu_architecture` | **NEW** - Compute capability mapping | ✅ |
| 12 | Compute capability | `gpu_compute_capability` | **NEW** - Combined format "7.5" | ✅ |
| 13 | Warp size | `gpu_warp_size` | Device properties | ✅ |

---

## Summary

**Total Core Metrics: 24/24 ✅ COMPLETE!**

- ✅ Kernel Metrics: 11/11
- ✅ GPU Metrics: 13/13

---

## New Columns Added (6 total)

1. **`has_branch_divergence`** - Binary flag (0/1) indicating kernels with control flow divergence
   - Value: 1 for `vector_add_divergent`, 0 for others

2. **`atomic_ops_count`** - Number of atomic operations
   - `atomic_hotspot`: N × iters
   - `histogram`: N
   - Others: 0

3. **`gpu_architecture`** - Architecture name (Turing, Ampere, Ada, Hopper, etc.)
   - Mapped from compute capability via `ARCHITECTURE_MAP`

4. **`gpu_compute_capability`** - Full format "X.Y" (e.g., "7.5")
   - Replaces split `gpu_cc_major` and `gpu_cc_minor`

5. **`peak_theoretical_gflops`** - Theoretical peak FP32 performance
   - RTX 2080 Ti: 13,450 GFLOPS
   - Looked up from `GPU_SPECS` table

6. **`peak_theoretical_bandwidth_gbps`** - Theoretical peak memory bandwidth
   - RTX 2080 Ti: 616 GB/s
   - Looked up from `GPU_SPECS` table

---

## Column Removed (1 total)

1. **`gpu_cc_major`** - Replaced by combined `gpu_compute_capability`

---

## Final Dataset Structure

**Total columns: 40** (was 36, added 6, removed 1, net +5)

### Column Organization:

1. **Kernel Identification** (3): kernel, regs, shmem
2. **Launch Configuration** (2): block, grid_blocks
3. **Performance Results** (2): mean_ms, std_ms
4. **Problem Sizes** (7): N, rows, cols, H, W, matN, size_kind
5. **Kernel Metrics** (8): FLOPs, BYTES, shared_bytes, working_set_bytes, arithmetic_intensity, mem_pattern, has_branch_divergence, atomic_ops_count
6. **GPU Hardware Specs** (10): gpu_device_name, gpu_architecture, gpu_compute_capability, gpu_sms, gpu_max_threads_per_sm, gpu_max_blocks_per_sm, gpu_regs_per_sm, gpu_shared_mem_per_sm, gpu_l2_bytes, gpu_warp_size
7. **GPU Performance Limits** (4): peak_theoretical_gflops, peak_theoretical_bandwidth_gbps, calibrated_mem_bandwidth_gbps, calibrated_compute_gflops
8. **Achieved Performance** (2): achieved_bandwidth_gbps, achieved_compute_gflops
9. **Performance Models** (2): T1_model_ms, speedup_model

---

## Implementation Details

### GPU Specs Lookup Table
```python
GPU_SPECS = {
    "NVIDIA GeForce RTX 2080 Ti": {
        "architecture": "Turing",
        "peak_gflops_fp32": 13450,
        "peak_bandwidth_gbps": 616,
    },
    # Additional GPUs: RTX 3090, RTX 3080, RTX 4090
}
```

### Architecture Mapping
```python
ARCHITECTURE_MAP = {
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
}
```

### Branch Divergence Detection
```python
DIVERGENT_KERNELS = {"vector_add_divergent"}
has_divergence = 1 if kernel in DIVERGENT_KERNELS else 0
```

### Atomic Operations Calculation
```python
if kernel == "atomic_hotspot":
    atomic_ops = N * iters
elif kernel == "histogram":
    atomic_ops = N
else:
    atomic_ops = 0
```

---

## Next Steps

To regenerate the dataset with all metrics:
```bash
cd gpu-perf
./scripts/gen_trials_2080ti.sh  # Generate trials with multiple sizes
# Then use your existing pipeline to build final CSV
```

The new dataset will have:
- ✅ **11/11 kernel metrics**
- ✅ **13/13 GPU metrics**
- ✅ **~65 rows** (multiple sizes per kernel)
- ✅ **40 columns** (clean, organized, complete)

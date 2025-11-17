# Recommended Dataset Improvements

## Current Status: What You Have ✅

### Kernel Metrics (9/11 core metrics captured):
- ✅ Total FLOPs (`FLOPs`)
- ✅ Total bytes (`BYTES`)
- ✅ Arithmetic intensity (`arithmetic_intensity`)
- ✅ Memory access pattern (`mem_pattern`)
- ✅ Working set size (`working_set_bytes`)
- ✅ Threads per block (`block`)
- ✅ Grid size (`grid_blocks`)
- ✅ Registers per thread (`regs`)
- ✅ Shared memory per block (`shmem`)
- ❌ Branch divergence (optional) - NOT CAPTURED
- ❌ Atomic ops (optional) - NOT CAPTURED

### GPU Metrics (11/13 core metrics captured):
- ❌ Peak theoretical FP32 GFLOPS - MISSING
- ❌ Peak theoretical bandwidth - MISSING
- ✅ Sustained bandwidth (`calibrated_mem_bandwidth_gbps`)
- ✅ Sustained compute (`calibrated_compute_gflops`)
- ✅ Number of SMs (`gpu_sms`)
- ✅ Max threads/SM (`gpu_max_threads_per_sm`)
- ✅ Max blocks/SM (`gpu_max_blocks_per_sm`)
- ✅ Registers/SM (`gpu_regs_per_sm`)
- ✅ Shared memory/SM (`gpu_shared_mem_per_sm`)
- ✅ L2 cache size (`gpu_l2_bytes`)
- ⚠️ Architecture name - PARTIAL (have device name, not architecture)
- ⚠️ Compute capability - SPLIT (only major, no minor)
- ✅ Warp size (`gpu_warp_size`)

---

## Recommended Changes

### 1. Add Missing GPU Metrics

**Add these columns:**
- `peak_theoretical_gflops` - Theoretical FP32 peak (13,450 for RTX 2080 Ti)
- `peak_theoretical_bandwidth_gbps` - Theoretical bandwidth peak (616 GB/s for RTX 2080 Ti)
- `gpu_architecture` - Architecture name (Turing, Ampere, Ada, etc.)
- `gpu_compute_capability` - Combined format "7.5" instead of split major/minor

**Why:** These are in your core metrics list and help understand achieved vs theoretical efficiency.

**How to calculate:**
- RTX 2080 Ti (sm_75): 68 SMs × 64 FP32 cores/SM × 1.545 GHz × 2 ops/cycle = 13,450 GFLOPS
- RTX 2080 Ti bandwidth: 352-bit × 14 Gbps GDDR6 / 8 = 616 GB/s

---

### 2. Simplify Size Representation

**Problem:** You have 6 size columns (N, rows, cols, H, W, matN) but only 1 is used per kernel.

**Current columns:**
```
N, rows, cols, H, W, matN, size_kind
```

**Proposed change - Option A (Simple):**
```
problem_size      # The actual size value used (e.g., 1048576)
size_unit         # What it represents (elements, matrix_dim, image_dim)
```

**Proposed change - Option B (Keep compatibility):**
Keep current but add a unified `problem_size` column that always contains the relevant size:
- For 1D kernels: problem_size = N
- For 2D kernels: problem_size = rows × cols
- For matmul: problem_size = matN

---

### 3. Add Branch Divergence and Atomic Ops Flags

**Add these columns:**
- `has_branch_divergence` - Boolean (1/0) indicating if kernel has control flow divergence
- `atomic_ops_count` - Estimated number of atomic operations (0 if none)

**Kernel mapping:**
```python
# In build_final_dataset.py, add to static_counts():

divergence_kernels = {"vector_add_divergent"}
atomic_kernels = {"histogram", "atomic_hotspot"}

has_divergence = 1 if kernel in divergence_kernels else 0

if kernel == "atomic_hotspot":
    atomic_ops = N * iters
elif kernel == "histogram":
    atomic_ops = N
else:
    atomic_ops = 0
```

---

### 4. Clarify shmem vs shared_bytes

**Current confusion:**
- `shmem` - Dynamic shared memory per block (from kernel launch)
- `shared_bytes` - Estimated total shared memory traffic

**Recommendation:**
Rename for clarity:
- `shmem` → `shmem_per_block_bytes` (or keep as is)
- `shared_bytes` → `shmem_total_estimate_bytes`

Or just drop `shared_bytes` if it's not adding value.

---

### 5. Add Occupancy Metrics

**Add columns:**
- `theoretical_occupancy` - Max warps per SM given resource constraints
- `achieved_occupancy` - Actual occupancy (if measured)
- `occupancy_limiter` - What limits occupancy (registers, shmem, blocks, etc.)

**Why:** Critical for understanding performance bottlenecks.

**Calculation:**
```python
# Theoretical max warps per SM
max_warps_per_sm = 32  # For Turing (sm_75)

# Limited by registers
threads_per_block = block
warps_per_block = (threads_per_block + 31) // 32
regs_per_thread = int(row['regs'])
blocks_limited_by_regs = gpu_regs_per_sm // (warps_per_block * 32 * regs_per_thread)

# Limited by shared memory
shmem_per_block = int(row['shmem'])
blocks_limited_by_shmem = gpu_shared_mem_per_sm // shmem_per_block if shmem_per_block > 0 else 999

# Limited by max blocks per SM
blocks_limited_by_limit = gpu_max_blocks_per_sm

# Actual limit
max_blocks = min(blocks_limited_by_regs, blocks_limited_by_shmem, blocks_limited_by_limit)
theoretical_occupancy = (max_blocks * warps_per_block) / max_warps_per_sm
```

---

### 6. Add Efficiency Metrics

**Add columns:**
- `compute_efficiency` - achieved_gflops / peak_theoretical_gflops
- `bandwidth_efficiency` - achieved_gbps / peak_theoretical_gbps
- `roofline_bound` - "compute" or "memory" based on arithmetic intensity

**Why:** Makes it easy to see which kernels are well-optimized.

---

### 7. Reorganize Column Order for Better Readability

**Suggested column order:**

```python
# 1. Kernel Identification
"kernel",

# 2. Problem Size
"problem_size", "size_unit",  # OR keep N, rows, cols, etc.

# 3. Launch Configuration
"block", "grid_blocks", "regs", "shmem",

# 4. Performance Results
"mean_ms", "std_ms",

# 5. Computational Characteristics
"FLOPs", "BYTES", "arithmetic_intensity",
"mem_pattern", "working_set_bytes",
"has_branch_divergence", "atomic_ops_count",

# 6. Achieved Performance
"achieved_compute_gflops", "achieved_bandwidth_gbps",
"compute_efficiency", "bandwidth_efficiency",

# 7. GPU Hardware Specs
"gpu_device_name", "gpu_architecture", "gpu_compute_capability",
"gpu_sms", "gpu_warp_size",
"gpu_max_threads_per_sm", "gpu_max_blocks_per_sm",
"gpu_regs_per_sm", "gpu_shared_mem_per_sm", "gpu_l2_bytes",

# 8. GPU Performance Limits
"peak_theoretical_gflops", "peak_theoretical_bandwidth_gbps",
"calibrated_compute_gflops", "calibrated_mem_bandwidth_gbps",

# 9. Occupancy Analysis
"theoretical_occupancy", "occupancy_limiter",

# 10. Performance Models
"roofline_bound", "T1_model_ms", "speedup_model"
```

---

## Priority Rankings

### HIGH Priority (Missing from your core metrics):
1. ✅ **Add peak theoretical GFLOPS and bandwidth**
2. ✅ **Add architecture name**
3. ✅ **Combine compute capability into "7.5" format**
4. ✅ **Add branch divergence flag**
5. ✅ **Add atomic ops count**

### MEDIUM Priority (Improves usability):
6. **Add compute/bandwidth efficiency metrics**
7. **Add occupancy calculations**
8. **Simplify size representation**

### LOW Priority (Nice to have):
9. **Add roofline bound classification**
10. **Rename columns for clarity**
11. **Reorder columns for readability**

---

## Implementation Order

I recommend implementing in this order:

1. **First:** Add missing GPU metrics (peak theoretical, architecture)
2. **Second:** Add branch divergence and atomic ops flags
3. **Third:** Add efficiency metrics
4. **Fourth:** Add occupancy calculations
5. **Fifth:** Reorganize columns

Would you like me to implement any of these changes?

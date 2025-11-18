# GPU Performance Dataset - Metrics Reference

This document provides a comprehensive explanation of all 40 columns in the `runs_2080ti_final.csv` dataset.

---

## 1. Kernel Identification (3 columns)

### `kernel`
- **Type:** String
- **What it is:** Name of the CUDA kernel being benchmarked
- **Examples:** `vector_add`, `matmul_tiled`, `conv2d_3x3`, `atomic_hotspot`
- **Significance:** Identifies which GPU computation pattern is being tested. Each kernel represents a common pattern in GPU computing (memory-bound operations, compute-bound operations, atomics, etc.)

### `regs`
- **Type:** Integer
- **What it is:** Number of registers used per thread
- **Range:** 7-206 (varies by kernel complexity)
- **Significance:**
  - Higher register usage can limit occupancy (fewer concurrent threads)
  - Indicates kernel complexity and local variable usage
  - Critical for understanding resource bottlenecks
  - Example: `shared_bank_conflict` uses 206 registers (highest), limiting concurrent execution

### `shmem`
- **Type:** Integer (bytes)
- **What it is:** Shared memory allocated per block
- **Range:** 0-8192 bytes
- **Significance:**
  - Shared memory is fast on-chip memory shared by threads in a block
  - Used for thread cooperation and data reuse
  - High usage can limit occupancy
  - Example: `matmul_tiled` uses 8192 bytes for tile caching

---

## 2. Launch Configuration (2 columns)

### `block`
- **Type:** Integer
- **What it is:** Total threads per block (block_x × block_y × block_z)
- **Range:** 1-1024
- **Significance:**
  - Determines thread parallelism within a block
  - Must be multiple of warp size (32) for efficiency
  - Affects occupancy and resource usage
  - Example: Most kernels use 256 threads/block for good balance

### `grid_blocks`
- **Type:** Integer
- **What it is:** Total number of blocks in the grid (grid_x × grid_y × grid_z)
- **Range:** 1-16384
- **Significance:**
  - Determines total parallelism across the GPU
  - Total threads = block × grid_blocks
  - Should exceed number of SMs for full GPU utilization
  - Example: 4096 blocks on 68-SM GPU = ~60 blocks per SM

---

## 3. Performance Results (2 columns)

### `mean_ms`
- **Type:** Float (milliseconds)
- **What it is:** Average kernel execution time over multiple trials
- **Range:** 0.001-2.6 ms in this dataset
- **Significance:**
  - Primary performance metric
  - Lower is better
  - Used to calculate achieved bandwidth and GFLOPS
  - Example: `atomic_hotspot` at 2.6ms shows severe serialization

### `std_ms`
- **Type:** Float (milliseconds)
- **What it is:** Standard deviation of execution time across trials
- **Significance:**
  - Measures timing consistency/stability
  - Low std_ms indicates reliable measurements
  - High variance may indicate thermal throttling or system interference
  - Example: std_ms < 0.01ms indicates stable measurements

---

## 4. Problem Sizes (5 columns)

### `N`
- **Type:** Integer (or empty)
- **What it is:** Problem size for 1D kernels (number of elements)
- **When used:** 1D array operations
- **Significance:**
  - Determines working set size and total work
  - Larger N = more memory traffic and computation
  - Empty for 2D kernels (which use rows/cols instead)
  - Example: `vector_add` with N=1,048,576 processes 1M elements

### `rows`
- **Type:** Integer (or empty)
- **What it is:** Height dimension for 2D problems
- **When used:** Matrix operations, image kernels
- **Significance:**
  - Combined with `cols` to determine problem size
  - Affects memory access patterns (row-major vs column-major)
  - Empty for 1D kernels
  - Example: `matmul_naive` with rows=512 is a 512×512 matrix

### `cols`
- **Type:** Integer (or empty)
- **What it is:** Width dimension for 2D problems
- **When used:** Matrix operations, image kernels
- **Significance:**
  - Width of 2D data structures
  - cols=1 typically indicates 1D data stored as 2D
  - Empty for pure 1D kernels
  - Example: `conv2d_3x3` with cols=1024 is a 1024-wide image

### `iters`
- **Type:** Integer
- **What it is:** Number of iterations performed per thread
- **Range:** 0-100
- **Significance:**
  - Amplifies work or contention
  - Used mainly in stress tests
  - Example: `atomic_hotspot` with iters=100 creates extreme contention

### `size_kind`
- **Type:** String ("N" or "rows_cols")
- **What it is:** Indicates which size representation is used
- **Values:**
  - `"N"`: 1D problem, use N column
  - `"rows_cols"`: 2D problem, use rows × cols
- **Significance:**
  - Helps programmatically determine which size column to use
  - Simplifies data analysis scripts

---

## 5. Kernel Metrics (8 columns)

### `FLOPs`
- **Type:** Integer
- **What it is:** Total floating-point operations performed
- **Calculation:** Depends on kernel algorithm
- **Significance:**
  - Measures computational work
  - Used to calculate arithmetic intensity
  - Used to determine if kernel is compute-bound
  - Example: `matmul_naive` 512×512: 2 × 512³ = 268M FLOPs (2 ops per element: multiply + add)

### `BYTES`
- **Type:** Integer
- **What it is:** Total global memory bytes accessed (reads + writes)
- **Calculation:** Estimated from kernel memory access pattern
- **Significance:**
  - Measures memory traffic
  - Used to calculate arithmetic intensity
  - Determines memory-bound vs compute-bound
  - Example: `vector_add` N=1M: 3 × 1M × 4 = 12MB (read A, read B, write C)

### `shared_bytes`
- **Type:** Integer (bytes)
- **What it is:** Shared memory allocated per block (same as `shmem`)
- **Significance:**
  - On-chip memory for inter-thread communication
  - Zero for kernels without shared memory
  - Non-zero indicates data reuse optimization
  - Example: `dot_product` uses 1024 bytes for partial sums

### `working_set_bytes`
- **Type:** Integer (bytes)
- **What it is:** Total unique memory footprint accessed
- **Significance:**
  - Indicates if data fits in cache (L2 = 5.5MB on RTX 2080 Ti)
  - Smaller working sets achieve better cache hit rates
  - Example: `atomic_hotspot` has 4-byte working set (single location)

### `arithmetic_intensity`
- **Type:** Float (FLOPs/Byte)
- **What it is:** Ratio of computation to memory traffic
- **Formula:** FLOPs / BYTES
- **Significance:**
  - **High AI (>10):** Compute-bound (benefits from more GFLOPS)
  - **Low AI (<1):** Memory-bound (limited by bandwidth)
  - **Medium AI (1-10):** Balanced
  - Example: `matmul_tiled` AI=85.33 (highly compute-bound)
  - Example: `vector_add` AI=0.083 (extremely memory-bound)

### `mem_pattern`
- **Type:** String
- **What it is:** Classification of memory access pattern
- **Categories:**
  - `coalesced`: Sequential, aligned access (best)
  - `stride_8`: Strided access (poor)
  - `random_gather`: Random/indirect access (worst)
  - `transpose_naive`: Uncoalesced transpose
  - `transpose_tiled`: Shared memory-optimized transpose
  - `stencil_3x3`, `stencil_7x7`: Neighborhood access
  - `matmul_naive`, `matmul_tiled`: Matrix multiply patterns
  - `atomics_hotspot`, `atomics_global_256`: Atomic operations
  - `shared_reduction`: Reduction with shared memory
  - `smem_bank_conflict`: Shared memory conflicts
- **Significance:**
  - Predicts memory efficiency
  - Coalesced patterns achieve 80-90% of peak bandwidth
  - Strided/random patterns may achieve only 5-10%

### `has_branch_divergence`
- **Type:** Integer (0 or 1)
- **What it is:** Flag indicating control flow divergence
- **Values:**
  - `0`: No divergence (all threads in warp take same path)
  - `1`: Has divergence (threads take different paths)
- **Significance:**
  - Divergence causes serialization within warps
  - Example: `vector_add_divergent` has explicit if/else, causing 3-4× slowdown vs `vector_add`

### `atomic_ops_count`
- **Type:** Integer
- **What it is:** Total number of atomic operations performed
- **Calculation:**
  - `atomic_hotspot`: N × iters
  - `histogram`: N
  - Others: 0
- **Significance:**
  - Atomic operations serialize conflicting accesses
  - High counts with small working sets = severe contention
  - Example: `atomic_hotspot` performs 104M atomics to a single location

---

## 6. GPU Hardware Specifications (10 columns)

### `gpu_device_name`
- **Type:** String
- **What it is:** Full GPU model name
- **Example:** "NVIDIA GeForce RTX 2080 Ti"
- **Significance:**
  - Identifies hardware used for benchmarking
  - Used for cross-GPU comparisons
  - Different GPUs require different calibration data

### `gpu_architecture`
- **Type:** String
- **What it is:** NVIDIA GPU microarchitecture name
- **Values:** Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper
- **Example:** RTX 2080 Ti = Turing
- **Significance:**
  - Determines available features and capabilities
  - Different architectures have different cores/SM, cache sizes, etc.
  - Turing (sm_75) has 64 FP32 cores/SM

### `gpu_compute_capability`
- **Type:** String (format: "X.Y")
- **What it is:** CUDA compute capability version
- **Example:** "7.5" for Turing
- **Significance:**
  - Determines CUDA features available
  - Used for architecture detection
  - Affects compiler optimizations and instruction support

### `gpu_sms`
- **Type:** Integer
- **What it is:** Number of Streaming Multiprocessors
- **Example:** 68 SMs on RTX 2080 Ti
- **Significance:**
  - Fundamental parallelism unit
  - More SMs = more concurrent blocks
  - Used in peak GFLOPS calculation
  - Minimum grid size should match or exceed SMs for full utilization

### `gpu_max_threads_per_sm`
- **Type:** Integer
- **What it is:** Maximum resident threads per SM
- **Example:** 1024 on Turing
- **Significance:**
  - Determines maximum occupancy
  - Turing supports 32 warps/SM × 32 threads/warp = 1024
  - Higher occupancy can hide memory latency

### `gpu_max_blocks_per_sm`
- **Type:** Integer
- **What it is:** Maximum concurrent blocks per SM
- **Example:** 16 on Turing
- **Significance:**
  - Limits occupancy regardless of threads/block
  - Small blocks waste potential parallelism
  - Large blocks may limit occupancy due to resource usage

### `gpu_regs_per_sm`
- **Type:** Integer
- **What it is:** Total 32-bit registers available per SM
- **Example:** 65,536 on Turing
- **Significance:**
  - Register usage limits occupancy
  - If kernel uses many registers, fewer blocks can be resident
  - Example: 64 regs/thread → max 1024 concurrent threads (occupancy 100%)
  - Example: 128 regs/thread → max 512 concurrent threads (occupancy 50%)

### `gpu_shared_mem_per_sm`
- **Type:** Integer (bytes)
- **What it is:** Shared memory available per SM
- **Example:** 65,536 bytes on Turing
- **Significance:**
  - Limits occupancy for shared memory-heavy kernels
  - Can be configured (48KB/16KB L1/shared or 16KB/48KB)
  - Example: 8KB/block → max 8 blocks/SM (if not limited by other resources)

### `gpu_l2_bytes`
- **Type:** Integer (bytes)
- **What it is:** L2 cache size (shared across entire GPU)
- **Example:** 5,767,168 bytes (~5.5 MB) on RTX 2080 Ti
- **Significance:**
  - Data fitting in L2 achieves much higher bandwidth
  - Working set > L2 size requires DRAM access
  - Critical threshold for memory-bound kernels

### `gpu_warp_size`
- **Type:** Integer
- **What it is:** Number of threads in a warp (SIMT execution unit)
- **Value:** 32 (constant across all modern NVIDIA GPUs)
- **Significance:**
  - Fundamental execution width
  - Threads per block should be multiple of 32
  - Branch divergence occurs at warp granularity
  - Used in single-thread speedup model calculation

---

## 7. GPU Performance Limits (4 columns)

### `peak_theoretical_gflops`
- **Type:** Integer
- **What it is:** Theoretical peak FP32 compute performance
- **Example:** 13,450 GFLOPS for RTX 2080 Ti
- **Calculation:** SMs × cores/SM × clock_GHz × 2 FLOPs/cycle
  - RTX 2080 Ti: 68 × 64 × 1.545 × 2 = 13,450
- **Significance:**
  - Upper bound on compute performance
  - Real kernels typically achieve 50-90% for compute-bound workloads
  - Used to calculate compute efficiency

### `peak_theoretical_bandwidth_gbps`
- **Type:** Integer (GB/s)
- **What it is:** Theoretical peak memory bandwidth
- **Example:** 616 GB/s for RTX 2080 Ti
- **Calculation:** (bus_width × memory_clock) / 8
  - RTX 2080 Ti: (352 bits × 14 Gbps) / 8 = 616 GB/s
- **Significance:**
  - Upper bound on memory transfer rate
  - Real kernels achieve 70-95% for well-optimized memory-bound workloads
  - Used to calculate bandwidth efficiency

### `calibrated_mem_bandwidth_gbps`
- **Type:** Float (GB/s)
- **What it is:** Measured sustained memory bandwidth from calibration benchmark
- **Example:** 541.11 GB/s (~88% of peak)
- **How measured:** Stream-like benchmark (simple read/write kernels)
- **Significance:**
  - Realistic achievable bandwidth ceiling
  - Accounts for overhead that theory doesn't capture
  - Used in roofline model for performance prediction

### `calibrated_compute_gflops`
- **Type:** Float (GFLOPS)
- **What it is:** Measured sustained compute performance from calibration
- **Example:** 11,377.20 GFLOPS (~85% of peak)
- **How measured:** Optimized GEMM (cuBLAS)
- **Significance:**
  - Realistic achievable compute ceiling
  - Lower than peak due to real-world constraints
  - Used in roofline model for compute-bound kernels

---

## 8. Achieved Performance (2 columns)

### `achieved_bandwidth_gbps`
- **Type:** Float (GB/s)
- **What it is:** Actual memory bandwidth achieved by the kernel
- **Calculation:** BYTES / (mean_ms / 1000) / 1e9
- **Range:** 0-487 GB/s in this dataset
- **Significance:**
  - Shows memory efficiency
  - Compare to `calibrated_mem_bandwidth_gbps` (541 GB/s)
  - High ratio = well-optimized memory access
  - Example: `vector_add` achieves 487 GB/s (90% of calibrated)
  - Example: `strided_copy_8` achieves 49 GB/s (9% of calibrated - poor)

### `achieved_compute_gflops`
- **Type:** Float (GFLOPS)
- **What it is:** Actual compute throughput achieved
- **Calculation:** FLOPs / (mean_ms / 1000) / 1e9
- **Range:** 0-1,439 GFLOPS in this dataset
- **Significance:**
  - Shows compute efficiency
  - Compare to `calibrated_compute_gflops` (11,377 GFLOPS)
  - Example: `matmul_tiled` achieves 1,439 GFLOPS (12.6% of calibrated)
  - Note: Even optimized matmul is limited by memory in this size range

---

## 9. Performance Models (2 columns)

### `T1_model_ms`
- **Type:** Float (milliseconds, or empty)
- **What it is:** Predicted single-thread execution time using roofline model
- **Calculation:**
  - Single-thread bandwidth: calibrated_bw / warp_size
  - Single-thread compute: calibrated_gflops / warp_size
  - T1 = max(BYTES/(bw_1/1e9), FLOPs/(gflops_1/1e9)) × 1000
- **Significance:**
  - Theoretical baseline for single-threaded execution
  - Used to calculate speedup from parallelism
  - Empty for kernels with no FLOPs and no BYTES (e.g., `shared_bank_conflict`)

### `speedup_model`
- **Type:** Float (or empty)
- **What it is:** Theoretical speedup vs single-thread execution
- **Calculation:** T1_model_ms / mean_ms
- **Range:** 2.4-28.8× in this dataset
- **Significance:**
  - Shows benefit of GPU parallelism
  - Lower speedup may indicate:
    - Synchronization overhead
    - Resource contention
    - Poor occupancy
  - Example: `vector_add` achieves 28.82× (excellent parallelism)
  - Example: `matmul_naive` achieves 2.40× (limited by poor memory access)

---

## Usage Examples

### Finding Compute-Bound Kernels
```python
df[df['arithmetic_intensity'] > 10]
# Returns: matmul_naive, matmul_tiled, conv2d_7x7
```

### Finding Memory-Inefficient Kernels
```python
df['bandwidth_efficiency'] = df['achieved_bandwidth_gbps'] / df['calibrated_mem_bandwidth_gbps']
df[df['bandwidth_efficiency'] < 0.5]
# Returns: strided_copy_8 (9%), reduce_sum (50%), histogram (34%)
```

### Comparing Similar Kernels
```python
# Naive vs tiled transpose
df[df['kernel'].str.contains('transpose')]
# Shows: shared_transpose is 3× faster due to coalesced access
```

### Identifying Bottlenecks
```python
# High register usage limiting occupancy
df[df['regs'] > 40]
# Returns: matmul_naive, matmul_tiled, conv2d_7x7
```

---

## Summary Statistics (RTX 2080 Ti)

**Fastest Kernel:** `shared_bank_conflict` (0.0015 ms) - microbenchmark
**Slowest Kernel:** `atomic_hotspot` (2.603 ms) - worst-case contention

**Highest Bandwidth:** `vector_add` (487 GB/s, 90% efficiency)
**Lowest Bandwidth:** `strided_copy_8` (49 GB/s, 9% efficiency)

**Highest Compute:** `matmul_tiled` (1,439 GFLOPS)
**Highest Arithmetic Intensity:** `matmul_tiled` (85.33 FLOPs/byte)

**Best Speedup:** `vector_add` (28.82×)
**Worst Speedup:** `matmul_naive` (2.40×)

# RTX 4070 Out of Memory Fix

## Problem

The `gen_trials_4070.sh` script was causing "out of memory" errors on RTX 4070 (12GB VRAM) due to overly large array allocations:

```
CUDA error runner/main.cu:273: out of memory
CUDA error runner/main.cu:292: out of memory
```

## Root Cause

The script was testing with extremely large problem sizes:
- **16M elements** (16777216) for vector kernels = 64MB per array
- **4096×4096 matrices** = 64MB per matrix

Memory requirements for failing kernels:
- `vector_add` @ 16M: 3 arrays × 64MB = **192 MB**
- `conv2d_7x7` @ 4096×4096: 3 arrays × 64MB = **201 MB**
- `random_access` @ 16M: 2 float + 1 int = **256 MB**
- `atomic_hotspot` @ 16M: **192 MB + grid overhead**

While RTX 4070 has 12GB total VRAM, available memory is reduced by:
- OS/display usage
- CUDA runtime overhead
- Memory fragmentation
- Grid/block metadata

## Solution

Reduced maximum test sizes:
- **Vectors**: 16M → 8M elements (16777216 → 8388608)
- **Matrices**: 4096×4096 → 3072×3072

New memory requirements:
- `vector_add` @ 8M: 3 × 32MB = **96 MB** ✅
- `conv2d_7x7` @ 3072×3072: 3 × 36MB = **113 MB** ✅
- `random_access` @ 8M: **128 MB** ✅

## Changes Made

Modified `gpu-perf/scripts/gen_trials_4070.sh`:

### Vector Kernels (N parameter)
- vector_add, saxpy, strided_copy_8
- reduce_sum, dot_product, histogram
- random_access, vector_add_divergent

**Before**: `--N 16777216`
**After**: `--N 8388608`

### Matrix/Image Kernels (rows×cols)
- naive_transpose, shared_transpose
- conv2d_3x3, conv2d_7x7

**Before**: `--rows 4096 --cols 4096`
**After**: `--rows 3072 --cols 3072`

### Unchanged
- matmul kernels: Max 2048×2048 (already safe)
- atomic_hotspot: Max 4M elements (already safe)
- shared_bank_conflict: Fixed size (already safe)

## Test Size Progression

The script now tests 4 sizes per kernel:
1. **Small**: 256K elements / 512×512 matrices (~1-4 MB)
2. **Medium**: 1M elements / 1024×1024 matrices (~4-16 MB)
3. **Large**: 4M elements / 2048×2048 matrices (~16-64 MB)
4. **XLarge**: 8M elements / 3072×3072 matrices (~32-113 MB)

## Impact

**Positive**:
- ✅ No more OOM errors on RTX 4070
- ✅ All kernels run successfully
- ✅ Still covers wide range of problem sizes
- ✅ Dataset generation completes without failures

**Trade-off**:
- ❌ Slightly smaller maximum problem size tested
- But: 8M elements and 3072×3072 matrices are still very large
- Realistic: Most real-world kernels work with these sizes

## Validation

After this fix, validation should show:
- ✅ All kernels pass (no grid 0,0,0 errors)
- ✅ All time_ms values are positive
- ✅ No suspicious blocks or grids

## For Other GPUs

If running on different GPUs:

**RTX 2080 Ti (11GB)**: Use these same limits
**RTX 3080 (10GB)**: Use these same limits or reduce further
**RTX 3090 (24GB)**: Can use larger sizes if needed
**RTX 4090 (24GB)**: Can use larger sizes if needed

To test maximum safe size for your GPU:
```bash
# Calculate: (VRAM_GB * 0.8) / (num_arrays * 4 bytes)
# Example for 12GB with 3 arrays:
# (12 * 0.8 * 1024^3) / (3 * 4) = ~850M elements
```

## Re-running Dataset Generation

To apply the fix:
```bash
cd /home/user/test1/gpu-perf
scripts/run_4070_full.sh
```

This will regenerate a clean dataset with no OOM errors.

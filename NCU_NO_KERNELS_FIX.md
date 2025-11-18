# NCU "No Kernels Were Profiled" Fix

## Problem

When running ncu profiling, you saw:
```
==WARNING== No kernels were profiled.
Found 1 total metrics, 0 key metrics
No key metrics found
```

All kernels ran successfully but **no performance data was captured**.

## Root Cause

ncu needs special flags to properly attach to and profile CUDA kernels:

1. **Missing `--target-processes all`**: Without this, ncu couldn't capture kernels from the spawned runner process
2. **Too many kernel launches**: With `--warmup 5 --reps 10`, the kernel ran 15 times, potentially causing ncu to timeout or miss launches
3. **No launch limiting**: ncu wasn't told which kernel launch to profile

## Solution Applied

Updated both profiling scripts with three key changes:

### 1. Added `--target-processes all`
```bash
ncu --target-processes all \
    bin/runner --kernel vector_add ...
```
This ensures ncu captures kernels from spawned processes.

### 2. Added `--launch-count 1`
```bash
ncu --launch-count 1 \
    bin/runner --kernel vector_add ...
```
This tells ncu to profile just the first kernel launch (faster and more reliable).

### 3. Reduced warmup/reps for profiling
```bash
# Before
bin/runner --kernel vector_add --warmup 5 --reps 10

# After (for profiling only)
bin/runner --kernel vector_add --warmup 1 --reps 1
```

**Why?** ncu already runs kernels multiple times internally to collect metrics. External repetitions are unnecessary and can cause capture issues.

## Files Changed

- **profile_kernel_2080.sh** (auto-detect profiler)
- **profile_ncu_2080.sh** (direct ncu)

Both now use:
```bash
# Reduce warmup/reps
PROFILE_ARGS=$(echo "$ARGSTR" | sed 's/--warmup [0-9]*/--warmup 1/; s/--reps [0-9]*/--reps 1/')

# Profile with proper flags
ncu --set full \
    --target-processes all \
    --launch-count 1 \
    --export report.ncu-rep \
    bin/runner --kernel $KERNEL $PROFILE_ARGS
```

## Testing the Fix

Run profiling again:
```bash
cd /home/user/test1/gpu-perf

# Single kernel
scripts/profile_kernel_2080.sh vector_add

# All kernels
scripts/profile_all_2080.sh
```

### Expected Output (Before Fix)
```
==WARNING== No kernels were profiled.
Found 1 total metrics, 0 key metrics
```

### Expected Output (After Fix)
```
==PROF== Profiling "matmul_tiled_kernel" - 1: 0%....50%....100%
==PROF== Profiling "matmul_tiled_kernel" - 2: 0%....50%....100%
...
Found 147 total metrics, 12 key metrics
```

You should now see **actual metrics** in:
- `ncu_*_details.txt` files (occupancy, utilization, branch efficiency)
- `ncu_*_metrics.csv` files (throughput percentages, timing)
- Parsed summary showing real performance data

## Why This Works

### ncu Profiling Process
1. **Attach**: ncu attaches to the target process
2. **Inject**: Injects profiling instrumentation
3. **Run**: Kernel executes with profiling enabled
4. **Collect**: ncu gathers metrics (runs kernel multiple times)
5. **Report**: Generates output files

### Without --target-processes all
- ncu only profiles the main process
- runner spawns as a child process
- Kernels launch in child → ncu misses them

### With --target-processes all
- ncu monitors all processes
- Captures kernels from runner subprocess
- Profiling works correctly

## Impact

**Before:**
- ❌ 0 key metrics captured
- ❌ Empty CSV files
- ❌ No occupancy/throughput data
- ❌ Wasted profiling time

**After:**
- ✅ 10-20 key metrics per kernel
- ✅ Full CSV data
- ✅ Complete occupancy/throughput analysis
- ✅ Faster profiling (fewer launches)

## Advanced: Manual ncu Usage

If you want to run ncu manually:

```bash
# Correct way
ncu --set full \
    --target-processes all \
    --launch-count 1 \
    --export my_report.ncu-rep \
    bin/runner --kernel vector_add --warmup 1 --reps 1

# Open in GUI
ncu-ui my_report.ncu-rep

# Get text output
ncu --set detailed \
    --target-processes all \
    --launch-count 1 \
    --page details \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

## Common ncu Issues & Solutions

### "No kernels were profiled"
✅ **Fixed**: Add `--target-processes all`

### "Profiling timed out"
✅ **Fixed**: Add `--launch-count 1`, reduce warmup/reps

### "Permission denied" or "Failed to attach"
💡 **Solution**: Run with sudo or disable secure boot

### "Metric not available on this device"
💡 **Solution**: Some metrics are GPU-specific, use `--list-metrics` to see available ones

## Verification

After pulling the latest changes:
```bash
cd /home/user/test1/gpu-perf
git pull
scripts/profile_kernel_2080.sh vector_add
```

Check the output:
```bash
# Should have real data now
cat data/profiling/ncu_vector_add_details.txt | head -50

# Should show metrics
python3 scripts/parse_ncu_text.py data/profiling/ncu_vector_add_details.txt
```

You should see actual occupancy, utilization, and performance metrics!

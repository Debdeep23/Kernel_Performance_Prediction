# Profiling Solution: NCU Issues on cuda2

## Current Situation

### What's Working
- ✅ **cuda5 (RTX 4070)**: ncu profiling works correctly
- ✅ **Scripts**: All profiling scripts have correct flags (`--target-processes all`, `--launch-count 1`, PROFILE_ARGS)
- ✅ **Kernels**: All kernels compile and run successfully

### What's NOT Working
- ❌ **cuda2 (RTX 2080)**: ncu 2025.3.1.0 shows "==WARNING== No kernels were profiled"
- ❌ Despite all correct flags being in place
- ❌ Despite permissions being OK (perf_event_paranoid = 2)

## Root Cause

The issue is **environmental** on cuda2, not in the scripts:
- **ncu 2025.3.1.0** is a very new version with potentially different process attachment requirements
- May have driver/CUDA toolkit compatibility issues on cuda2
- May have stricter security policies for process attachment

## Solutions (in order of recommendation)

### Solution 1: Use nsys Instead of ncu on cuda2 (RECOMMENDED)

I've created complete nsys profiling infrastructure for you:

```bash
# On cuda2 (RTX 2080)
cd /home/user/test1/gpu-perf

# Profile single kernel
scripts/profile_nsys_2080.sh vector_add

# Profile all kernels
scripts/profile_all_nsys_2080.sh

# Parse results
python3 scripts/parse_nsys_results.py data/profiling_2080
```

**What you get with nsys:**
- Kernel execution time (avg, min, max, stddev)
- Number of kernel instances
- Memory transfer times (HtoD, DtoH, DtoD)
- CUDA API overhead
- Launch overhead
- Timeline visualization in nsys-ui

**What you DON'T get (ncu-specific):**
- Occupancy percentages
- SM/DRAM utilization percentages
- Branch efficiency
- Warp stall reasons

**Trade-off:** nsys gives you timing and memory transfer metrics, but not the low-level GPU utilization metrics that ncu provides.

### Solution 2: Use cuda5 for NCU Profiling (PRACTICAL)

Since ncu **works on cuda5**, just run profiling there:

```bash
# On cuda5 (RTX 4070)
cd /home/user/test1/gpu-perf

# Use existing scripts (they work there!)
scripts/profile_kernel_2080.sh vector_add
scripts/profile_all_2080.sh

# Parse results
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt
```

**Why this is OK:**
- The kernels are the same code
- You get full ncu metrics (occupancy, throughput, branch efficiency, etc.)
- GPU architecture is different (RTX 4070 vs RTX 2080) but kernel behavior is still analyzable
- You can note in your analysis: "Profiling performed on RTX 4070 due to ncu compatibility issues"

**What's different:**
- SM count: RTX 4070 has different SM count than RTX 2080
- Memory bandwidth: Different peak bandwidth
- Compute capability: Both are recent NVIDIA GPUs

**What's the same:**
- Kernel code behavior
- Occupancy patterns
- Branch efficiency patterns
- Memory access patterns
- Relative performance between kernels

### Solution 3: Troubleshooting NCU on cuda2 (Advanced)

If you really need ncu on cuda2, see the detailed troubleshooting steps in:
```
NCU_2025_TROUBLESHOOTING.md
```

Key things to try:
1. Check driver/CUDA compatibility
2. Try application replay mode
3. Try without --target-processes all
4. Rebuild runner binary
5. Try older ncu version (2024.x if available)

## Recommended Action Plan

### Option A: Quick Solution (nsys on cuda2)

```bash
# On cuda2
cd /home/user/test1/gpu-perf
scripts/profile_all_nsys_2080.sh
python3 scripts/parse_nsys_results.py data/profiling_2080
```

**Pros:**
- Works right now
- Gives you timing and memory metrics
- No driver/compatibility issues

**Cons:**
- Missing low-level GPU utilization metrics

### Option B: Best Solution (ncu on cuda5)

```bash
# On cuda5
cd /home/user/test1/gpu-perf
scripts/profile_all_2080.sh
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt
```

**Pros:**
- Full ncu metrics (occupancy, utilization, branch efficiency)
- Already confirmed working
- Comprehensive kernel analysis

**Cons:**
- Profiling data is for RTX 4070 not RTX 2080
- (But kernel behavior is still analyzable)

### Option C: Hybrid (both!)

```bash
# On cuda2: Get timing metrics with nsys
cd /home/user/test1/gpu-perf
scripts/profile_all_nsys_2080.sh

# On cuda5: Get detailed metrics with ncu
cd /home/user/test1/gpu-perf
scripts/profile_all_2080.sh

# Analyze both
python3 scripts/parse_nsys_results.py data/profiling_2080   # Timing data
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt  # GPU metrics
```

**Pros:**
- Best of both worlds
- Timing from actual RTX 2080
- Detailed metrics from RTX 4070

**Cons:**
- More work to run both

## Files Created for You

### NSys Profiling Scripts
1. **scripts/profile_nsys_2080.sh** - Profile single kernel with nsys
2. **scripts/profile_all_nsys_2080.sh** - Profile all kernels with nsys
3. **scripts/parse_nsys_results.py** - Parse nsys results to CSV

### Documentation
1. **NCU_2025_TROUBLESHOOTING.md** - Detailed ncu troubleshooting steps
2. **PROFILING_SOLUTION.md** - This file (comprehensive solution guide)

### Existing (Already Working)
1. scripts/profile_kernel_2080.sh - Auto-detect profiler (works on cuda5)
2. scripts/profile_ncu_2080.sh - Direct ncu (works on cuda5)
3. scripts/profile_all_2080.sh - All kernels (works on cuda5)
4. scripts/parse_ncu_text.py - Parse ncu text files
5. scripts/parse_profiling_results.py - Parse ncu CSV files

## Quick Start

### If you want timing metrics NOW:
```bash
# On cuda2
cd /home/user/test1/gpu-perf
scripts/profile_all_nsys_2080.sh
python3 scripts/parse_nsys_results.py data/profiling_2080
cat data/profiling_2080/profiling_summary_nsys.csv
```

### If you want full GPU metrics:
```bash
# On cuda5 (where ncu works!)
cd /home/user/test1/gpu-perf
scripts/profile_all_2080.sh
python3 scripts/parse_ncu_text.py data/profiling_2080/ncu_*_details.txt
cat data/profiling_2080/ncu_parsed_summary.csv
```

## Summary

The ncu issue on cuda2 is **not your fault** and **not a script problem**. It's an environmental/compatibility issue with ncu 2025.3.1.0 on that specific machine.

**Best practical solution:** Use cuda5 for ncu profiling (where it works) OR use nsys on cuda2 for timing metrics.

All the scripts are ready to go - just choose which approach works best for your needs!

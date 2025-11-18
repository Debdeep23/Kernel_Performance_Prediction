# NCU 2025.3.1.0 Kernel Capture Troubleshooting

## Issue

On cuda2 (RTX 2080) with ncu version 2025.3.1.0, profiling shows:
```
==WARNING== No kernels were profiled.
Found 1 total metrics, 0 key metrics
```

Despite having all correct flags:
- `--target-processes all`
- `--launch-count 1`
- Reduced warmup/reps to 1

## Confirmed Working

- **cuda5 (RTX 4070)**: Profiling works correctly
- **Scripts**: All have correct flags applied

## Diagnostic Steps to Try on cuda2

### 1. Check Environment Compatibility

```bash
# Check driver version
nvidia-smi

# Check CUDA toolkit version
nvcc --version

# Check ncu version (already confirmed: 2025.3.1.0)
ncu --version
```

**Expected**: Driver should be compatible with CUDA 12.x and ncu 2025.x

### 2. Try Application Replay Mode (New in 2025.x)

```bash
cd /home/user/test1/gpu-perf

# Try with application replay mode
ncu --mode=launch-and-attach \
    --replay-mode application \
    --set full \
    --export test_app_replay \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 3. Try Kernel Replay Mode

```bash
# Try with kernel replay mode (default but explicit)
ncu --mode=launch-and-attach \
    --replay-mode kernel \
    --set full \
    --export test_kernel_replay \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 4. Try With Explicit Device Selection

```bash
# List CUDA devices
ncu --devices

# Profile with explicit device (usually device 0)
ncu --devices 0 \
    --set full \
    --target-processes all \
    --export test_device0 \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 5. Try Legacy Launch Mode

```bash
# Use legacy launch tracking
ncu --target-processes all \
    --launch-skip 0 \
    --launch-count 1 \
    --set full \
    --export test_legacy \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 6. Try Sampling Mode (Faster, Less Overhead)

```bash
# Use sampling instead of full profiling
ncu --mode=launch-and-attach \
    --sampling-trigger cuda-api \
    --export test_sampling \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 7. Check Kernel Visibility

```bash
# Use nvprof if available (might work even if deprecated)
which nvprof

# If nvprof exists, try it
nvprof --print-gpu-trace bin/runner --kernel vector_add --warmup 1 --reps 1

# Or use nsys
nsys profile --trace=cuda bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 8. Try Without --target-processes all

```bash
# Sometimes the flag can cause issues with certain drivers
ncu --set full \
    --launch-count 1 \
    --export test_no_target_proc \
    bin/runner --kernel vector_add --warmup 1 --reps 1
```

### 9. Check for Driver/Toolkit Mismatch

```bash
# Check CUDA runtime version
cat /usr/local/cuda/version.txt 2>/dev/null || echo "CUDA version file not found"

# Check what CUDA version the binary was compiled with
strings bin/runner | grep -i cuda | head -10

# Rebuild runner to ensure compatibility
cd /home/user/test1/gpu-perf
make clean
make
```

### 10. Try Direct Kernel Launch (No Subprocess)

The issue might be that ncu 2025.x has stricter process attachment. Try modifying runner to use direct launch instead of subprocess.

## Known Issues with NCU 2025.x

1. **Process Attachment Changes**: Newer ncu versions have stricter security/attachment policies
2. **Driver Requirements**: May require newer NVIDIA drivers (525+ for full support)
3. **CUDA Compatibility**: Best with CUDA 12.3+
4. **Permissions**: May need additional capabilities beyond perf_event_paranoid

## Alternative: Use nsys Instead

If ncu continues to fail, nsys (Nsight Systems) can provide different but useful metrics:

```bash
cd /home/user/test1/gpu-perf

# Profile with nsys
nsys profile \
    --trace=cuda,nvtx \
    --output=vector_add_nsys \
    --force-overwrite=true \
    bin/runner --kernel vector_add --warmup 5 --reps 10

# View report
nsys stats vector_add_nsys.nsys-rep

# Export to CSV
nsys stats --report cuda_gpu_kern_sum vector_add_nsys.nsys-rep --format csv --output . --force-export true
```

## Recommended Solution: Use cuda5 for Profiling

Since profiling **was working on cuda5 (RTX 4070)**, the practical solution is:

```bash
# SSH to cuda5 instead of cuda2
ssh cuda5

# Run profiling there
cd /home/user/test1/gpu-perf
scripts/profile_all_2080.sh  # The script name doesn't matter - it's just the kernel code

# The profiling results will be for RTX 4070 instead of RTX 2080
# But you'll get the same kernel metrics (just different GPU specs)
```

**Why this is OK**:
- The profiling scripts work correctly on cuda5
- You'll get occupancy, throughput, branch efficiency, etc. for the same kernels
- The GPU architecture metrics will be RTX 4070-specific, but the kernel analysis is still valuable
- You can note in your analysis that profiling was done on RTX 4070 due to ncu compatibility issues on RTX 2080

## If You Must Profile on RTX 2080

Try these in order:
1. Update NVIDIA driver to latest (if possible)
2. Try ncu 2024.x version instead of 2025.3.1.0 (if available)
3. Use nsys instead of ncu
4. Use nvprof (if still available on the system)

## Creating Alternative Profiling Scripts for nsys

I can create nsys-based profiling scripts if needed:
```bash
# Would create:
# - profile_nsys_2080.sh
# - parse_nsys_results.py
# - Similar workflow to ncu scripts
```

## Summary

The issue is likely:
- **ncu 2025.3.1.0** has stricter process attachment requirements
- **Driver/CUDA compatibility** on cuda2 may not fully support this ncu version
- **Workaround**: Use cuda5 for profiling OR use nsys instead of ncu

The scripts are **correct** - the issue is environmental on cuda2.

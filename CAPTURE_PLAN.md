# How to Capture Missing Metrics

## Summary: ALL Missing Metrics Can Be Captured! ✅

---

## 1. Branch Divergence Flag (EASY - 5 min)

### Can we capture it? ✅ YES - Static Analysis
### Method: Add kernel classification

**What we CAN capture:**
- Boolean flag indicating if kernel has branch divergence
- Based on source code analysis

**What we CANNOT capture without profiling:**
- Actual divergence percentage at runtime
- Warp efficiency metrics
- (Would need Nsight Compute or nvprof for runtime metrics)

**Implementation:**
```python
# In build_final_dataset.py, add to static_counts() function

DIVERGENT_KERNELS = {
    "vector_add_divergent",  # Has explicit if/else per thread
}

# Add to return values:
has_divergence = 1 if kernel in DIVERGENT_KERNELS else 0

# Return: FLOPs, BYTES, SH, WS, AI, pat, has_divergence
```

**Add to CSV:**
```python
r["has_branch_divergence"] = has_divergence
```

---

## 2. Atomic Operations Count (EASY - 5 min)

### Can we capture it? ✅ YES - Static Calculation
### Method: Calculate from kernel parameters

**What we CAN capture:**
- Theoretical number of atomic operations
- Based on problem size and kernel logic

**Formulas by kernel:**
```python
# atomic_hotspot: Each of N threads does 'iters' atomics to same location
atomic_ops = N * iters

# histogram: Each of N elements does 1 atomic increment
atomic_ops = N

# Other kernels: 0
atomic_ops = 0
```

**Implementation:**
```python
# In build_final_dataset.py, modify static_counts()

def static_counts(row):
    k = row["kernel"]
    # ... existing code ...

    # Calculate atomic operations
    atomic_ops = 0
    if k == "atomic_hotspot":
        atomic_ops = N * max(1, iters)
    elif k == "histogram":
        atomic_ops = N

    # Return: FLOPs, BYTES, SH, WS, AI, pat, has_divergence, atomic_ops
```

---

## 3. Peak Theoretical FP32 GFLOPS (EASY - 5 min)

### Can we capture it? ✅ YES - From GPU Specs
### Method: Calculate from architecture specs or use known values

**Formula:**
```
Peak GFLOPS = SMs × Cores_per_SM × Clock_GHz × FLOPs_per_cycle
```

**For RTX 2080 Ti (sm_75 / Turing):**
```
Peak = 68 SMs × 64 FP32_cores/SM × 1.545 GHz × 2 FLOPs/cycle
     = 13,450 GFLOPS (spec sheet value)

Conservative (base clock):
Peak = 68 SMs × 64 × 1.350 GHz × 2 = 11,674 GFLOPS
```

**Implementation Option 1 - Lookup Table:**
```python
# Add to build_final_dataset.py

GPU_SPECS = {
    "NVIDIA GeForce RTX 2080 Ti": {
        "architecture": "Turing",
        "compute_capability": "7.5",
        "peak_gflops_fp32": 13450,  # Boost clock
        "peak_bandwidth_gbps": 616,
        "base_clock_ghz": 1.350,
        "boost_clock_ghz": 1.545,
    },
    "NVIDIA GeForce RTX 3090": {
        "architecture": "Ampere",
        "compute_capability": "8.6",
        "peak_gflops_fp32": 35580,
        "peak_bandwidth_gbps": 936,
    },
    # Add more GPUs as needed
}
```

**Implementation Option 2 - Calculate from specs:**
```python
def calculate_peak_gflops(device_name, sms, cc_major, cc_minor):
    # Cores per SM by architecture
    CORES_PER_SM = {
        (7, 5): 64,   # Turing
        (8, 0): 64,   # Ampere GA100
        (8, 6): 128,  # Ampere GA102/GA104
        (8, 9): 128,  # Ada Lovelace
        (9, 0): 128,  # Hopper
    }

    cores = CORES_PER_SM.get((cc_major, cc_minor), 64)

    # Estimate boost clock from model (or hardcode known values)
    if "2080 Ti" in device_name:
        clock_ghz = 1.545
    elif "3090" in device_name:
        clock_ghz = 1.695
    else:
        clock_ghz = 1.5  # Conservative estimate

    return sms * cores * clock_ghz * 2  # 2 FLOPs per cycle (FMA)
```

**Recommended:** Use lookup table for accuracy, fallback to calculation.

---

## 4. Peak Theoretical Bandwidth (EASY - 5 min)

### Can we capture it? ✅ YES - From Memory Specs
### Method: Known specifications or calculate from memory config

**Formula:**
```
Peak BW (GB/s) = (Memory_Bus_Width_bits × Memory_Clock_Gbps) / 8
```

**For RTX 2080 Ti:**
```
Peak BW = (352 bits × 14 Gbps) / 8 = 616 GB/s
```

**Implementation:**
```python
GPU_SPECS = {
    "NVIDIA GeForce RTX 2080 Ti": {
        "peak_bandwidth_gbps": 616,
        "memory_bus_width_bits": 352,
        "memory_clock_gbps": 14,
    },
}

# Or calculate:
def calculate_peak_bandwidth(device_name):
    if "2080 Ti" in device_name:
        return 352 * 14 / 8  # = 616 GB/s
    elif "3090" in device_name:
        return 384 * 19.5 / 8  # = 936 GB/s
    # ... etc
```

---

## 5. Architecture Name (TRIVIAL - 2 min)

### Can we capture it? ✅ YES - From Compute Capability
### Method: Map compute capability to architecture

**Mapping:**
```python
ARCHITECTURE_MAP = {
    (7, 5): "Turing",
    (7, 0): "Volta",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (6, 1): "Pascal",
    (6, 0): "Pascal",
}

def get_architecture(cc_major, cc_minor):
    return ARCHITECTURE_MAP.get((cc_major, cc_minor), "Unknown")
```

**Implementation:**
```python
# In build_final_dataset.py
arch = get_architecture(P["cc_major"], P["cc_minor"])
r["gpu_architecture"] = arch
```

---

## 6. Full Compute Capability (TRIVIAL - 1 min)

### Can we capture it? ✅ YES - Already Have It!
### Method: Combine major.minor from props file

**Current situation:**
- We HAVE both major (7) and minor (5) in `props_2080ti.out`
- We just removed `gpu_cc_minor` in our cleanup
- Need to create combined "7.5" format

**Implementation:**
```python
# Instead of:
r["gpu_cc_major"] = str(P["cc_major"])

# Do:
r["gpu_compute_capability"] = f"{P['cc_major']}.{P['cc_minor']}"
# Result: "7.5"
```

---

## Complete Implementation Plan

### Phase 1: Add GPU Specs (5 min)
```python
# In build_final_dataset.py, add at top:

GPU_SPECS = {
    "NVIDIA GeForce RTX 2080 Ti": {
        "architecture": "Turing",
        "peak_gflops_fp32": 13450,
        "peak_bandwidth_gbps": 616,
    },
}

ARCHITECTURE_MAP = {
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
}
```

### Phase 2: Modify static_counts() (5 min)
```python
def static_counts(row):
    # ... existing code ...

    # Branch divergence
    DIVERGENT_KERNELS = {"vector_add_divergent"}
    has_divergence = 1 if k in DIVERGENT_KERNELS else 0

    # Atomic operations
    atomic_ops = 0
    if k == "atomic_hotspot":
        atomic_ops = N * max(1, iters)
    elif k == "histogram":
        atomic_ops = N

    return FLOPs, BYTES, SH, WS, AI, pat, has_divergence, atomic_ops
```

### Phase 3: Add to dataset (5 min)
```python
# After reading GPU props:
device_name = P["device_name"]
specs = GPU_SPECS.get(device_name, {})
arch = ARCHITECTURE_MAP.get((P["cc_major"], P["cc_minor"]), "Unknown")

for r in agg:
    # ... existing fields ...

    # New GPU metrics
    r["gpu_architecture"] = arch
    r["gpu_compute_capability"] = f"{P['cc_major']}.{P['cc_minor']}"
    r["peak_theoretical_gflops"] = specs.get("peak_gflops_fp32", 0)
    r["peak_theoretical_bandwidth_gbps"] = specs.get("peak_bandwidth_gbps", 0)

    # New kernel metrics
    r["has_branch_divergence"] = has_divergence
    r["atomic_ops_count"] = atomic_ops
```

### Phase 4: Update column list (2 min)
```python
flds = [
    "kernel","regs","shmem",
    "block","grid_blocks",
    "mean_ms","std_ms",
    "N","rows","cols","H","W","matN","size_kind",
    "FLOPs","BYTES","shared_bytes","working_set_bytes",
    "arithmetic_intensity","mem_pattern",
    "has_branch_divergence","atomic_ops_count",  # NEW
    "gpu_device_name","gpu_architecture","gpu_compute_capability",  # NEW
    "gpu_sms","gpu_max_threads_per_sm","gpu_max_blocks_per_sm",
    "gpu_regs_per_sm","gpu_shared_mem_per_sm","gpu_l2_bytes","gpu_warp_size",
    "peak_theoretical_gflops","peak_theoretical_bandwidth_gbps",  # NEW
    "calibrated_mem_bandwidth_gbps","calibrated_compute_gflops",
    "achieved_bandwidth_gbps","achieved_compute_gflops",
    "T1_model_ms","speedup_model"
]
```

---

## What We CANNOT Capture (Without Profiling Tools)

### Runtime Metrics (Would need Nsight Compute):
- ❌ Actual warp divergence percentage
- ❌ Actual occupancy (vs theoretical)
- ❌ Cache hit rates (L1/L2)
- ❌ Memory transaction efficiency
- ❌ Stall reasons (memory/execution/sync)
- ❌ Tensor core utilization (for newer GPUs)

### Advanced Metrics:
- ❌ Power consumption
- ❌ Temperature during execution
- ❌ PCIe transfer overhead

**But these are not in your core metrics list, so we're good!** ✅

---

## Summary

| Metric | Can Capture? | Method | Difficulty | Time |
|--------|-------------|---------|-----------|------|
| Branch divergence flag | ✅ YES | Static kernel classification | Easy | 5 min |
| Atomic ops count | ✅ YES | Calculate from parameters | Easy | 5 min |
| Peak theoretical GFLOPS | ✅ YES | GPU spec lookup | Easy | 5 min |
| Peak theoretical BW | ✅ YES | GPU spec lookup | Easy | 5 min |
| Architecture name | ✅ YES | Compute capability mapping | Trivial | 2 min |
| Full compute capability | ✅ YES | Combine major.minor | Trivial | 1 min |

**Total implementation time: ~20 minutes**

All missing metrics from your core list can be captured! 🎉

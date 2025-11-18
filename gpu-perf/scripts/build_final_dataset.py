#!/usr/bin/env python3
import csv, re, sys, glob, math, statistics

# ---------- GPU specifications lookup ----------
GPU_SPECS = {
    "NVIDIA GeForce RTX 2080 Ti": {
        "architecture": "Turing",
        "peak_gflops_fp32": 13450,  # 68 SMs × 64 cores/SM × 1.545 GHz boost × 2 FLOPs/cycle
        "peak_bandwidth_gbps": 616,  # 352-bit × 14 Gbps / 8
    },
    "NVIDIA GeForce RTX 3090": {
        "architecture": "Ampere",
        "peak_gflops_fp32": 35580,
        "peak_bandwidth_gbps": 936,
    },
    "NVIDIA GeForce RTX 3080": {
        "architecture": "Ampere",
        "peak_gflops_fp32": 29770,
        "peak_bandwidth_gbps": 760,
    },
    "NVIDIA GeForce RTX 4090": {
        "architecture": "Ada Lovelace",
        "peak_gflops_fp32": 82580,
        "peak_bandwidth_gbps": 1008,
    },
}

ARCHITECTURE_MAP = {
    (6, 0): "Pascal",
    (6, 1): "Pascal",
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
}

# ---------- tiny helpers ----------
def I(x, d=0):
    try:
        if x is None or x == "": return d
        return int(float(str(x).strip()))
    except:
        return d

def F(x, d=0.0):
    try:
        if x is None or x == "": return d
        return float(str(x).strip())
    except:
        return d

def clean_int_field(s):
    if s is None: return 0
    m = re.findall(r'-?\d+', str(s))
    return int(m[0]) if m else 0

def kv_from_file(path):
    kv = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # Check if line has multiple key=value pairs (e.g., "major=7 minor=5")
            # by looking for space-separated tokens that all contain '='
            tokens = ln.split()
            if all('=' in token for token in tokens):
                # Multiple key=value pairs on one line
                for token in tokens:
                    k, v = token.split('=', 1)
                    kv[k] = v
            elif '=' in ln:
                # Single key=value pair (may have spaces in value)
                k, v = ln.split('=', 1)
                kv[k.strip()] = v.strip()
    return kv

# ---------- size + shape sanitation ----------
def pick_size_family(row):
    rows, cols = I(row.get("rows")), I(row.get("cols"))
    N    = I(row.get("N"))

    # fallbacks: accept whatever the trial used
    if N == 0 and rows > 0:
        N = rows if cols <= 1 else rows * cols

    # Mutually exclusive: either N (1D) or rows/cols (2D), never both
    out = {"N":"", "rows":"", "cols":""}

    # 2D kernels: rows > 0 and cols > 1
    if rows > 0 and cols > 1:
        # 2D kernel: only populate rows/cols
        out["rows"] = str(rows)
        out["cols"] = str(cols)
    else:
        # 1D kernel: consolidate into N only
        if N == 0 and rows > 0:
            N = rows * max(1, cols)
        if N > 0:
            out["N"] = str(N)

    return out

def sane_block_grid(row):
    bx = max(1, clean_int_field(row.get("block_x")))
    by = max(1, clean_int_field(row.get("block_y")))
    bz = max(1, clean_int_field(row.get("block_z")))
    gx = max(1, clean_int_field(row.get("grid_x")))
    gy = max(1, clean_int_field(row.get("grid_y")))
    gz = max(1, clean_int_field(row.get("grid_z")))
    return bx,by,bz,gx,gy,gz

# ---------- per-kernel FLOPs/BYTES model ----------
def static_counts(row):
    k = row["kernel"]
    # sizes with fallbacks
    rows, cols = I(row.get("rows")), I(row.get("cols"))
    N    = I(row.get("N"))
    iters = I(row.get("iters"))
    blk   = max(1, I(row.get("block")))

    # Derive N from rows/cols if needed
    if N == 0 and rows > 0:
        N = rows if cols <= 1 else rows * cols

    # For matmul kernels, derive matN from rows (assumes square matrices)
    matN = rows if rows > 0 and rows == cols else 0

    FLOPs = 0; BYTES = 0; WS = 0; pat = "coalesced"

    if k == "vector_add":
        FLOPs = N
        BYTES = 3*N*4
        WS    = BYTES
        pat   = "coalesced"
    elif k == "vector_add_divergent":
        FLOPs = N
        BYTES = 3*N*4
        WS    = BYTES
        pat   = "divergent"
    elif k == "saxpy":
        FLOPs = 2*N
        BYTES = 3*N*4
        WS    = BYTES
    elif k == "strided_copy_8":
        touched = N//8
        BYTES = 2*touched*4
        WS    = BYTES
        pat   = "stride_8"
    elif k == "naive_transpose":
        elems = rows*cols
        BYTES = 2*elems*4
        WS    = BYTES
        pat   = "transpose_naive"
    elif k == "shared_transpose":
        elems = rows*cols
        BYTES = 2*elems*4
        WS    = BYTES
        pat   = "transpose_tiled"
    elif k == "matmul_naive":
        FLOPs = 2*matN*matN*matN
        BYTES = 4*(matN*matN*3)
        WS    = BYTES
        pat   = "matmul_naive"
    elif k == "matmul_tiled":
        FLOPs = 2*matN*matN*matN
        BYTES = 4*(matN*matN*3)
        WS    = BYTES
        pat   = "matmul_tiled"
    elif k == "reduce_sum":
        FLOPs   = max(0, N-1)
        partial = max(1, (N // (2*blk))) * 4
        BYTES   = N*4 + partial + 4
        WS      = N*4
        pat     = "shared_reduction"
    elif k == "dot_product":
        FLOPs   = 2*N
        partial = max(1, (N // (2*blk))) * 4
        BYTES   = 2*N*4 + partial + 4
        WS      = 2*N*4
        pat     = "shared_reduction"
    elif k == "histogram":
        BYTES = N*4 + N*8
        WS    = N*4 + 256*4
        pat   = "atomics_global_256"
    elif k == "random_access":
        BYTES = 2*N*4
        WS    = BYTES
        pat   = "random_gather"
    elif k == "atomic_hotspot":
        BYTES = max(1,N)*max(1,iters)*8
        WS    = 4
        pat   = "atomics_hotspot"
    elif k == "conv2d_3x3":
        elems = max(1, rows*cols)
        FLOPs = 18 * elems
        BYTES = 2  * elems * 4
        WS    = elems * 4
        pat   = "stencil_3x3"
    elif k == "conv2d_7x7":
        elems = max(1, rows*cols)
        FLOPs = 98 * elems
        BYTES = 2  * elems * 4
        WS    = elems * 4
        pat   = "stencil_7x7"
    elif k == "shared_bank_conflict":
        WS  = 4096  # shared memory only kernel
        pat = "smem_bank_conflict"
    else:
        # fallback: try not to emit zeros if we can infer an element count
        n = 0
        if rows>0 and cols>0: n = rows*cols
        elif N>0: n = N
        BYTES = 2*n*4 if n>0 else 0
        WS    = BYTES
        pat   = "unknown"

    AI = (FLOPs/float(BYTES)) if BYTES>0 else 0.0

    # Branch divergence flag (kernels with explicit control flow divergence)
    DIVERGENT_KERNELS = {"vector_add_divergent"}
    has_divergence = 1 if k in DIVERGENT_KERNELS else 0

    # Atomic operations count
    atomic_ops = 0
    if k == "atomic_hotspot":
        atomic_ops = max(1, N) * max(1, iters)
    elif k == "histogram":
        atomic_ops = N

    return FLOPs, BYTES, WS, AI, pat, has_divergence, atomic_ops

# ---------- T1 + speedup ----------
def add_T1_and_speedup(row, peak_gflops, peak_gbps, warp_size):
    fl = F(row.get("FLOPs"))
    by = F(row.get("BYTES"))
    mean_ms = F(row.get("mean_ms"))
    if fl==0 and by==0:
        row["T1_model_ms"] = ""
        row["speedup_model"] = ""
        return
    g1 = peak_gflops/warp_size if peak_gflops>0 else 0.0
    b1 = peak_gbps  /warp_size if peak_gbps>0   else 0.0
    tcomp = (1000.0*fl/(g1*1e9)) if (fl>0 and g1>0) else 0.0
    tmem  = (1000.0*by/(b1*1e9)) if (by>0 and b1>0) else 0.0
    T1 = max(tcomp, tmem)
    row["T1_model_ms"]   = f"{T1:.6f}"
    row["speedup_model"] = f"{(T1/mean_ms):.2f}" if (T1>0 and mean_ms>0) else ""

# ---------- read props + ceilings ----------
def read_props(props_out):
    kv = kv_from_file(props_out)
    out = {}
    out["device_name"] = kv.get("name","")
    # compute capability broken across two lines in your props output
    out["cc_major"] = int(re.sub(r'[^0-9]','', kv.get("major","0")) or 0)
    out["cc_minor"] = int(re.sub(r'[^0-9]','', kv.get("minor","0")) or 0)
    for k in ["multiProcessorCount","maxThreadsPerMultiProcessor","maxBlocksPerMultiProcessor",
              "regsPerMultiprocessor","sharedMemPerMultiprocessor","sharedMemPerBlockOptin",
              "maxThreadsPerBlock","warpSize","l2CacheSizeBytes"]:
        v = kv.get(k)
        out[k] = int(re.sub(r'[^0-9\-]','', v) or 0) if v is not None else 0
    return out

def read_ceiling(path, key):
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith(key+"="):
                try:
                    return float(ln.split("=",1)[1])
                except:
                    return 0.0
    return 0.0

# ---------- aggregate trials -> rows ----------
def aggregate_trials(trial_glob):
    rows = []
    groups = {}  # key -> list of time_ms

    for path in glob.glob(trial_glob):
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            for r in rd:
                # sanitize sizes + block/grid
                size = pick_size_family(r)
                for k,v in size.items():
                    r[k] = v
                bx,by,bz,gx,gy,gz = sane_block_grid(r)
                r["block_x"], r["block_y"], r["block_z"] = str(bx),str(by),str(bz)
                r["grid_x"],  r["grid_y"],  r["grid_z"]  = str(gx),str(gy),str(gz)
                r["block"]       = str(bx*by*bz)
                r["grid_blocks"] = str(gx*gy*gz)

                key = (
                    r["kernel"], r["args"], r["regs"], r["shmem"], r["device_name"],
                    r["block_x"], r["block_y"], r["block_z"], r["grid_x"], r["grid_y"], r["grid_z"],
                    r.get("warmup",""), r.get("reps",""),
                    r.get("N",""), r.get("rows",""), r.get("cols",""),
                    r.get("iters","")
                )
                groups.setdefault(key, []).append(F(r.get("time_ms"), 0.0))

    out = []
    for key, times in groups.items():
        if not times: continue
        mean_ms = statistics.fmean(times)
        std_ms  = statistics.pstdev(times) if len(times)>1 else 0.0
        (kernel,args,regs,shmem,device_name,
         bx,by,bz,gx,gy,gz,warmup,reps,
         N,rows,cols,iters) = key
        out.append({
            "kernel": kernel, "args": args, "regs": regs, "shmem": shmem, "device_name": device_name,
            "block_x": bx, "block_y": by, "block_z": bz,
            "grid_x": gx, "grid_y": gy, "grid_z": gz,
            "warmup": warmup, "reps": reps,
            "trials": str(len(times)),
            "mean_ms": f"{mean_ms:.6f}",
            "std_ms": f"{std_ms:.6f}",
            "N": N, "rows": rows, "cols": cols,
            "block": str(I(bx)*I(by)*I(bz)),
            "grid_blocks": str(I(gx)*I(gy)*I(gz)),
        })
    return out

def main():
    if len(sys.argv) != 6:
        print("usage: build_final_dataset.py <trial_glob> <props_out> <stream_out> <gemm_out> <final_csv>", file=sys.stderr)
        sys.exit(1)

    trial_glob, props_out, stream_out, gemm_out, final_csv = sys.argv[1:]

    # 1) aggregate per-kernel
    agg = aggregate_trials(trial_glob)

    # 2) per-kernel static counts + size_kind
    for r in agg:
        FLOPs,BYTES,WS,AI,pat,has_divergence,atomic_ops = static_counts(r)
        r["FLOPs"] = str(FLOPs)
        r["BYTES"] = str(BYTES)
        r["shared_bytes"] = r.get("shmem", "0")  # Use actual shmem from kernel launch
        r["working_set_bytes"] = str(WS)
        r["arithmetic_intensity"] = f"{AI:.6f}"
        r["mem_pattern"] = pat
        r["has_branch_divergence"] = str(has_divergence)
        r["atomic_ops_count"] = str(atomic_ops)
        # Determine size_kind
        rows = I(r.get("rows"))
        cols = I(r.get("cols"))
        if rows > 0 and cols > 1:
            r["size_kind"] = "rows_cols"
        else:
            r["size_kind"] = "N"

    # 3) GPU metrics + sustained ceilings
    P = read_props(props_out)
    bw = read_ceiling(stream_out, "SUSTAINED_MEM_BW_GBPS")
    fl = read_ceiling(gemm_out,   "SUSTAINED_COMPUTE_GFLOPS")
    warp = P.get("warpSize", 32) or 32

    # Get GPU specs from lookup table
    device_name = P["device_name"]
    specs = GPU_SPECS.get(device_name, {})
    architecture = ARCHITECTURE_MAP.get((P["cc_major"], P.get("cc_minor", 0)), "Unknown")
    compute_capability = f"{P['cc_major']}.{P.get('cc_minor', 0)}"
    peak_gflops = specs.get("peak_gflops_fp32", 0)
    peak_bandwidth = specs.get("peak_bandwidth_gbps", 0)

    for r in agg:
        r.update({
            "gpu_device_name": P["device_name"],
            "gpu_architecture": architecture,
            "gpu_compute_capability": compute_capability,
            "gpu_sms": str(P["multiProcessorCount"]),
            "gpu_max_threads_per_sm": str(P["maxThreadsPerMultiProcessor"]),
            "gpu_max_blocks_per_sm": str(P["maxBlocksPerMultiProcessor"]),
            "gpu_regs_per_sm": str(P["regsPerMultiprocessor"]),
            "gpu_shared_mem_per_sm": str(P["sharedMemPerMultiprocessor"]),
            "gpu_l2_bytes": str(P["l2CacheSizeBytes"]),
            "gpu_warp_size": str(P["warpSize"]),
            "peak_theoretical_gflops": str(peak_gflops),
            "peak_theoretical_bandwidth_gbps": str(peak_bandwidth),
            "calibrated_mem_bandwidth_gbps": f"{bw:.2f}",
            "calibrated_compute_gflops": f"{fl:.2f}",
        })
        # Calculate achieved metrics
        mean_ms = F(r.get("mean_ms"))
        flops = F(r.get("FLOPs"))
        bytes_accessed = F(r.get("BYTES"))
        if mean_ms > 0:
            r["achieved_compute_gflops"] = f"{(flops/(mean_ms/1000.0)/1e9):.2f}"
            r["achieved_bandwidth_gbps"] = f"{(bytes_accessed/(mean_ms/1000.0)/1e9):.2f}"
        else:
            r["achieved_compute_gflops"] = "0.00"
            r["achieved_bandwidth_gbps"] = "0.00"
        add_T1_and_speedup(r, fl, bw, warp)

    # 4) write final CSV with all metrics (11/11 kernel metrics + 13/13 GPU metrics)
    flds = [
        # Kernel identification
        "kernel","regs","shmem",
        # Launch configuration
        "block","grid_blocks",
        # Performance results
        "mean_ms","std_ms",
        # Problem sizes
        "N","rows","cols","iters","size_kind",
        # Kernel metrics (11 total)
        "FLOPs","BYTES","shared_bytes","working_set_bytes",
        "arithmetic_intensity","mem_pattern",
        "has_branch_divergence","atomic_ops_count",
        # GPU hardware specs (13 total)
        "gpu_device_name","gpu_architecture","gpu_compute_capability",
        "gpu_sms","gpu_max_threads_per_sm","gpu_max_blocks_per_sm",
        "gpu_regs_per_sm","gpu_shared_mem_per_sm","gpu_l2_bytes","gpu_warp_size",
        # GPU performance limits
        "peak_theoretical_gflops","peak_theoretical_bandwidth_gbps",
        "calibrated_mem_bandwidth_gbps","calibrated_compute_gflops",
        # Achieved performance
        "achieved_bandwidth_gbps","achieved_compute_gflops",
        # Performance models
        "T1_model_ms","speedup_model"
    ]
    with open(final_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds, extrasaction='ignore')
        w.writeheader()
        for r in agg:
            w.writerow(r)

    print(f"[OK] wrote {final_csv}")

if __name__ == "__main__":
    main()


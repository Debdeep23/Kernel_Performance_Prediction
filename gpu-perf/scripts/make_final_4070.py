#!/usr/bin/env python3
import csv, glob, json, math, os, re, statistics as stat

DATA = "data"
TAG = "4070"   # RTX 4070

# ---------- helpers ----------
def F(x):
    try: return float(str(x).strip())
    except: return 0.0

def I(x):
    try: return int(str(x).strip())
    except: return 0

def extract_iters_from_args(args):
    """Extract --iters N from args string"""
    m = re.search(r'--iters\s+(\d+)', args)
    return int(m.group(1)) if m else 0

def grab_last_number(pattern, text):
    out = None
    for ln in text.splitlines():
        m = re.search(pattern, ln)
        if m:
            out = float(m.group(1))
    return out or 0.0

def parse_props_text(txt):
    # Accept both "k=v" per line and "major=8 minor=9" on one line.
    kv = {}
    for ln in txt.splitlines():
        for token in ln.strip().split():
            if "=" in token:
                k,v = token.split("=",1)
                kv[k] = v
    # Normalize ints when possible
    int_keys = [
        "multiProcessorCount","maxThreadsPerMultiProcessor","maxBlocksPerMultiProcessor",
        "regsPerMultiprocessor","sharedMemPerMultiprocessor","warpSize","l2CacheSizeBytes",
        "maxThreadsPerBlock","major","minor"
    ]
    for k in int_keys:
        if k in kv:
            try:
                kv[k] = int(re.sub(r"[^\d-].*$","", kv[k]))
            except: pass
    return kv

def sustained_from_files(stream_txt, gemm_txt):
    # Prefer explicit SUSTAINED_ lines, otherwise fallback to last GBps=/GFLOPS=
    def get(name, txt, fallback_pat):
        for ln in reversed(txt.splitlines()):
            if ln.startswith(name + "="):
                try: return float(ln.split("=",1)[1].strip())
                except: pass
        return grab_last_number(fallback_pat, txt)

    bw = get("SUSTAINED_MEM_BW_GBPS", stream_txt, r"GBps=([0-9.]+)")
    fl = get("SUSTAINED_COMPUTE_GFLOPS", gemm_txt, r"GFLOPS=([0-9.]+)")
    return bw, fl

def roofline_time_ms(flops, bytes_, peak_gflops, peak_gbps):
    # Handle pure-mem / pure-compute sensibly
    t_comp = 1000.0 * (flops  / (peak_gflops*1e9)) if peak_gflops>0 and flops>0  else 0.0
    t_mem  = 1000.0 * (bytes_ / (peak_gbps  *1e9)) if peak_gbps>0   and bytes_>0 else 0.0
    if flops>0 and bytes_>0: return max(t_comp, t_mem)
    if flops>0:               return t_comp
    if bytes_>0:              return t_mem
    return float("nan")

# ---------- static counts per kernel (global bytes only) ----------
def static_counts(row):
    k    = row["kernel"]
    N    = I(row.get("N", ""))
    rows = I(row.get("rows", ""))
    cols = I(row.get("cols", ""))
    iters= I(row.get("iters", ""))

    # Derive matN and H/W from normalized rows/cols
    matN = rows if (rows > 0 and rows == cols) else 0
    H = rows
    W = cols

    bx = I(row.get("block_x","1")); by = I(row.get("block_y","1")); bz = I(row.get("block_z","1"))
    gx = I(row.get("grid_x","1"));  gy = I(row.get("grid_y","1"));  gz = I(row.get("grid_z","1"))
    total_threads = bx*by*bz * gx*gy*gz

    flops = 0
    gbytes = 0
    shared_bytes = 0
    working_set = 0
    mem_pattern = ""

    if k == "vector_add":
        flops  = N * 1
        gbytes = N * (4+4+4)
        working_set = N * 12
        mem_pattern = "coalesced"

    elif k == "saxpy":
        flops  = N * 2
        gbytes = N * (4+4+4)
        working_set = N * 12
        mem_pattern = "coalesced"

    elif k == "strided_copy_8":
        M = (N + 7) // 8
        flops  = 0
        gbytes = M * (4+4)
        working_set = M * 8
        mem_pattern = "stride_8"

    elif k == "reduce_sum":
        flops  = N               # ~1 add/elt lower bound
        gbytes = N*4 + 4*max(1,(N // max(1,(bx*2))))
        working_set = N*4
        mem_pattern = "shared_reduction"

    elif k == "dot_product":
        flops  = N*2
        gbytes = N*8 + 4*max(1,(N // max(1,(bx*2))))
        working_set = N*8
        mem_pattern = "shared_reduction"

    elif k == "histogram":
        # treat atomic RMW ≈ 8B
        flops  = 0
        gbytes = N * 8
        working_set = 256*4
        mem_pattern = "atomic_bins"

    elif k == "naive_transpose":
        NE     = rows*cols
        gbytes = NE * (4+4)
        working_set = NE*8
        mem_pattern = "coalesced_rw"

    elif k == "shared_transpose":
        NE     = rows*cols
        gbytes = NE * (4+4)
        shared_bytes = 32*32*4
        working_set = NE*8
        mem_pattern = "coalesced_rw_shared"

    elif k in ("matmul_naive","matmul_tiled"):
        flops  = 2 * matN*matN*matN
        gbytes = 3 * matN*matN * 4
        working_set = gbytes
        mem_pattern = "dense_gemm"

    elif k == "conv2d_3x3":
        NE = H*W
        flops  = NE * 18
        gbytes = NE * (4+4)
        working_set = NE*8
        mem_pattern = "stencil_3x3"

    elif k == "conv2d_7x7":
        NE = H*W
        flops  = NE * 98
        gbytes = NE * (4+4)
        working_set = NE*8
        mem_pattern = "stencil_7x7"

    elif k == "random_access":
        flops  = 0
        gbytes = N * (4+4+4)     # A + idx + B
        working_set = N*12
        mem_pattern = "gather_scatter"

    elif k == "vector_add_divergent":
        flops  = N * 1
        gbytes = N * (4+4+4)
        working_set = N*12
        mem_pattern = "divergent"

    elif k == "shared_bank_conflict":
        # shared only; no global traffic. keep flops tiny to avoid 0/0
        flops  = max(1, total_threads)
        gbytes = 0
        shared_bytes = I(row.get("shmem","0"))
        working_set = 0
        mem_pattern = "shared_only"

    elif k == "atomic_hotspot":
        atomic_ops = total_threads * max(1, iters)
        gbytes = atomic_ops * 8
        working_set = 4
        mem_pattern = "atomic_hotspot"

    # Arithmetic intensity (vs global bytes)
    if gbytes>0 and flops>0:
        ai = flops / gbytes
    elif flops>0 and gbytes==0:
        ai = float("inf")
    else:
        ai = 0.0

    return flops, gbytes, ai, working_set, mem_pattern, shared_bytes

# ---------- aggregator ----------
def aggregate_trials():
    files = sorted(glob.glob(os.path.join(DATA, f"trials_*__{TAG}.csv")))
    if not files:
        raise SystemExit(f"No trials found: {DATA}/trials_*__{TAG}.csv")

    rows = []
    for p in files:
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            rows.extend(rd)

    # group by stable key (ignore per-trial time_ms)
    def key(r):
        keys = ["kernel","args","regs","shmem","device_name",
                "block_x","block_y","block_z","grid_x","grid_y","grid_z",
                "warmup","reps","N","rows","cols","matN","H","W","iters","block","grid_blocks"]
        return tuple(r.get(k,"") for k in keys)

    groups = {}
    for r in rows:
        k = key(r)
        groups.setdefault(k, []).append(F(r.get("time_ms","0")))

    # produce aggregated rows
    out = []
    for k, times in groups.items():
        row = {}
        keys = ["kernel","args","regs","shmem","device_name",
                "block_x","block_y","block_z","grid_x","grid_y","grid_z",
                "warmup","reps","N","rows","cols","matN","H","W","iters","block","grid_blocks"]
        for i,kk in enumerate(keys):
            row[kk] = k[i]
        row["trials"] = len(times)
        row["mean_ms"] = f"{(sum(times)/len(times)):.6f}"
        row["std_ms"]  = f"{(stat.pstdev(times) if len(times)>1 else 0.0):.6f}"

        # Extract iters from args if not already present
        if not row.get("iters") or row["iters"] == "":
            row["iters"] = str(extract_iters_from_args(row.get("args", "")))

        # Normalize sizes: make N/rows/cols mutually exclusive
        N = I(row.get("N", ""))
        rows_val = I(row.get("rows", ""))
        cols_val = I(row.get("cols", ""))
        matN = I(row.get("matN", ""))
        H = I(row.get("H", ""))
        W = I(row.get("W", ""))

        # Derive final N, rows, cols
        if matN > 0:
            # Matrix kernels: use rows=cols=matN
            row["N"] = ""
            row["rows"] = str(matN)
            row["cols"] = str(matN)
            row["size_kind"] = "rows_cols"
        elif H > 0 and W > 0:
            # Conv kernels: use rows=H, cols=W
            row["N"] = ""
            row["rows"] = str(H)
            row["cols"] = str(W)
            row["size_kind"] = "rows_cols"
        elif rows_val > 0 and cols_val > 1:
            # 2D kernels: keep rows/cols
            row["N"] = ""
            row["rows"] = str(rows_val)
            row["cols"] = str(cols_val)
            row["size_kind"] = "rows_cols"
        else:
            # 1D kernels: consolidate to N only
            if N > 0:
                problem_size = N
            elif rows_val > 0:
                problem_size = rows_val * max(1, cols_val)
            else:
                problem_size = 0

            row["N"] = str(problem_size) if problem_size > 0 else ""
            row["rows"] = "0"
            row["cols"] = "0"
            row["size_kind"] = "N"

        out.append(row)

    return out

def main():
    # 1) aggregate trials
    agg = aggregate_trials()

    # 2) read GPU files
    props_txt  = open(os.path.join(DATA,"props_4070.out")).read()
    stream_txt = open(os.path.join(DATA,"stream_like_4070.out")).read()
    gemm_txt   = open(os.path.join(DATA,"gemm_cublas_4070.out")).read()

    P = parse_props_text(props_txt)
    BW_sus, FLOP_sus = sustained_from_files(stream_txt, gemm_txt)

    warp_size = P.get("warpSize", 32)
    l2_bytes  = int(P.get("l2CacheSizeBytes", 0))

    # fallbacks for "theoretical" so columns are never blank
    peak_theoretical_gflops = FLOP_sus
    peak_theoretical_gbps   = BW_sus

    # 3) write final CSV with updated schema (matches current runs_4070_final.csv)
    fieldnames = [
        "kernel","regs","shmem","mean_ms","std_ms",
        "N","rows","cols","iters",
        "block","grid_blocks","size_kind",
        "FLOPs","BYTES","arithmetic_intensity","working_set_bytes","shared_bytes","mem_pattern",
        "gpu_device_name","gpu_cc_major","gpu_sms",
        "gpu_max_threads_per_sm","gpu_max_blocks_per_sm",
        "gpu_regs_per_sm","gpu_shared_mem_per_sm",
        "gpu_l2_bytes","gpu_warp_size",
        "calibrated_mem_bandwidth_gbps","calibrated_compute_gflops",
        "achieved_bandwidth_gbps","achieved_compute_gflops",
        "T1_model_ms","speedup_model"
    ]

    out_path = os.path.join(DATA,"runs_4070_final.csv")
    with open(out_path, "w", newline='') as g:
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for r in agg:
            # static counts
            fl, gb, ai, ws, pat, shb = static_counts(r)

            # static columns
            r["FLOPs"]                 = fl
            r["BYTES"]                 = gb
            r["arithmetic_intensity"]  = ("inf" if ai==float("inf") else f"{ai:.6f}")
            r["working_set_bytes"]     = ws
            r["shared_bytes"]          = shb
            r["mem_pattern"]           = pat

            # gpu metrics (updated field names to match schema)
            r["gpu_device_name"]       = r.get("device_name","")
            r["gpu_cc_major"]          = P.get("major","")
            r["gpu_sms"]               = P.get("multiProcessorCount","")
            r["gpu_warp_size"]         = P.get("warpSize","")
            r["gpu_max_threads_per_sm"] = P.get("maxThreadsPerMultiProcessor","")
            r["gpu_max_blocks_per_sm"]  = P.get("maxBlocksPerMultiProcessor","")
            r["gpu_regs_per_sm"]       = P.get("regsPerMultiprocessor","")
            r["gpu_shared_mem_per_sm"] = P.get("sharedMemPerMultiprocessor","")
            r["gpu_l2_bytes"]          = l2_bytes
            r["calibrated_compute_gflops"] = f"{FLOP_sus:.2f}"
            r["calibrated_mem_bandwidth_gbps"] = f"{BW_sus:.2f}"

            # achieved metrics
            mean_ms = F(r.get("mean_ms","0"))
            if mean_ms > 0:
                r["achieved_compute_gflops"] = f"{(fl/(mean_ms/1000.0)/1e9):.2f}"
                r["achieved_bandwidth_gbps"] = f"{(gb/(mean_ms/1000.0)/1e9):.2f}"
            else:
                r["achieved_compute_gflops"] = "0.00"
                r["achieved_bandwidth_gbps"] = "0.00"

            # single-thread baseline & speedup
            peak1_gflops = FLOP_sus / warp_size if FLOP_sus>0 else 0.0
            peak1_gbps   = BW_sus   / warp_size if BW_sus>0   else 0.0
            T1 = roofline_time_ms(fl, gb, peak1_gflops, peak1_gbps)
            r["T1_model_ms"]   = "" if math.isnan(T1) else f"{T1:.6f}"
            r["speedup_model"] = "" if math.isnan(T1) or mean_ms<=0 else f"{(T1/mean_ms):.2f}"

            # ensure all declared fields are present
            for f in fieldnames:
                if f not in r: r[f] = ""

            w.writerow({f:r[f] for f in fieldnames})

    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()

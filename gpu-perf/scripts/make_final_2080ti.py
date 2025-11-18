#!/usr/bin/env python3
import csv, glob, json, math, os, re, statistics as stat

DATA = "data"
TAG = "2080ti"   # change if you run on another card

# ---------- helpers ----------
def F(x):
    try: return float(str(x).strip())
    except: return 0.0

def I(x):
    try: return int(str(x).strip())
    except: return 0

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
    H    = I(row.get("H", ""))
    W    = I(row.get("W", ""))
    matN = I(row.get("matN", ""))
    iters= I(row.get("iters", ""))

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
        out.append(row)

    return out

def main():
    # 1) aggregate trials
    agg = aggregate_trials()

    # 2) read GPU files
    props_txt  = open(os.path.join(DATA,"props_2080ti.out")).read()
    stream_txt = open(os.path.join(DATA,"stream_like_2080ti.out")).read()
    gemm_txt   = open(os.path.join(DATA,"gemm_cublas_2080ti.out")).read()

    P = parse_props_text(props_txt)
    BW_sus, FLOP_sus = sustained_from_files(stream_txt, gemm_txt)

    warp_size = P.get("warpSize", 32)
    l2_bytes  = int(P.get("l2CacheSizeBytes", 0))

    # fallbacks for "theoretical" so columns are never blank
    peak_theoretical_gflops = FLOP_sus
    peak_theoretical_gbps   = BW_sus

    # 3) write final CSV
    fieldnames = [
        # measured + launch shape
        "kernel","args","device_name","regs","shmem",
        "block_x","block_y","block_z","grid_x","grid_y","grid_z",
        "warmup","reps","trials","mean_ms","std_ms",
        # size family (only one set will be non-empty per kernel)
        "N","rows","cols","matN","H","W","block","grid_blocks",
        # static counts
        "FLOPs","BYTES","arithmetic_intensity","working_set_bytes","shared_bytes","mem_pattern",
        # GPU metrics
        "gpu_name","compute_capability","sm_count","warp_size",
        "max_threads_per_sm","max_blocks_per_sm",
        "registers_per_sm","shared_mem_per_sm",
        "peak_theoretical_gflops","peak_theoretical_gbps",
        "sustained_compute_gflops","sustained_mem_bandwidth_gbps",
        "gpu_l2_bytes","estimated_l2_bytes",
        # single-thread baseline + speedup
        "T1_model_ms","speedup_model"
    ]

    out_path = os.path.join(DATA,"runs_2080ti_final.csv")
    with open(out_path, "w", newline='') as g:
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for r in agg:
            # static counts
            fl, gb, ai, ws, pat, shb = static_counts(r)

            # gpu metrics
            r["gpu_name"]            = r.get("device_name","")
            r["compute_capability"]  = f'{P.get("major","")}.{P.get("minor","")}'
            r["sm_count"]            = P.get("multiProcessorCount","")
            r["warp_size"]           = P.get("warpSize","")
            r["max_threads_per_sm"]  = P.get("maxThreadsPerMultiProcessor","")
            r["max_blocks_per_sm"]   = P.get("maxBlocksPerMultiProcessor","")
            r["registers_per_sm"]    = P.get("regsPerMultiprocessor","")
            r["shared_mem_per_sm"]   = P.get("sharedMemPerMultiprocessor","")
            r["peak_theoretical_gflops"]  = peak_theoretical_gflops
            r["peak_theoretical_gbps"]    = peak_theoretical_gbps
            r["sustained_compute_gflops"] = FLOP_sus
            r["sustained_mem_bandwidth_gbps"] = BW_sus
            r["gpu_l2_bytes"]        = l2_bytes
            r["estimated_l2_bytes"]  = min(int(gb), 2*l2_bytes) if l2_bytes>0 else int(gb)

            # static columns
            r["FLOPs"]                 = fl
            r["BYTES"]                 = gb
            r["arithmetic_intensity"]  = ("inf" if ai==float("inf") else f"{ai:.6f}")
            r["working_set_bytes"]     = ws
            r["shared_bytes"]          = shb
            r["mem_pattern"]           = pat

            # single-thread baseline & speedup
            peak1_gflops = FLOP_sus / warp_size if FLOP_sus>0 else 0.0
            peak1_gbps   = BW_sus   / warp_size if BW_sus>0   else 0.0
            T1 = roofline_time_ms(fl, gb, peak1_gflops, peak1_gbps)
            mean_ms = F(r.get("mean_ms","0"))
            r["T1_model_ms"]   = "" if math.isnan(T1) else f"{T1:.6f}"
            r["speedup_model"] = "" if math.isnan(T1) or mean_ms<=0 else f"{(T1/mean_ms):.2f}"

            # ensure all declared fields are present
            for f in fieldnames:
                if f not in r: r[f] = ""

            w.writerow({f:r[f] for f in fieldnames})

    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()


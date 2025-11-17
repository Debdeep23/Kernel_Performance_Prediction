#!/usr/bin/env python3
import csv, re, sys, glob, math, statistics

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
    pat = re.compile(r'^([A-Za-z0-9_]+)=(.*)$')
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            m = pat.match(ln)
            if not m: continue
            k, v = m.group(1), m.group(2)
            kv[k] = v
    return kv

# ---------- size + shape sanitation ----------
def pick_size_family(row):
    rows, cols = I(row.get("rows")), I(row.get("cols"))
    N    = I(row.get("N"))
    H, W = I(row.get("H")), I(row.get("W"))
    matN = I(row.get("matN"))

    # fallbacks: accept whatever the trial used
    if N == 0 and rows > 0:
        N = rows if cols <= 1 else rows * cols
    if H == 0 and W == 0 and rows > 0 and cols > 0:
        H, W = rows, cols
    if matN == 0 and rows > 0 and (cols == 0 or rows == cols):
        matN = rows

    # emit exactly ONE family
    out = {"N":"", "rows":"", "cols":"", "H":"", "W":"", "matN":""}
    if rows > 0 and cols > 0:
        out["rows"], out["cols"] = str(rows), str(cols)
    elif matN > 0:
        out["matN"] = str(matN)
    elif H > 0 and W > 0:
        out["H"], out["W"] = str(H), str(W)
    elif N > 0:
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
    H, W = I(row.get("H")), I(row.get("W"))
    matN = I(row.get("matN"))
    iters = I(row.get("iters"))
    blk   = max(1, I(row.get("block")))

    if N == 0 and rows > 0:
        N = rows if cols <= 1 else rows * cols
    if H == 0 and W == 0 and rows > 0 and cols > 0:
        H, W = rows, cols
    if matN == 0 and rows > 0 and (cols == 0 or rows == cols):
        matN = rows

    FLOPs = 0; BYTES = 0; SH = 0; WS = 0; pat = "coalesced"

    if k in ("vector_add","vector_add_divergent"):
        FLOPs = N
        BYTES = 3*N*4
        WS    = BYTES
        pat   = "coalesced"
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
        SH    = elems*4
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
        SH    = 2*matN*matN*4
        WS    = BYTES
        pat   = "matmul_tiled"
    elif k == "reduce_sum":
        FLOPs   = max(0, N-1)
        partial = max(1, (N // (2*blk))) * 4
        BYTES   = N*4 + partial + 4
        SH      = blk*4
        WS      = N*4
        pat     = "shared_reduction"
    elif k == "dot_product":
        FLOPs   = 2*N
        partial = max(1, (N // (2*blk))) * 4
        BYTES   = 2*N*4 + partial + 4
        SH      = blk*4
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
        elems = max(1, rows*cols if rows*cols>0 else H*W)
        FLOPs = 18 * elems
        BYTES = 2  * elems * 4
        WS    = elems * 4
        pat   = "stencil_3x3"
    elif k == "conv2d_7x7":
        elems = max(1, rows*cols if rows*cols>0 else H*W)
        FLOPs = 98 * elems
        BYTES = 2  * elems * 4
        WS    = elems * 4
        pat   = "stencil_7x7"
    elif k == "shared_bank_conflict":
        SH  = max(1, blk)*4
        WS  = SH
        pat = "smem_bank_conflict"
    else:
        # fallback: try not to emit zeros if we can infer an element count
        n = 0
        if rows>0 and cols>0: n = rows*cols
        elif N>0: n = N
        elif H>0 and W>0: n = H*W
        BYTES = 2*n*4 if n>0 else 0
        WS    = BYTES
        pat   = "unknown"

    AI = (FLOPs/float(BYTES)) if BYTES>0 else 0.0
    return FLOPs, BYTES, SH, WS, AI, pat

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
                    r.get("N",""), r.get("rows",""), r.get("cols",""), r.get("H",""), r.get("W",""), r.get("matN",""),
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
         N,rows,cols,H,W,matN,iters) = key
        out.append({
            "kernel": kernel, "args": args, "regs": regs, "shmem": shmem, "device_name": device_name,
            "block_x": bx, "block_y": by, "block_z": bz,
            "grid_x": gx, "grid_y": gy, "grid_z": gz,
            "warmup": warmup, "reps": reps,
            "trials": str(len(times)),
            "mean_ms": f"{mean_ms:.6f}",
            "std_ms": f"{std_ms:.6f}",
            "N": N, "rows": rows, "cols": cols, "H": H, "W": W, "matN": matN,
            "iters": iters,
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

    # 2) per-kernel static counts
    for r in agg:
        FLOPs,BYTES,SH,WS,AI,pat = static_counts(r)
        r["FLOPs"] = str(FLOPs)
        r["BYTES"] = str(BYTES)
        r["shared_bytes"] = str(SH)
        r["working_set_bytes"] = str(WS)
        r["arithmetic_intensity"] = f"{AI:.6f}"
        r["mem_pattern"] = pat

    # 3) GPU metrics + sustained ceilings
    P = read_props(props_out)
    bw = read_ceiling(stream_out, "SUSTAINED_MEM_BW_GBPS")
    fl = read_ceiling(gemm_out,   "SUSTAINED_COMPUTE_GFLOPS")
    warp = P.get("warpSize", 32) or 32

    for r in agg:
        r.update({
            "gpu_device_name": P["device_name"],
            "gpu_cc_major": str(P["cc_major"]),
            "gpu_cc_minor": str(P["cc_minor"]),
            "gpu_sms": str(P["multiProcessorCount"]),
            "gpu_max_threads_per_sm": str(P["maxThreadsPerMultiProcessor"]),
            "gpu_max_blocks_per_sm": str(P["maxBlocksPerMultiProcessor"]),
            "gpu_regs_per_sm": str(P["regsPerMultiprocessor"]),
            "gpu_shared_mem_per_sm": str(P["sharedMemPerMultiprocessor"]),
            "gpu_l2_bytes": str(P["l2CacheSizeBytes"]),
            "gpu_warp_size": str(P["warpSize"]),
            "sustained_mem_bandwidth_gbps": f"{bw:.2f}",
            "sustained_compute_gflops": f"{fl:.2f}",
        })
        add_T1_and_speedup(r, fl, bw, warp)

    # 4) write final CSV
    flds = [
        "kernel","args","regs","shmem","device_name",
        "block_x","block_y","block_z","grid_x","grid_y","grid_z",
        "block","grid_blocks",
        "warmup","reps","trials","mean_ms","std_ms",
        "N","rows","cols","H","W","matN","iters",
        "FLOPs","BYTES","shared_bytes","working_set_bytes","arithmetic_intensity","mem_pattern",
        "gpu_device_name","gpu_cc_major","gpu_cc_minor","gpu_sms",
        "gpu_max_threads_per_sm","gpu_max_blocks_per_sm","gpu_regs_per_sm","gpu_shared_mem_per_sm",
        "gpu_l2_bytes","gpu_warp_size",
        "sustained_mem_bandwidth_gbps","sustained_compute_gflops",
        "T1_model_ms","speedup_model"
    ]
    with open(final_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in agg:
            w.writerow(r)

    print(f"[OK] wrote {final_csv}")

if __name__ == "__main__":
    main()


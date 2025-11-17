import sys, csv, math

inp  = sys.argv[1]
outp = sys.argv[2]

def I(x):
    try: return int(x)
    except: return 0

def counts_for_kernel(k, rows, cols, block, iters):
    N = rows if cols == 1 else rows*cols
    FLOPs = 0
    BYTES = 0
    shared_bytes = 0
    mem_pattern = "coalesced"
    working_set_bytes = 0

    if k == "vector_add":
        FLOPs = N
        BYTES = 3*N*4
        working_set_bytes = BYTES

    elif k == "saxpy":
        FLOPs = 2*N
        BYTES = 3*N*4
        working_set_bytes = BYTES

    elif k == "strided_copy_8":
        # copy every 8th element
        touched = N//8
        FLOPs = 0
        BYTES = 2*touched*4
        mem_pattern = "stride_8"
        working_set_bytes = 2*touched*4

    elif k == "naive_transpose":
        FLOPs = 0
        BYTES = 2*rows*cols*4
        mem_pattern = "transpose_naive"
        working_set_bytes = BYTES

    elif k == "shared_transpose":
        FLOPs = 0
        BYTES = 2*rows*cols*4
        # rough SMEM tile traffic proxy (not double-counting global)
        shared_bytes = rows*cols*4
        mem_pattern = "transpose_tiled"
        working_set_bytes = BYTES

    elif k == "matmul_naive":
        # rows==cols==matN (after normalization)
        n = rows
        FLOPs = 2*n*n*n
        BYTES = 4*(n*n*3)             # A,B,C once (pessimistic but stable)
        working_set_bytes = BYTES

    elif k == "matmul_tiled":
        n = rows
        FLOPs = 2*n*n*n
        BYTES = 4*(n*n*3)
        shared_bytes = 2*n*n*4         # tiles exchanged through SMEM (proxy)
        working_set_bytes = BYTES

    elif k == "reduce_sum":
        # 2*elements per thread reduction pattern
        FLOPs = N
        BYTES = N*4 + max(1, N//(2*max(1,block)))*4
        shared_bytes = max(1,block)*4
        mem_pattern = "shared_reduction"
        working_set_bytes = int(3*N*4)

    elif k == "dot_product":
        FLOPs = 2*N
        BYTES = 2*N*4 + max(1, N//(2*max(1,block)))*4
        shared_bytes = max(1,block)*4
        mem_pattern = "shared_reduction"
        working_set_bytes = 2*N*4

    elif k == "histogram":
        # one atomic per element into 256 bins (uint32)
        FLOPs = 0
        BYTES = N*8                    # RMW ~ 2 * 4B
        mem_pattern = "atomics_global_256"
        working_set_bytes = N*4 + 256*4

    elif k == "random_access":
        FLOPs = 0
        BYTES = 2*N*4                  # one gather + one write
        mem_pattern = "random_gather"
        working_set_bytes = 2*N*4

    elif k == "shared_bank_conflict":
        FLOPs = 0
        BYTES = 0                      # global-traffic-free by design
        shared_bytes = max(1,block)*4
        mem_pattern = "smem_bank_conflict"
        working_set_bytes = shared_bytes

    elif k == "atomic_hotspot":
        FLOPs = 0
        BYTES = N * max(1,iters) * 8   # many RMWs to one counter
        mem_pattern = "atomics_hotspot"
        working_set_bytes = 4

    return FLOPs, BYTES, shared_bytes, mem_pattern, int(working_set_bytes)

rd = csv.DictReader(open(inp))
base = rd.fieldnames
add  = ["FLOPs","BYTES","shared_bytes","working_set_bytes","mem_pattern","arithmetic_intensity"]
out_fields = base + [f for f in add if f not in base]

w = csv.DictWriter(open(outp,"w",newline=""), fieldnames=out_fields)
w.writeheader()

for r in rd:
    k     = r["kernel"]
    rows  = I(r.get("rows",""))
    cols  = I(r.get("cols",""))
    block = I(r.get("block","") or r.get("block_x",""))
    iters = I(r.get("iters",""))

    fl, by, sb, mp, ws = counts_for_kernel(k, rows, cols, block, iters)
    r["FLOPs"] = fl
    r["BYTES"] = by
    r["shared_bytes"] = sb
    r["working_set_bytes"] = ws
    r["mem_pattern"] = mp
    r["arithmetic_intensity"] = (fl/float(by)) if by>0 else 0.0
    w.writerow(r)

print(f"[OK] wrote {outp}")


#!/usr/bin/env python3
import sys, csv

# Usage:
#   python3 scripts/static_counts.py <runs_in.csv> <runs_out.csv>
#
# Input is the aggregated "runs" CSV (one row per kernel/shape).
# Output adds static metrics: FLOPs, BYTES, arithmetic_intensity, working_set_bytes,
# shared_bytes (per-block), mem_pattern, size_kind, and normalizes size fields.

def I(x):
    try: return int(float(x))
    except: return 0

def F(x):
    try: return float(x)
    except: return 0.0

def first_missing_add(fields, name):
    if name not in fields:
        fields.append(name)

def size_from_row(r):
    """
    Normalize sizes:
      - Prefer explicit N if present
      - Else infer N from rows*cols when cols>1, otherwise N=rows
      - For convs, also set H=W from rows/cols if absent
    Returns (size_kind, N, rows, cols)
    """
    rows = I(r.get("rows"))
    cols = I(r.get("cols"))
    N    = I(r.get("N"))  # may be missing in input

    # Fallbacks: unify to either N or rows/cols
    if N == 0:
        if rows > 0 and cols > 1:
            N = rows * cols
        elif rows > 0 and cols <= 1:
            N = rows

    # If both rows and cols are zero but N > 0, prefer 1D representation
    if rows == 0 and cols == 0 and N > 0:
        rows, cols = N, 1

    # Choose size_kind
    if rows > 0 and cols > 0 and cols != 1:
        size_kind = "rows_cols"
    else:
        size_kind = "N"

    return size_kind, N, rows, cols

def per_kernel_counts(k, r, size_kind, N, rows, cols):
    """
    Return FLOPs, BYTES, shared_bytes (per-block), mem_pattern,
           working_set_bytes, conv_padding
    All byte counts assume float32 unless kernel is clearly integer (histogram bins).
    """
    block = I(r.get("block"))           # threads per block
    iters = I(r.get("iters"))           # some kernels use an --iters arg

    FLOPs = 0
    BYTES = 0
    shared_bytes = 0       # PER-BLOCK estimate
    mem_pattern = "coalesced"
    working_set_bytes = 0
    conv_padding = ""      # "same" or "valid" if you want to label convs

    # Treat sizes
    R = rows
    C = cols

    if k == "vector_add":
        # C = A + B
        FLOPs = N                   # 1 add
        BYTES = 3 * N * 4           # read A,B; write C
        working_set_bytes = BYTES

    elif k == "vector_add_divergent":
        FLOPs = N
        BYTES = 3 * N * 4
        working_set_bytes = BYTES
        mem_pattern = "divergent"

    elif k == "saxpy":
        # y = a*x + y
        FLOPs = 2 * N               # mul + add
        BYTES = 3 * N * 4           # read x,y; write y
        working_set_bytes = BYTES

    elif k == "strided_copy_8":
        # If your kernel actually touched all elements, count full traffic (read+write)
        BYTES = 2 * N * 4
        working_set_bytes = BYTES
        FLOPs = 0
        mem_pattern = "strided_8"

    elif k == "random_access":
        # 1 random read + 1 write
        FLOPs = 0
        BYTES = 2 * N * 4
        working_set_bytes = BYTES
        mem_pattern = "random_gather_scatter"

    elif k == "reduce_sum":
        # Parallel reduction over N float32
        # FLOPs: ~N-1 adds
        FLOPs = max(0, N - 1)
        # BYTES: read N*4 + write partials (one per grid block) + final result
        grid_blocks = I(r.get("grid_blocks"))
        partials = max(1, grid_blocks)
        BYTES = N*4 + partials*4 + 4
        working_set_bytes = N*4
        shared_bytes = max(1, block) * 4   # per-block sMem (one float per thread)
        mem_pattern = "shared_reduction"

    elif k == "dot_product":
        # dot of length N: N mul + N-1 add ~ 2N FLOPs
        FLOPs = 2 * N
        # reads x and y, writes partials (one per grid block) + final result
        grid_blocks = I(r.get("grid_blocks"))
        partials = max(1, grid_blocks)
        BYTES = 2*N*4 + partials*4 + 4
        working_set_bytes = 2*N*4
        shared_bytes = max(1, block) * 4
        mem_pattern = "shared_reduction"

    elif k == "histogram":
        # Assume 256 bins of uint32, one atomic per input
        FLOPs = 0
        BYTES = N * 8     # RMW ~ read+write a 4B counter
        working_set_bytes = N*4 + 256*4
        mem_pattern = "atomics_256bins"

    elif k == "atomic_hotspot":
        # many atomics to a single counter (iters per element)
        FLOPs = 0
        it = iters if iters > 0 else 1
        BYTES = N * it * 8
        working_set_bytes = 4
        mem_pattern = "atomics_hotspot"

    elif k == "naive_transpose":
        # read + write every element once
        # rows x cols matrix, float32
        if R>0 and C>0:
            elements = R*C
            FLOPs = 0
            BYTES = 2 * elements * 4
            working_set_bytes = BYTES
            mem_pattern = "transpose_naive"

    elif k == "shared_transpose":
        if R>0 and C>0:
            elements = R*C
            FLOPs = 0
            BYTES = 2 * elements * 4
            # per-block SMEM is small; keep per-block estimate tied to launch shmem
            shared_bytes = I(r.get("shmem"))  # per-block dynamic shmem
            working_set_bytes = BYTES
            mem_pattern = "transpose_tiled"

    elif k == "matmul_naive":
        # square matrices of size rows x cols assumed rows==cols for your runs
        n = R if R>0 else cols if cols>0 else 0
        if n == 0:
            n = I(r.get("matN"))  # fallback if present
        # FLOPs ~ 2*n^3 for one multiply-accumulate per product
        FLOPs = 2 * n * n * n
        # BYTES pessimistic: read A,B,C once
        BYTES = 4 * (n*n*3)
        working_set_bytes = BYTES
        mem_pattern = "mm_naive"

    elif k == "matmul_tiled":
        n = R if R>0 else cols if cols>0 else 0
        if n == 0:
            n = I(r.get("matN"))
        FLOPs = 2 * n * n * n
        BYTES = 4 * (n*n*3)          # still count gross global traffic
        # per-block shared memory is dynamic shmem for tiles
        shared_bytes = I(r.get("shmem"))
        working_set_bytes = BYTES
        mem_pattern = "mm_tiled"

    elif k == "conv2d_3x3":
        # Assume SAME padding => output R*C, 9 MACs each => 18 FLOPs per output
        if R>0 and C>0:
            out_elems = R * C
            FLOPs = out_elems * 18
            # read input + write output (ignore filter reuse detail)
            BYTES = (R*C*4) + (R*C*4)
            working_set_bytes = BYTES  # input + output
            mem_pattern = "stencil_3x3"
            conv_padding = "same"

    elif k == "conv2d_7x7":
        if R>0 and C>0:
            out_elems = R * C
            FLOPs = out_elems * (2 * 7 * 7)  # 49 MACs => 98 FLOPs per output
            BYTES = (R*C*4) + (R*C*4)
            working_set_bytes = BYTES  # input + output
            mem_pattern = "stencil_7x7"
            conv_padding = "same"

    elif k == "shared_bank_conflict":
        # No global traffic; keep BYTES=0; SMEM per-block only
        FLOPs = 0
        BYTES = 0
        shared_bytes = I(r.get("shmem"))   # per-block
        working_set_bytes = shared_bytes
        mem_pattern = "smem_bank_conflict"

    return FLOPs, BYTES, shared_bytes, mem_pattern, working_set_bytes, conv_padding


if __name__ == "__main__":
    runs_in, runs_out = sys.argv[1], sys.argv[2]

    rd = csv.DictReader(open(runs_in, newline=""))
    # Build output header = input header + our fields (unique)
    out_fields = list(rd.fieldnames) if rd.fieldnames else []

    # Ensure we have normalized size columns in the header
    first_missing_add(out_fields, "N")
    first_missing_add(out_fields, "rows")
    first_missing_add(out_fields, "cols")
    first_missing_add(out_fields, "size_kind")

    # Add static metrics
    for extra in [
        "FLOPs", "BYTES", "arithmetic_intensity",
        "working_set_bytes", "shared_bytes", "mem_pattern",
        "conv_padding"
    ]:
        first_missing_add(out_fields, extra)

    w = csv.DictWriter(open(runs_out, "w", newline=""), fieldnames=out_fields)
    w.writeheader()

    for r in rd:
        k = r.get("kernel", "")
        size_kind, N, rows, cols = size_from_row(r)

        FLOPs, BYTES, shared_bytes, mem_pattern, working_set_bytes, conv_padding = \
            per_kernel_counts(k, r, size_kind, N, rows, cols)

        # arithmetic intensity (handle 0 safely)
        ai = (float(FLOPs) / float(BYTES)) if BYTES > 0 else 0.0

        # Update row ONLY with fields declared in header
        updates = {
            "N": str(N),
            "rows": str(rows),
            "cols": str(cols),
            "size_kind": size_kind,
            "FLOPs": str(FLOPs),
            "BYTES": str(BYTES),
            "arithmetic_intensity": f"{ai:.6f}",
            "working_set_bytes": str(working_set_bytes),
            "shared_bytes": str(shared_bytes),
            "mem_pattern": mem_pattern,
            "conv_padding": conv_padding,
        }
        r.update({k:v for k,v in updates.items() if k in out_fields})
        w.writerow(r)


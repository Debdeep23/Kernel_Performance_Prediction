#!/usr/bin/env python3
import sys, csv, glob, math

# Usage: python3 scripts/aggregate_trials.py data/trials_*__2080ti.csv > data/runs_2080ti.csv

def I(x):
    try: return int(float(x))
    except: return 0

def F(x):
    try: return float(x)
    except: return 0.0

files = []
for arg in sys.argv[1:]:
    files.extend(glob.glob(arg))
if not files:
    print("kernel,args,regs,shmem,device_name,block_x,block_y,block_z,grid_x,grid_y,grid_z,warmup,reps,trials,mean_ms,std_ms,rows,cols,iters,block,grid_blocks")
    sys.exit(0)

# Group by stable shape/size identity (no free-form args)
groups = {}  # key -> list of time_ms
meta   = {}  # key -> representative row

def key_of(r):
    return (
        r.get("kernel",""),
        I(r.get("rows")), I(r.get("cols")),
        I(r.get("block_x")), I(r.get("block_y")), I(r.get("block_z")),
        I(r.get("grid_x")),  I(r.get("grid_y")),  I(r.get("grid_z")),
        I(r.get("regs")),    I(r.get("shmem")),
        I(r.get("warmup")),  I(r.get("reps")),
    )

for path in files:
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            k = key_of(r)
            t = F(r.get("time_ms"))
            if t <= 0:  # skip broken lines
                continue
            groups.setdefault(k, []).append(t)
            # keep a representative row’s metadata
            if k not in meta:
                # normalize some fields
                r["block"] = str(max(1, I(r.get("block_x"))*I(r.get("block_y"))*I(r.get("block_z"))))
                r["grid_blocks"] = str(max(1, I(r.get("grid_x"))*I(r.get("grid_y"))*I(r.get("grid_z"))))
                meta[k] = r

# Output
out_fields = [
    "kernel","args","regs","shmem","device_name",
    "block_x","block_y","block_z","grid_x","grid_y","grid_z",
    "warmup","reps","trials","mean_ms","std_ms",
    "rows","cols","block","grid_blocks"
]
w = csv.DictWriter(sys.stdout, fieldnames=out_fields)
w.writeheader()

for k, times in groups.items():
    r = meta[k]
    n = len(times)
    mean = sum(times)/n
    var  = sum((x-mean)**2 for x in times)/n if n>1 else 0.0
    std  = math.sqrt(var)

    out = {c: r.get(c,"") for c in out_fields}
    out["trials"]  = str(n)
    out["mean_ms"] = f"{mean:.6f}"
    out["std_ms"]  = f"{std:.6f}"
    w.writerow(out)


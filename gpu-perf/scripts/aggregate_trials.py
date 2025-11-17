import sys, csv, glob, statistics as stats

paths = sorted(glob.glob(sys.argv[1]))
rows = []
for p in paths:
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        rows.extend(rd)

def key(r):
    return (r["kernel"],
            r.get("block_x",""), r.get("block_y",""), r.get("block_z",""),
            r.get("grid_x",""),  r.get("grid_y",""),  r.get("grid_z",""))

groups = {}
for r in rows:
    groups.setdefault(key(r), []).append(r)

out = []
for k, rs in groups.items():
    times = [float(r["time_ms"]) for r in rs]
    regs  = [int(r.get("regs","0")) for r in rs if r.get("regs")]
    shmem = [int(r.get("shmem","0")) for r in rs if r.get("shmem")]
    devn  = [r.get("device_name","") for r in rs]

    r0 = rs[0]
    o = {
      "kernel": r0["kernel"],
      "args":   r0.get("args",""),
      "regs":   max(regs) if regs else 0,
      "shmem":  max(shmem) if shmem else 0,
      "device_name": devn[0] if devn else "",
      "block_x": r0.get("block_x",""), "block_y": r0.get("block_y",""), "block_z": r0.get("block_z",""),
      "grid_x":  r0.get("grid_x",""),  "grid_y":  r0.get("grid_y",""),  "grid_z":  r0.get("grid_z",""),
      "warmup":  r0.get("warmup",""), "reps": r0.get("reps",""),
      "trials":  len(times),
      "mean_ms": f"{stats.mean(times):.6f}",
      "std_ms":  f"{(stats.pstdev(times) if len(times)>1 else 0.0):.6f}",
    }
    # carry unified size fields if present in trial CSVs
    for fld in ("rows","cols","iters","block","grid_blocks"):
        if fld in r0: o[fld] = r0[fld]
    out.append(o)

w = csv.DictWriter(sys.stdout, fieldnames=list(out[0].keys()))
w.writeheader()
w.writerows(out)


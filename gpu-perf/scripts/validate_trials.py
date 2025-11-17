#!/usr/bin/env python3
import csv, glob, os, math

files = sorted(glob.glob("data/trials_*__2080ti.csv"))
if not files:
    raise SystemExit("No trials found (data/trials_*__2080ti.csv).")

required = ["kernel","args","regs","shmem","device_name",
            "block_x","block_y","block_z","grid_x","grid_y","grid_z",
            "warmup","reps","time_ms",
            "N","rows","cols","matN","H","W","iters","block","grid_blocks"]

kept, dropped = 0, 0
for path in files:
    with open(path, newline='') as f:
        rd = list(csv.DictReader(f))
    if not rd:
        print(f"[EMPTY] {path}")
        continue

    # keep header order but ensure all required fields exist
    fieldnames = list(rd[0].keys())
    for k in required:
        if k not in fieldnames:
            fieldnames.append(k)

    clean = []
    for r in rd:
        try:
            t = float(r.get("time_ms","0"))
            bx = int(r.get("block_x","0")); gx = int(r.get("grid_x","0"))
        except:
            t, bx, gx = 0.0, 0, 0

        ok = (t > 0.0) and (bx > 0) and (gx > 0) and r.get("kernel","")
        if ok:
            clean.append({k: r.get(k,"") for k in fieldnames})
            kept += 1
        else:
            dropped += 1

    with open(path, "w", newline='') as g:
        csv.DictWriter(g, fieldnames=fieldnames).writeheader()
        csv.DictWriter(g, fieldnames=fieldnames).writerows(clean)

print(f"[OK] validated trials: kept={kept} dropped={dropped}")


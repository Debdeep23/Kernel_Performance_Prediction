#!/usr/bin/env python3
import csv, sys, re

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

def warn(msg):
    print(msg)

def main():
    if len(sys.argv) != 2:
        print("usage: validate_csv.py <csv>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    issues = 0
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        need = ["kernel","args","regs","shmem","device_name",
                "block_x","block_y","block_z","grid_x","grid_y","grid_z",
                "warmup","reps","time_ms"]
        miss = [c for c in need if c not in rd.fieldnames]
        if miss:
            warn(f"{path}: missing fields: {', '.join(miss)}")
            issues += 1
            print(f"[FAIL] {path} had {issues} issues")
            return

        for i,r in enumerate(rd, start=2):
            k = r["kernel"]
            # block flat vs reported "block" if present
            bx,by,bz = I(r.get("block_x")), I(r.get("block_y")), I(r.get("block_z"))
            calc_block = max(1,bx)*max(1,by)*max(1,bz)
            if "block" in rd.fieldnames:
                rb = I(r.get("block"))
                if rb>0 and rb != calc_block:
                    warn(f"[{path}:{i}] {k}: block mismatch flat={calc_block} vs {rb}")

            # size family presence
            sizes = [I(r.get("N")), I(r.get("rows")), I(r.get("cols")), I(r.get("H")), I(r.get("W")), I(r.get("matN"))]
            if sum(1 for v in sizes if v>0) == 0:
                warn(f"[{path}:{i}] {k}: missing problem size family")
                issues += 1

            # zero/negative x/y/z are suspicious
            gx,gy,gz = I(r.get("grid_x")), I(r.get("grid_y")), I(r.get("grid_z"))
            if min(bx,by,bz,gx,gy,gz) <= 0:
                warn(f"[{path}:{i}] {k}: suspicious block/grid: bx={bx} by={by} bz={bz} gx={gx} gy={gy} gz={gz}")
                issues += 1

    print(("[OK] " if issues==0 else "[WARN] ") + f"{path} checked with {issues} issue(s)")

if __name__ == "__main__":
    main()


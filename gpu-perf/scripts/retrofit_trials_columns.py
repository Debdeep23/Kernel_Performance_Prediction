#!/usr/bin/env python3
"""
Retrofit a trials CSV to include a superset of columns used by the pipeline.
Missing columns are added and filled with neutral defaults (0 or empty).

Usage:
  python3 scripts/retrofit_trials_columns.py data/trials_<kernel>__<tag>.csv
"""

import csv
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

TARGET_FIELDS = [
    "kernel","args","regs","shmem","device_name",
    "block_x","block_y","block_z","grid_x","grid_y","grid_z",
    "warmup","reps","trials","time_ms",
    # size family
    "N","rows","cols","matN","H","W",
    # extras
    "iters","block","grid_blocks"
]

DEFAULTS = {
    "kernel":"", "args":"", "regs":"0", "shmem":"0", "device_name":"",
    "block_x":"0","block_y":"0","block_z":"0","grid_x":"0","grid_y":"0","grid_z":"0",
    "warmup":"0","reps":"1","trials":"1","time_ms":"0",
    "N":"0","rows":"0","cols":"0","matN":"0","H":"0","W":"0",
    "iters":"0","block":"0","grid_blocks":"0"
}

def retrofit(path: Path):
    with open(path, newline="") as f_in, NamedTemporaryFile("w", newline="", delete=False) as f_out:
        rd = csv.DictReader(f_in)
        # Build writer with full header
        w  = csv.DictWriter(f_out, fieldnames=TARGET_FIELDS)
        w.writeheader()

        # Stream rows, filling defaults for missing columns
        for row in rd:
            newrow = {k: row.get(k, DEFAULTS[k]) for k in TARGET_FIELDS}
            # also carry unknown columns if any, by ignoring them (not needed downstream)
            w.writerow(newrow)

    shutil.move(f_out.name, path)
    print(f"[OK] retrofitted {path} to include required columns")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: scripts/retrofit_trials_columns.py data/trials_<kernel>__<tag>.csv", file=sys.stderr)
        sys.exit(2)
    retrofit(Path(sys.argv[1]))


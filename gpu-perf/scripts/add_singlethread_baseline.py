#!/usr/bin/env python3
import sys, json, csv

RUNS_IN, CALIB_JSON, RUNS_OUT, WARP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

C = json.load(open(CALIB_JSON))
peak_gflops = float(C.get("sustained_compute_gflops", 0.0))
peak_gbps   = float(C.get("sustained_mem_bandwidth_gbps", 0.0))

peak_gflops_1 = peak_gflops / WARP if peak_gflops>0 else 0.0
peak_gbps_1   = peak_gbps   / WARP if peak_gbps>0   else 0.0

rd = csv.DictReader(open(RUNS_IN))

# Remove useless columns: args, device_name, block_x/y/z, grid_x/y/z, warmup, reps, trials, iters, conv_padding, gpu_cc_minor
exclude_cols = {"args", "device_name", "block_x", "block_y", "block_z", "grid_x", "grid_y", "grid_z",
                "warmup", "reps", "trials", "iters", "conv_padding", "gpu_cc_minor"}
fields = [f for f in rd.fieldnames if f not in exclude_cols] + ["T1_model_ms","speedup_model"]
w  = csv.DictWriter(open(RUNS_OUT,"w",newline=""), fieldnames=fields, extrasaction='ignore')
w.writeheader()

for r in rd:
    fl = float(r.get("FLOPs") or 0.0)
    by = float(r.get("BYTES") or 0.0)
    Tpar = float(r.get("mean_ms") or r.get("time_ms") or 0.0)

    if fl==0.0 and by==0.0:
        r["T1_model_ms"] = ""
        r["speedup_model"] = ""
        w.writerow(r); continue

    t_comp = (1000.0*fl/(peak_gflops_1*1e9)) if (fl>0 and peak_gflops_1>0) else 0.0
    t_mem  = (1000.0*by/(peak_gbps_1  *1e9)) if (by>0 and peak_gbps_1  >0) else 0.0
    T1 = max(t_comp, t_mem)
    r["T1_model_ms"] = f"{T1:.6f}"
    r["speedup_model"] = f"{(T1/Tpar):.2f}" if (T1>0 and Tpar>0) else ""
    w.writerow(r)


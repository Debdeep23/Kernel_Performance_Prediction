import sys, json, csv

runs_in, calib_json, runs_out, warp = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

C = json.load(open(calib_json))
peak_gflops = float(C["sustained_compute_gflops"])
peak_gbps   = float(C["sustained_mem_bandwidth_gbps"])

peak_gflops_1 = peak_gflops / warp if peak_gflops>0 else 0.0
peak_gbps_1   = peak_gbps   / warp if peak_gbps>0   else 0.0

rd = csv.DictReader(open(runs_in))
out_fields = rd.fieldnames + ["T1_model_ms","speedup_model"]
w = csv.DictWriter(open(runs_out,"w",newline=""), fieldnames=out_fields)
w.writeheader()

for r in rd:
    fl   = float(r.get("FLOPs") or 0.0)
    by   = float(r.get("BYTES") or 0.0)
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

print(f"[OK] wrote {runs_out}")


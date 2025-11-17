import sys, csv, re

def kv_from_file(path):
    kv = {}
    pat = re.compile(r'^([A-Za-z0-9_]+)=(.*)$')
    for ln in open(path):
        ln = ln.strip()
        m = pat.match(ln)
        if m:
            kv[m.group(1)] = m.group(2)
    return kv

def read_props(props_out):
    kv = kv_from_file(props_out)
    out = {}
    out["device_name"] = kv.get("name","")
    # split "major=7 minor=5"
    maj = re.search(r'\d+', kv.get("major","0"))
    mino= re.search(r'\d+', kv.get("minor","0"))
    out["cc_major"] = int(maj.group(0)) if maj else 0
    out["cc_minor"] = int(mino.group(0)) if mino else 0

    for k in ["multiProcessorCount","maxThreadsPerMultiProcessor","maxBlocksPerMultiProcessor",
              "regsPerMultiprocessor","sharedMemPerMultiprocessor","sharedMemPerBlockOptin",
              "maxThreadsPerBlock","warpSize","l2CacheSizeBytes"]:
        v = kv.get(k,"0")
        v = re.sub(r'[^0-9\-]','', v)
        out[k] = int(v) if v else 0
    return out

def pick(path, key):
    for ln in open(path):
        if ln.startswith(key+"="):
            try: return float(ln.split("=",1)[1])
            except: return 0.0
    return 0.0

if __name__ == "__main__":
    runs_in, props_out, stream_out, gemm_out, runs_out = sys.argv[1:6]
    P  = read_props(props_out)
    bw = pick(stream_out, "SUSTAINED_MEM_BW_GBPS")
    fl = pick(gemm_out,   "SUSTAINED_COMPUTE_GFLOPS")

    rd = csv.DictReader(open(runs_in))
    flds = rd.fieldnames + [
        "gpu_device_name","gpu_cc_major","gpu_cc_minor","gpu_sms",
        "gpu_max_threads_per_sm","gpu_max_blocks_per_sm","gpu_regs_per_sm",
        "gpu_shared_mem_per_sm","gpu_l2_bytes","gpu_warp_size",
        "sustained_mem_bandwidth_gbps","sustained_compute_gflops"
    ]
    w = csv.DictWriter(open(runs_out,"w",newline=""), fieldnames=flds)
    w.writeheader()

    for r in rd:
        r.update({
          "gpu_device_name": P["device_name"],
          "gpu_cc_major": P["cc_major"],
          "gpu_cc_minor": P["cc_minor"],
          "gpu_sms": P["multiProcessorCount"],
          "gpu_max_threads_per_sm": P["maxThreadsPerMultiProcessor"],
          "gpu_max_blocks_per_sm": P["maxBlocksPerMultiProcessor"],
          "gpu_regs_per_sm": P["regsPerMultiprocessor"],
          "gpu_shared_mem_per_sm": P["sharedMemPerMultiprocessor"],
          "gpu_l2_bytes": P["l2CacheSizeBytes"],
          "gpu_warp_size": P["warpSize"],
          "sustained_mem_bandwidth_gbps": f"{bw:.2f}",
          "sustained_compute_gflops": f"{fl:.2f}",
        })
        w.writerow(r)

    print(f"[OK] wrote {runs_out}")


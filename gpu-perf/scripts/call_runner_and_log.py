#!/usr/bin/env python3
import sys, os, csv, re, subprocess, shlex, pathlib

def parse_args_kv(argstr):
    kv = {}
    toks = shlex.split(argstr)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t.lstrip("-")
            v = None
            if i+1 < len(toks) and not toks[i+1].startswith("--"):
                v = toks[i+1]; i += 1
            kv[k] = v if v is not None else "1"
        i += 1
    return kv

def pick_size_family(kv):
    def I(x): 
        try: return int(x)
        except: return 0
    rows = I(kv.get("rows",0)); cols = I(kv.get("cols",0))
    N    = I(kv.get("N",0));    matN = I(kv.get("matN",0))
    H    = I(kv.get("H",0));    W    = I(kv.get("W",0))
    out = {"N":"", "rows":"", "cols":"", "matN":"", "H":"", "W":""}
    if rows>0 and cols>0:
        out["rows"]=str(rows); out["cols"]=str(cols)
    elif N>0:
        out["N"]=str(N)
    elif matN>0:
        out["matN"]=str(matN)
    elif H>0 and W>0:
        out["H"]=str(H); out["W"]=str(W)
    return out

def parse_tuple(line, tag):
    m = re.search(rf"{tag}=\((\d+)(?:,(\d+))?(?:,(\d+))?\)", line)
    if not m: return (1,1,1)  # safe default
    parts = [int(x) for x in m.groups() if x is not None]
    if len(parts)==1: return (parts[0],1,1)
    if len(parts)==2: return (parts[0],parts[1],1)
    return (parts[0],parts[1],parts[2])

def parse_time_ms(line):
    m = re.search(r"time_ms=([0-9]*\.?[0-9]+)", line)
    return float(m.group(1)) if m else 0.0

def parse_regs_shmem(line, regs0, shmem0):
    r = regs0; s = shmem0
    m = re.search(r"regs=(\d+)", line);   r = int(m.group(1)) if m else r
    m = re.search(r"shmem=(\d+)", line);  s = int(m.group(1)) if m else s
    return r, s

def main():
    # CLI
    cfg = {"kernel":None,"args":"","regs":"0","shmem":"0","device":"","outfile":None,"trials":"10"}
    argv = sys.argv[1:]
    i=0
    while i < len(argv):
        if argv[i].startswith("--") and i+1 < len(argv):
            k = argv[i][2:]; v = argv[i+1]
            if k in cfg: cfg[k]=v
            i += 2
        else:
            i += 1
    if not cfg["kernel"] or not cfg["outfile"]:
        print("call_runner_and_log.py: --kernel and --outfile are required", file=sys.stderr)
        sys.exit(2)

    kernel = cfg["kernel"]
    argstr = cfg["args"]
    regs   = int(cfg["regs"])
    shmem  = int(cfg["shmem"])
    device = cfg["device"]
    trials = int(cfg["trials"])
    outfile= cfg["outfile"]

    pathlib.Path(os.path.dirname(outfile)).mkdir(parents=True, exist_ok=True)

    # run runner
    cmd = ["bin/runner", kernel] + shlex.split(argstr)
    try:
        out = subprocess.checkoutput(cmd, stderr=subprocess.STDOUT, text=True)
    except AttributeError:
        # Py3.7 compatibility
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    except subprocess.CalledProcessError as e:
        print(e.output, file=sys.stderr); raise

    # grab the summary line
    line = ""
    for ln in out.splitlines():
        if "block=(" in ln and "grid=(" in ln and "time_ms=" in ln:
            line = ln.strip()
    if not line and out.splitlines():
        line = out.splitlines()[-1].strip()

    bx,by,bz = parse_tuple(line, "block")
    gx,gy,gz = parse_tuple(line, "grid")
    regs, shmem = parse_regs_shmem(line, regs, shmem)
    t_ms = parse_time_ms(line)

    kv = parse_args_kv(argstr)
    warmup = int(kv.get("warmup","0") or 0)
    reps   = int(kv.get("reps","0") or 0)
    size   = pick_size_family(kv)

    threads_per_block = bx*by*bz
    grid_blocks       = gx*gy*gz

    fields = [
      "kernel","args","regs","shmem","device_name",
      "block_x","block_y","block_z","grid_x","grid_y","grid_z",
      "warmup","reps","trials","time_ms",
      "N","rows","cols","matN","H","W",
      "block","grid_blocks"
    ]
    row = {
      "kernel": kernel, "args": argstr, "regs": str(regs), "shmem": str(shmem),
      "device_name": device,
      "block_x": str(bx), "block_y": str(by), "block_z": str(bz),
      "grid_x": str(gx), "grid_y": str(gy), "grid_z": str(gz),
      "warmup": str(warmup), "reps": str(reps), "trials": str(trials),
      "time_ms": f"{t_ms:.6f}",
      "N": size["N"], "rows": size["rows"], "cols": size["cols"],
      "matN": size["matN"], "H": size["H"], "W": size["W"],
      "block": str(threads_per_block), "grid_blocks": str(grid_blocks)
    }

    write_header = not os.path.exists(outfile)
    with open(outfile, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        w.writerow(row)

    print(f"[OK] {kernel}: time_ms={row['time_ms']} block=({bx},{by},{bz}) grid=({gx},{gy},{gz}) → {outfile}")

if __name__ == "__main__":
    main()


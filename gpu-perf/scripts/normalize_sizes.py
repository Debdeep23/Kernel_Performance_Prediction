import sys, csv

inp  = sys.argv[1]
outp = sys.argv[2]

def I(x):
    try: return int(x)
    except: return 0

rd = csv.DictReader(open(inp))
base = [f for f in rd.fieldnames if f not in ("N","matN","H","W")]
for need in ("rows","cols","size_kind"):
    if need not in base: base.append(need)

w = csv.DictWriter(open(outp,"w",newline=""), fieldnames=base)
w.writeheader()

for r in rd:
    N    = I(r.get("N",""))
    matN = I(r.get("matN",""))
    H    = I(r.get("H",""))
    W    = I(r.get("W",""))
    rows = I(r.get("rows",""))
    cols = I(r.get("cols",""))

    kind = "unknown"
    if matN > 0:
        rows, cols, kind = matN, matN, "matN"
    elif H > 0 or W > 0:
        rows = H if H>0 else rows
        cols = W if W>0 else cols
        if rows==0 and cols>0: rows=1
        if cols==0 and rows>0: cols=1
        kind = "HW"
    elif rows > 0 or cols > 0:
        if rows==0 and cols>0: rows=1
        if cols==0 and rows>0: cols=1
        kind = "rows_cols"
    elif N > 0:
        rows, cols, kind = N, 1, "N_as_rows1D"
    else:
        rows = cols = 0
        kind = "none"

    r["rows"] = rows
    r["cols"] = cols
    r["size_kind"] = kind
    for k in ("N","matN","H","W"):
        if k in r: del r[k]
    w.writerow(r)

print(f"[OK] normalized sizes -> {outp}")


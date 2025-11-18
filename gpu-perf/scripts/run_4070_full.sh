#!/usr/bin/env bash
set -euo pipefail

echo "=== 0) Clean old 4070 data ==="
rm -f data/trials_*__4070.csv data/runs_4070.csv data/runs_4070_final.csv

echo "=== 1) Build calibration tools ==="
mkdir -p bin data
cd calibration

echo "Building props..."
nvcc -o ../bin/props_4070 props.cu

echo "Building stream_like..."
nvcc -o ../bin/stream_like_4070 stream_like.cu

echo "Building gemm_cublas..."
nvcc -o ../bin/gemm_cublas_4070 gemm_cublas.cu -lcublas

cd ..

echo "=== 2) Run calibration benchmarks ==="
echo "Capturing GPU properties..."
bin/props_4070 > data/props_4070.out

echo "Measuring memory bandwidth..."
bin/stream_like_4070 > data/stream_like_4070.out

echo "Measuring compute throughput..."
bin/gemm_cublas_4070 > data/gemm_cublas_4070.out

echo "=== 3) Verify calibration files ==="
echo "Props file:"
head -3 data/props_4070.out
echo ""
echo "Stream file:"
tail -1 data/stream_like_4070.out
echo ""
echo "GEMM file:"
tail -1 data/gemm_cublas_4070.out
echo ""

echo "=== 4) Generate device calibration JSON ==="
python3 - <<'PYEOF'
import re, json

# Read props
props = {}
with open('data/props_4070.out') as f:
    for ln in f:
        if '=' in ln:
            k, v = ln.strip().split('=', 1)
            props[k] = v

# Read stream
stream_bw = 0.0
with open('data/stream_like_4070.out') as f:
    for ln in f:
        if ln.startswith('SUSTAINED_MEM_BW_GBPS='):
            stream_bw = float(ln.split('=')[1])

# Read gemm
gemm_gflops = 0.0
with open('data/gemm_cublas_4070.out') as f:
    for ln in f:
        if ln.startswith('SUSTAINED_COMPUTE_GFLOPS='):
            gemm_gflops = float(ln.split('=')[1])

# Create calibration JSON
calib = {
    "device_name": props.get("name", "RTX 4070"),
    "warp_size": int(props.get("warpSize", "32")),
    "sm_count": int(props.get("multiProcessorCount", "46")),
    "sustained_mem_bandwidth_gbps": stream_bw,
    "sustained_compute_gflops": gemm_gflops,
    "sm_limits": {
        "max_threads_per_sm": int(props.get("maxThreadsPerMultiProcessor", "1536")),
        "max_blocks_per_sm": int(props.get("maxBlocksPerMultiProcessor", "24")),
        "registers_per_sm": int(props.get("regsPerMultiprocessor", "65536")),
        "shared_mem_per_sm_bytes": int(props.get("sharedMemPerMultiprocessor", "102400"))
    }
}

with open('data/device_calibration_4070.json', 'w') as f:
    json.dump(calib, f, indent=2)

print("Created data/device_calibration_4070.json")
PYEOF

echo ""
echo "=== 5) Trials (10 per kernel with multiple sizes) ==="
scripts/gen_trials_4070.sh

echo "=== 6) Validate trials ==="
for f in data/trials_*__4070.csv; do
  if [ -f "$f" ]; then
    python3 scripts/validate_csv.py "$f" || echo "Warning: validation failed for $f"
  fi
done

echo "=== 7) Generate final dataset (with iters, normalized sizes, and all metrics) ==="
python3 scripts/make_final_4070.py

echo "=== 8) Peek at final dataset ==="
head -5 data/runs_4070_final.csv

echo ""
echo "=========================================="
echo "✓ RTX 4070 pipeline complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - data/props_4070.out (GPU properties)"
echo "  - data/stream_like_4070.out (memory bandwidth calibration)"
echo "  - data/gemm_cublas_4070.out (compute calibration)"
echo "  - data/device_calibration_4070.json (device calibration summary)"
echo "  - data/trials_*__4070.csv (16 raw trial files)"
echo "  - data/runs_4070_final.csv (final dataset with iters and all metrics)"
echo ""
TOTAL_ROWS=$(wc -l < data/runs_4070_final.csv)
TOTAL_COLS=$(head -1 data/runs_4070_final.csv | tr ',' '\n' | wc -l)
echo "Final dataset: data/runs_4070_final.csv"
echo "  Rows: $TOTAL_ROWS"
echo "  Columns: $TOTAL_COLS"
echo ""

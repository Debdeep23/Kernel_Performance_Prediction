#!/usr/bin/env bash
set -euo pipefail

# TITAN V Benchmarking Script
#
# IMPORTANT: TITAN V requires CUDA 11.x or older (sm_70 support)
# CUDA 13.0+ dropped Volta support
#
# Before running:
#   module load cuda-11.4
#   nvcc --version  # Verify CUDA 11.4 is loaded
#
# Allow GPU selection via environment variable
# Usage: CUDA_VISIBLE_DEVICES=0 ./scripts/run_titanv_full.sh
# or:    GPU_ID=1 ./scripts/run_titanv_full.sh
if [ -n "${GPU_ID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "=== Using GPU $GPU_ID ==="
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "=== Using GPU(s): $CUDA_VISIBLE_DEVICES ==="
else
    echo "=== Using default GPU (set GPU_ID=N or CUDA_VISIBLE_DEVICES=N to select specific GPU) ==="
fi

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

echo ""
echo "=== 0) Clean old TITAN V data ==="
rm -f data/trials_*__titanv.csv data/runs_titanv*.csv
rm -f data/device_calibration_titanv.json
rm -f data/props_titanv.out data/stream_like_titanv.out data/gemm_cublas_titanv.out
rm -f data/ptxas_titanv.log

echo "=== 1) Build calibration tools ==="
mkdir -p bin data
cd calibration

# Force rebuild by removing old binaries
rm -f ../bin/props_titanv ../bin/stream_like_titanv ../bin/gemm_cublas_titanv

echo "Building props (sm_70 for Volta TITAN V)..."
nvcc -arch=sm_70 -o ../bin/props_titanv props.cu

echo "Building stream_like (sm_70 for Volta TITAN V)..."
nvcc -arch=sm_70 -o ../bin/stream_like_titanv stream_like.cu

echo "Building gemm_cublas (sm_70 for Volta TITAN V)..."
nvcc -arch=sm_70 -o ../bin/gemm_cublas_titanv gemm_cublas.cu -lcublas

cd ..

echo "=== 2) Run calibration benchmarks ==="
echo "Capturing GPU properties..."
bin/props_titanv > data/props_titanv.out

echo "Measuring memory bandwidth..."
bin/stream_like_titanv > data/stream_like_titanv.out

echo "Measuring compute throughput..."
bin/gemm_cublas_titanv > data/gemm_cublas_titanv.out

echo "=== 3) Verify calibration files ==="
echo "Props file:"
head -3 data/props_titanv.out
echo ""
echo "Stream file:"
tail -1 data/stream_like_titanv.out
echo ""
echo "GEMM file:"
tail -1 data/gemm_cublas_titanv.out
echo ""

echo "=== 4) Generate device calibration JSON ==="
python3 - <<'PYEOF'
import re, json, sys

# Read props
props = {}
with open('data/props_titanv.out') as f:
    for ln in f:
        if '=' in ln:
            k, v = ln.strip().split('=', 1)
            props[k] = v

# Read stream
stream_bw = 0.0
with open('data/stream_like_titanv.out') as f:
    for ln in f:
        if ln.startswith('SUSTAINED_MEM_BW_GBPS='):
            stream_bw = float(ln.split('=')[1])

# Read gemm
gemm_gflops = 0.0
with open('data/gemm_cublas_titanv.out') as f:
    for ln in f:
        if ln.startswith('SUSTAINED_COMPUTE_GFLOPS='):
            gemm_gflops = float(ln.split('=')[1])

# Validate calibration values for TITAN V
print(f"Validating calibration values...")
print(f"  Memory bandwidth: {stream_bw:.2f} GB/s")
print(f"  Compute throughput: {gemm_gflops:.2f} GFLOPS")

errors = []
if not (400 < stream_bw < 800):
    errors.append(f"Memory bandwidth {stream_bw:.2f} GB/s is outside expected range (400-800 GB/s)")
if not (8000 < gemm_gflops < 20000):
    errors.append(f"Compute throughput {gemm_gflops:.2f} GFLOPS is outside expected range (8000-20000 GFLOPS)")

if errors:
    print("\n❌ ERROR: Calibration values are suspicious!")
    for err in errors:
        print(f"  - {err}")
    print("\nExpected values for TITAN V:")
    print("  - Memory bandwidth: ~550-650 GB/s")
    print("  - Compute: ~13000-15000 GFLOPS (FP32)")
    print("\nPlease check:")
    print("  1. data/stream_like_titanv.out")
    print("  2. data/gemm_cublas_titanv.out")
    print("  3. Are you running on the correct GPU?")
    sys.exit(1)

print("✓ Calibration values look reasonable")

# Create calibration JSON
calib = {
    "device_name": props.get("name", "NVIDIA TITAN V"),
    "warp_size": int(props.get("warpSize", "32")),
    "sm_count": int(props.get("multiProcessorCount", "80")),
    "sustained_mem_bandwidth_gbps": stream_bw,
    "sustained_compute_gflops": gemm_gflops,
    "sm_limits": {
        "max_threads_per_sm": int(props.get("maxThreadsPerMultiProcessor", "2048")),
        "max_blocks_per_sm": int(props.get("maxBlocksPerMultiProcessor", "32")),
        "registers_per_sm": int(props.get("regsPerMultiprocessor", "65536")),
        "shared_mem_per_sm_bytes": int(props.get("sharedMemPerMultiprocessor", "98304"))
    }
}

with open('data/device_calibration_titanv.json', 'w') as f:
    json.dump(calib, f, indent=2)

print("Created data/device_calibration_titanv.json")
PYEOF

echo ""
echo "=== 5) Trials (10 per kernel with multiple sizes) ==="
scripts/gen_trials_titanv.sh

echo "=== 6) Validate trials ==="
for f in data/trials_*__titanv.csv; do
  if [ -f "$f" ]; then
    python3 scripts/validate_csv.py "$f" || echo "Warning: validation failed for $f"
  fi
done

echo "=== 7) Aggregate → runs_titanv.csv ==="
python3 scripts/aggregate_trials.py data/trials_*__titanv.csv > data/runs_titanv.csv

echo "=== 8) Static counts → runs_titanv_with_counts.csv ==="
python3 scripts/static_counts.py data/runs_titanv.csv data/runs_titanv_with_counts.csv

echo "=== 9) Enrich with GPU metrics → runs_titanv_enriched.csv ==="
python3 scripts/enrich_with_gpu_metrics.py \
  data/runs_titanv_with_counts.csv \
  data/props_titanv.out \
  data/stream_like_titanv.out \
  data/gemm_cublas_titanv.out \
  data/runs_titanv_enriched.csv

echo "=== 10) Add single-thread baseline → runs_titanv_final.csv ==="
python3 scripts/add_singlethread_baseline.py \
  data/runs_titanv_enriched.csv \
  data/device_calibration_titanv.json \
  data/runs_titanv_final.csv \
  32

echo "=== 11) Peek ==="
head -5 data/runs_titanv_final.csv

echo ""
echo "=========================================="
echo "✓ TITAN V pipeline complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - data/props_titanv.out"
echo "  - data/stream_like_titanv.out"
echo "  - data/gemm_cublas_titanv.out"
echo "  - data/device_calibration_titanv.json"
echo "  - data/trials_*__titanv.csv (16 files)"
echo "  - data/runs_titanv.csv"
echo "  - data/runs_titanv_with_counts.csv"
echo "  - data/runs_titanv_enriched.csv"
echo "  - data/runs_titanv_final.csv (includes iters column)"
echo ""
TOTAL_ROWS=$(wc -l < data/runs_titanv_final.csv)
TOTAL_COLS=$(head -1 data/runs_titanv_final.csv | tr ',' '\n' | wc -l)
echo "Final dataset: data/runs_titanv_final.csv"
echo "  Rows: $TOTAL_ROWS"
echo "  Columns: $TOTAL_COLS"
echo ""

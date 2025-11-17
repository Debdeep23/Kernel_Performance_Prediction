#!/usr/bin/env bash
set -euo pipefail

echo "=== 0) Clean binaries and CSVs ==="
mkdir -p bin data
rm -f bin/props bin/stream_like bin/gemm_cublas bin/runner
rm -f data/trials_*__*.csv data/runs_*.csv data/*_2080ti.out

echo "=== 1) Build calibration tools (sm_75) ==="
nvcc -O3 -arch=sm_75 -o bin/props         calibration/props.cu
nvcc -O3 -arch=sm_75 -o bin/stream_like   calibration/stream_like.cu
nvcc -O3 -arch=sm_75 -lcublas -o bin/gemm_cublas calibration/gemm_cublas.cu

echo "=== 2) Collect props + sustained ceilings ==="
bin/props       > data/props_2080ti.out
bin/stream_like > data/stream_like_2080ti.out
bin/gemm_cublas > data/gemm_cublas_2080ti.out

# Ensure SUSTAINED_* lines exist
bw=$(awk -F'=' '/GBps=/{print $NF}' data/stream_like_2080ti.out | sort -nr | head -1)
fl=$(awk -F'=' '/GFLOPS=/{print $NF}' data/gemm_cublas_2080ti.out | sort -nr | head -1)
grep -q '^SUSTAINED_MEM_BW_GBPS=' data/stream_like_2080ti.out || echo "SUSTAINED_MEM_BW_GBPS=$bw" >> data/stream_like_2080ti.out
grep -q '^SUSTAINED_COMPUTE_GFLOPS=' data/gemm_cublas_2080ti.out || echo "SUSTAINED_COMPUTE_GFLOPS=$fl" >> data/gemm_cublas_2080ti.out

echo "=== 3) Build runner (sm_75) ==="
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_75 -DTILE=32 \
  -o bin/runner runner/main.cu 2> data/ptxas_2080ti.log

echo "=== 4) Run trials (10 each) on 2080 Ti ==="
chmod +x scripts/run_trials.sh
# keep your full list; here’s the usual set
scripts/run_trials.sh vector_add            "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100" 12 0 10
scripts/run_trials.sh saxpy                 "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100" 12 0 10
scripts/run_trials.sh strided_copy_8        "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  8 0 10
scripts/run_trials.sh naive_transpose       "--rows 2048    --cols 2048          --warmup 20 --reps 100"    8 0 10
scripts/run_trials.sh shared_transpose      "--rows 2048    --cols 2048          --warmup 20 --reps 100"   10 4224 10
scripts/run_trials.sh matmul_tiled          "--rows 512     --cols 512           --warmup 10 --reps 50"    37 8192 10
scripts/run_trials.sh matmul_naive          "--rows 512     --cols 512           --warmup 10 --reps 50"    40 0    10
scripts/run_trials.sh reduce_sum            "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100" 10 1024 10
scripts/run_trials.sh dot_product           "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100" 15 1024 10
scripts/run_trials.sh histogram             "--rows 1048576 --cols 1   --block 256 --warmup 10  --reps 50" 10 1024 10
scripts/run_trials.sh conv2d_3x3            "--rows 1024    --cols 1024          --warmup 10 --reps 50"    30 0    10
scripts/run_trials.sh conv2d_7x7            "--rows 1024    --cols 1024          --warmup 10 --reps 50"    40 0    10
scripts/run_trials.sh random_access         "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100" 10 0    10
scripts/run_trials.sh vector_add_divergent  "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100" 15 0    10
scripts/run_trials.sh shared_bank_conflict  "--rows 1       --cols 1             --warmup 20 --reps 100"  206 4096 10
scripts/run_trials.sh atomic_hotspot        "--rows 1048576 --cols 1   --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10

echo "=== 5) Aggregate trials → runs_2080ti.csv ==="
python3 scripts/aggregate_trials.py "data/trials_*__2080ti.csv" > data/runs_2080ti.csv

echo "=== 6) Normalize sizes to rows,cols ==="
python3 scripts/normalize_sizes.py data/runs_2080ti.csv data/runs_2080ti_norm.csv

echo "=== 7) Attach static counts (FLOPs/BYTES/AI/etc.) ==="
python3 scripts/static_counts.py data/runs_2080ti_norm.csv data/runs_2080ti_with_counts.csv
head -5 data/runs_2080ti_with_counts.csv || true

echo "=== 8) Enrich with GPU metrics + sustained ceilings ==="
python3 scripts/enrich_with_gpu_metrics.py \
  data/runs_2080ti_with_counts.csv \
  data/props_2080ti.out \
  data/stream_like_2080ti.out \
  data/gemm_cublas_2080ti.out \
  data/runs_2080ti_enriched.csv
head -5 data/runs_2080ti_enriched.csv || true

echo "=== 9) Add single-thread baseline (warp=32) → final CSV ==="
python3 scripts/add_singlethread_baseline.py \
  data/runs_2080ti_enriched.csv \
  data/device_calibration_2080ti.json \
  data/runs_2080ti_final.csv \
  32
head -5 data/runs_2080ti_final.csv || true

echo "=== DONE: data/runs_2080ti_final.csv ==="


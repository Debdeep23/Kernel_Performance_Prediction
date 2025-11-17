#!/usr/bin/env bash
set -euo pipefail

echo "=== 0) Clean ==="
rm -f data/trials_*__2080ti.csv data/runs_2080ti*.csv

echo "=== 1) Trials with MULTIPLE SIZES (10 per kernel config) ==="
# This enhanced script runs each kernel with different problem sizes to generate more data

# ========== vector_add (1D kernels) ==========
# Test different N sizes and block sizes
echo "Running vector_add variants..."
scripts/run_trials.sh vector_add  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh vector_add  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh vector_add  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh vector_add  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh vector_add  "--rows 4194304 --cols 1 --block 256 --warmup 20 --reps 100"  12 0 10

# ========== saxpy (1D kernels) ==========
echo "Running saxpy variants..."
scripts/run_trials.sh saxpy  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh saxpy  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh saxpy  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh saxpy  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  12 0 10
scripts/run_trials.sh saxpy  "--rows 4194304 --cols 1 --block 256 --warmup 20 --reps 100"  12 0 10

# ========== strided_copy_8 (1D kernels) ==========
echo "Running strided_copy_8 variants..."
scripts/run_trials.sh strided_copy_8  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  8 0 10
scripts/run_trials.sh strided_copy_8  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  8 0 10
scripts/run_trials.sh strided_copy_8  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  8 0 10
scripts/run_trials.sh strided_copy_8  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  8 0 10

# ========== Transpose (2D kernels) ==========
# Test different matrix sizes
echo "Running transpose variants..."
scripts/run_trials.sh naive_transpose    "--rows 512  --cols 512  --warmup 20 --reps 100"  16 0 10
scripts/run_trials.sh naive_transpose    "--rows 1024 --cols 1024 --warmup 20 --reps 100"  16 0 10
scripts/run_trials.sh naive_transpose    "--rows 2048 --cols 2048 --warmup 20 --reps 100"  16 0 10
scripts/run_trials.sh naive_transpose    "--rows 4096 --cols 4096 --warmup 20 --reps 100"  16 0 10

scripts/run_trials.sh shared_transpose   "--rows 512  --cols 512  --warmup 20 --reps 100"  32 4224 10
scripts/run_trials.sh shared_transpose   "--rows 1024 --cols 1024 --warmup 20 --reps 100"  32 4224 10
scripts/run_trials.sh shared_transpose   "--rows 2048 --cols 2048 --warmup 20 --reps 100"  32 4224 10
scripts/run_trials.sh shared_transpose   "--rows 4096 --cols 4096 --warmup 20 --reps 100"  32 4224 10

# ========== Matrix Multiply ==========
# Test different matrix sizes (computationally intensive)
echo "Running matmul variants..."
scripts/run_trials.sh matmul_naive  "--rows 256 --cols 256 --warmup 10 --reps 50"  40 0    10
scripts/run_trials.sh matmul_naive  "--rows 512 --cols 512 --warmup 10 --reps 50"  40 0    10
scripts/run_trials.sh matmul_naive  "--rows 1024 --cols 1024 --warmup 10 --reps 50"  40 0    10

scripts/run_trials.sh matmul_tiled  "--rows 256 --cols 256 --warmup 10 --reps 50"  37 8192 10
scripts/run_trials.sh matmul_tiled  "--rows 512 --cols 512 --warmup 10 --reps 50"  37 8192 10
scripts/run_trials.sh matmul_tiled  "--rows 1024 --cols 1024 --warmup 10 --reps 50"  37 8192 10

# ========== Reductions ==========
echo "Running reduction variants..."
scripts/run_trials.sh reduce_sum  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  10 1024 10
scripts/run_trials.sh reduce_sum  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  10 1024 10
scripts/run_trials.sh reduce_sum  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  10 1024 10
scripts/run_trials.sh reduce_sum  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  10 1024 10

scripts/run_trials.sh dot_product  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  15 1024 10
scripts/run_trials.sh dot_product  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  15 1024 10
scripts/run_trials.sh dot_product  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  15 1024 10
scripts/run_trials.sh dot_product  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  15 1024 10

# ========== Histogram ==========
echo "Running histogram variants..."
scripts/run_trials.sh histogram  "--rows 262144  --cols 1 --block 128 --warmup 10 --reps 50"  10 1024 10
scripts/run_trials.sh histogram  "--rows 524288  --cols 1 --block 256 --warmup 10 --reps 50"  10 1024 10
scripts/run_trials.sh histogram  "--rows 1048576 --cols 1 --block 256 --warmup 10 --reps 50"  10 1024 10
scripts/run_trials.sh histogram  "--rows 2097152 --cols 1 --block 512 --warmup 10 --reps 50"  10 1024 10

# ========== Convolutions (2D) ==========
echo "Running conv2d variants..."
scripts/run_trials.sh conv2d_3x3  "--rows 512  --cols 512  --warmup 10 --reps 50"  30 0 10
scripts/run_trials.sh conv2d_3x3  "--rows 1024 --cols 1024 --warmup 10 --reps 50"  30 0 10
scripts/run_trials.sh conv2d_3x3  "--rows 2048 --cols 2048 --warmup 10 --reps 50"  30 0 10

scripts/run_trials.sh conv2d_7x7  "--rows 512  --cols 512  --warmup 10 --reps 50"  40 0 10
scripts/run_trials.sh conv2d_7x7  "--rows 1024 --cols 1024 --warmup 10 --reps 50"  40 0 10
scripts/run_trials.sh conv2d_7x7  "--rows 2048 --cols 2048 --warmup 10 --reps 50"  40 0 10

# ========== Memory Access Patterns ==========
echo "Running memory access pattern variants..."
scripts/run_trials.sh random_access  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  10 0 10
scripts/run_trials.sh random_access  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  10 0 10
scripts/run_trials.sh random_access  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  10 0 10
scripts/run_trials.sh random_access  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  10 0 10

scripts/run_trials.sh vector_add_divergent  "--rows 262144  --cols 1 --block 128 --warmup 20 --reps 100"  15 0 10
scripts/run_trials.sh vector_add_divergent  "--rows 524288  --cols 1 --block 256 --warmup 20 --reps 100"  15 0 10
scripts/run_trials.sh vector_add_divergent  "--rows 1048576 --cols 1 --block 256 --warmup 20 --reps 100"  15 0 10
scripts/run_trials.sh vector_add_divergent  "--rows 2097152 --cols 1 --block 512 --warmup 20 --reps 100"  15 0 10

# ========== Atomics ==========
echo "Running atomic variants..."
scripts/run_trials.sh atomic_hotspot  "--rows 262144  --cols 1 --block 128 --iters 50  --warmup 10 --reps 50"  7 0 10
scripts/run_trials.sh atomic_hotspot  "--rows 524288  --cols 1 --block 256 --iters 100 --warmup 10 --reps 50"  7 0 10
scripts/run_trials.sh atomic_hotspot  "--rows 1048576 --cols 1 --block 256 --iters 100 --warmup 10 --reps 50"  7 0 10
scripts/run_trials.sh atomic_hotspot  "--rows 2097152 --cols 1 --block 512 --iters 200 --warmup 10 --reps 50"  7 0 10

# ========== Shared Memory Bank Conflict (only one config) ==========
echo "Running shared_bank_conflict..."
scripts/run_trials.sh shared_bank_conflict  "--warmup 20 --reps 100"  206 4096 10

echo "=== 2) Validate trials ==="
for f in data/trials_*__2080ti.csv; do
  scripts/validate_csv.py "$f"
done

echo "=== 3) Aggregate → runs_2080ti.csv ==="
python3 scripts/aggregate_trials.py data/trials_*__2080ti.csv > data/runs_2080ti.csv

echo "=== 4) Static counts → runs_2080ti_with_counts.csv ==="
python3 scripts/static_counts.py data/runs_2080ti.csv data/runs_2080ti_with_counts.csv

echo "=== 5) Enrich with GPU metrics → runs_2080ti_enriched.csv ==="
python3 scripts/enrich_with_gpu_metrics.py \
  data/runs_2080ti_with_counts.csv \
  data/props_2080ti.out \
  data/stream_like_2080ti.out \
  data/gemm_cublas_2080ti.out \
  data/runs_2080ti_enriched.csv

echo "=== 6) Add single-thread baseline → runs_2080ti_final.csv ==="
python3 scripts/add_singlethread_baseline.py \
  data/runs_2080ti_enriched.csv \
  data/device_calibration_2080ti.json \
  data/runs_2080ti_final.csv \
  32

echo "=== 7) Peek final dataset ==="
wc -l data/runs_2080ti_final.csv
head -5 data/runs_2080ti_final.csv

echo "=== DONE! Final dataset has $(tail -n +2 data/runs_2080ti_final.csv | wc -l) rows ==="

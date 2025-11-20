#!/usr/bin/env bash
set -euo pipefail

echo "=== wipe old 2080ti trials ==="
rm -f data/trials_*__2080ti.csv

echo "=== build runner for sm_75 (Turing / 2080 Ti) ==="
mkdir -p bin data
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_75 -DTILE=32 \
  -o bin/runner runner/main.cu 2> data/ptxas_2080ti.log
test -x bin/runner

chmod +x scripts/run_trials.sh

echo "=== Running vector_add with multiple sizes ==="
scripts/run_trials.sh vector_add "--N 262144 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti
scripts/run_trials.sh vector_add "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti
scripts/run_trials.sh vector_add "--N 4194304 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti
scripts/run_trials.sh vector_add "--N 16777216 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti

echo "=== Running saxpy with multiple sizes ==="
scripts/run_trials.sh saxpy "--N 262144 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti
scripts/run_trials.sh saxpy "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti
scripts/run_trials.sh saxpy "--N 4194304 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti
scripts/run_trials.sh saxpy "--N 16777216 --block 256 --warmup 20 --reps 100" 12 0 10 2080ti

echo "=== Running strided_copy_8 with multiple sizes ==="
scripts/run_trials.sh strided_copy_8 "--N 262144 --block 256 --warmup 20 --reps 100" 8 0 10 2080ti
scripts/run_trials.sh strided_copy_8 "--N 1048576 --block 256 --warmup 20 --reps 100" 8 0 10 2080ti
scripts/run_trials.sh strided_copy_8 "--N 4194304 --block 256 --warmup 20 --reps 100" 8 0 10 2080ti
scripts/run_trials.sh strided_copy_8 "--N 16777216 --block 256 --warmup 20 --reps 100" 8 0 10 2080ti

echo "=== Running naive_transpose with multiple sizes ==="
scripts/run_trials.sh naive_transpose "--rows 512 --cols 512 --warmup 20 --reps 100" 8 0 10 2080ti
scripts/run_trials.sh naive_transpose "--rows 1024 --cols 1024 --warmup 20 --reps 100" 8 0 10 2080ti
scripts/run_trials.sh naive_transpose "--rows 2048 --cols 2048 --warmup 20 --reps 100" 8 0 10 2080ti
scripts/run_trials.sh naive_transpose "--rows 4096 --cols 4096 --warmup 20 --reps 100" 8 0 10 2080ti

echo "=== Running shared_transpose with multiple sizes ==="
scripts/run_trials.sh shared_transpose "--rows 512 --cols 512 --warmup 20 --reps 100" 10 4224 10 2080ti
scripts/run_trials.sh shared_transpose "--rows 1024 --cols 1024 --warmup 20 --reps 100" 10 4224 10 2080ti
scripts/run_trials.sh shared_transpose "--rows 2048 --cols 2048 --warmup 20 --reps 100" 10 4224 10 2080ti
scripts/run_trials.sh shared_transpose "--rows 4096 --cols 4096 --warmup 20 --reps 100" 10 4224 10 2080ti

echo "=== Running matmul_naive with multiple sizes ==="
scripts/run_trials.sh matmul_naive "--matN 256 --warmup 10 --reps 50" 40 0 10 2080ti
scripts/run_trials.sh matmul_naive "--matN 512 --warmup 10 --reps 50" 40 0 10 2080ti
scripts/run_trials.sh matmul_naive "--matN 1024 --warmup 10 --reps 50" 40 0 10 2080ti
scripts/run_trials.sh matmul_naive "--matN 2048 --warmup 10 --reps 50" 40 0 10 2080ti

echo "=== Running matmul_tiled with multiple sizes ==="
scripts/run_trials.sh matmul_tiled "--matN 256 --warmup 10 --reps 50" 37 8192 10 2080ti
scripts/run_trials.sh matmul_tiled "--matN 512 --warmup 10 --reps 50" 37 8192 10 2080ti
scripts/run_trials.sh matmul_tiled "--matN 1024 --warmup 10 --reps 50" 37 8192 10 2080ti
scripts/run_trials.sh matmul_tiled "--matN 2048 --warmup 10 --reps 50" 37 8192 10 2080ti

echo "=== Running reduce_sum with multiple sizes ==="
scripts/run_trials.sh reduce_sum "--N 262144 --block 256 --warmup 20 --reps 100" 10 1024 10 2080ti
scripts/run_trials.sh reduce_sum "--N 1048576 --block 256 --warmup 20 --reps 100" 10 1024 10 2080ti
scripts/run_trials.sh reduce_sum "--N 4194304 --block 256 --warmup 20 --reps 100" 10 1024 10 2080ti
scripts/run_trials.sh reduce_sum "--N 16777216 --block 256 --warmup 20 --reps 100" 10 1024 10 2080ti

echo "=== Running dot_product with multiple sizes ==="
scripts/run_trials.sh dot_product "--N 262144 --block 256 --warmup 20 --reps 100" 15 1024 10 2080ti
scripts/run_trials.sh dot_product "--N 1048576 --block 256 --warmup 20 --reps 100" 15 1024 10 2080ti
scripts/run_trials.sh dot_product "--N 4194304 --block 256 --warmup 20 --reps 100" 15 1024 10 2080ti
scripts/run_trials.sh dot_product "--N 16777216 --block 256 --warmup 20 --reps 100" 15 1024 10 2080ti

echo "=== Running histogram with multiple sizes ==="
scripts/run_trials.sh histogram "--N 262144 --block 256 --warmup 10 --reps 50" 10 1024 10 2080ti
scripts/run_trials.sh histogram "--N 1048576 --block 256 --warmup 10 --reps 50" 10 1024 10 2080ti
scripts/run_trials.sh histogram "--N 4194304 --block 256 --warmup 10 --reps 50" 10 1024 10 2080ti
scripts/run_trials.sh histogram "--N 16777216 --block 256 --warmup 10 --reps 50" 10 1024 10 2080ti

echo "=== Running conv2d_3x3 with multiple sizes ==="
scripts/run_trials.sh conv2d_3x3 "--H 512 --W 512 --warmup 10 --reps 50" 30 0 10 2080ti
scripts/run_trials.sh conv2d_3x3 "--H 1024 --W 1024 --warmup 10 --reps 50" 30 0 10 2080ti
scripts/run_trials.sh conv2d_3x3 "--H 2048 --W 2048 --warmup 10 --reps 50" 30 0 10 2080ti
scripts/run_trials.sh conv2d_3x3 "--H 4096 --W 4096 --warmup 10 --reps 50" 30 0 10 2080ti

echo "=== Running conv2d_7x7 with multiple sizes ==="
scripts/run_trials.sh conv2d_7x7 "--H 512 --W 512 --warmup 10 --reps 50" 40 0 10 2080ti
scripts/run_trials.sh conv2d_7x7 "--H 1024 --W 1024 --warmup 10 --reps 50" 40 0 10 2080ti
scripts/run_trials.sh conv2d_7x7 "--H 2048 --W 2048 --warmup 10 --reps 50" 40 0 10 2080ti
scripts/run_trials.sh conv2d_7x7 "--H 4096 --W 4096 --warmup 10 --reps 50" 40 0 10 2080ti

echo "=== Running random_access with multiple sizes ==="
scripts/run_trials.sh random_access "--N 262144 --block 256 --warmup 20 --reps 100" 10 0 10 2080ti
scripts/run_trials.sh random_access "--N 1048576 --block 256 --warmup 20 --reps 100" 10 0 10 2080ti
scripts/run_trials.sh random_access "--N 4194304 --block 256 --warmup 20 --reps 100" 10 0 10 2080ti
scripts/run_trials.sh random_access "--N 16777216 --block 256 --warmup 20 --reps 100" 10 0 10 2080ti

echo "=== Running vector_add_divergent with multiple sizes ==="
scripts/run_trials.sh vector_add_divergent "--N 262144 --block 256 --warmup 20 --reps 100" 15 0 10 2080ti
scripts/run_trials.sh vector_add_divergent "--N 1048576 --block 256 --warmup 20 --reps 100" 15 0 10 2080ti
scripts/run_trials.sh vector_add_divergent "--N 4194304 --block 256 --warmup 20 --reps 100" 15 0 10 2080ti
scripts/run_trials.sh vector_add_divergent "--N 16777216 --block 256 --warmup 20 --reps 100" 15 0 10 2080ti

echo "=== Running shared_bank_conflict (fixed size) ==="
scripts/run_trials.sh shared_bank_conflict "--warmup 20 --reps 100" 206 4096 10 2080ti

echo "=== Running atomic_hotspot with multiple sizes and iterations ==="
scripts/run_trials.sh atomic_hotspot "--N 262144 --block 256 --iters 50 --warmup 10 --reps 50" 7 0 10 2080ti
scripts/run_trials.sh atomic_hotspot "--N 262144 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 2080ti
scripts/run_trials.sh atomic_hotspot "--N 1048576 --block 256 --iters 50 --warmup 10 --reps 50" 7 0 10 2080ti
scripts/run_trials.sh atomic_hotspot "--N 1048576 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 2080ti
scripts/run_trials.sh atomic_hotspot "--N 4194304 --block 256 --iters 50 --warmup 10 --reps 50" 7 0 10 2080ti
scripts/run_trials.sh atomic_hotspot "--N 4194304 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 2080ti

echo "=== done generating trials ==="
echo "Total trial files generated:"
ls -1 data/trials_*__2080ti.csv | wc -l
ls -1 data/trials_*__2080ti.csv


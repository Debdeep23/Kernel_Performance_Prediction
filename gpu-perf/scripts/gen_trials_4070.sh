#!/usr/bin/env bash
set -euo pipefail

echo "=== wipe old 4070 trials ==="
rm -f data/trials_*__4070.csv

echo "=== build runner for sm_89 (Ada Lovelace / RTX 4070) ==="
mkdir -p bin data
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_89 -DTILE=32 \
  -o bin/runner runner/main.cu 2> data/ptxas_4070.log
test -x bin/runner

# Helper the suite relies on
chmod +x scripts/run_trials.sh

# Run kernels with MULTIPLE SIZES to generate more data
# Format: kernel_name "args" regs shmem trials device
# We'll test 4 different sizes: Small (256K), Medium (1M), Large (4M), XLarge (8M)
# Note: Reduced XLarge from 16M to 8M to avoid OOM on RTX 4070 (12GB VRAM)

echo "=== Running vector_add with multiple sizes ==="
scripts/run_trials.sh vector_add "--N 262144 --block 256 --warmup 20 --reps 100" 12 0 10 4070
scripts/run_trials.sh vector_add "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10 4070
scripts/run_trials.sh vector_add "--N 4194304 --block 256 --warmup 20 --reps 100" 12 0 10 4070
scripts/run_trials.sh vector_add "--N 8388608 --block 256 --warmup 20 --reps 100" 12 0 10 4070

echo "=== Running saxpy with multiple sizes ==="
scripts/run_trials.sh saxpy "--N 262144 --block 256 --warmup 20 --reps 100" 12 0 10 4070
scripts/run_trials.sh saxpy "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10 4070
scripts/run_trials.sh saxpy "--N 4194304 --block 256 --warmup 20 --reps 100" 12 0 10 4070
scripts/run_trials.sh saxpy "--N 8388608 --block 256 --warmup 20 --reps 100" 12 0 10 4070

echo "=== Running strided_copy_8 with multiple sizes ==="
scripts/run_trials.sh strided_copy_8 "--N 262144 --block 256 --warmup 20 --reps 100" 8 0 10 4070
scripts/run_trials.sh strided_copy_8 "--N 1048576 --block 256 --warmup 20 --reps 100" 8 0 10 4070
scripts/run_trials.sh strided_copy_8 "--N 4194304 --block 256 --warmup 20 --reps 100" 8 0 10 4070
scripts/run_trials.sh strided_copy_8 "--N 8388608 --block 256 --warmup 20 --reps 100" 8 0 10 4070

echo "=== Running naive_transpose with multiple sizes ==="
scripts/run_trials.sh naive_transpose "--rows 512 --cols 512 --warmup 20 --reps 100" 8 0 10 4070
scripts/run_trials.sh naive_transpose "--rows 1024 --cols 1024 --warmup 20 --reps 100" 8 0 10 4070
scripts/run_trials.sh naive_transpose "--rows 2048 --cols 2048 --warmup 20 --reps 100" 8 0 10 4070
scripts/run_trials.sh naive_transpose "--rows 3072 --cols 3072 --warmup 20 --reps 100" 8 0 10 4070

echo "=== Running shared_transpose with multiple sizes ==="
scripts/run_trials.sh shared_transpose "--rows 512 --cols 512 --warmup 20 --reps 100" 10 4224 10 4070
scripts/run_trials.sh shared_transpose "--rows 1024 --cols 1024 --warmup 20 --reps 100" 10 4224 10 4070
scripts/run_trials.sh shared_transpose "--rows 2048 --cols 2048 --warmup 20 --reps 100" 10 4224 10 4070
scripts/run_trials.sh shared_transpose "--rows 3072 --cols 3072 --warmup 20 --reps 100" 10 4224 10 4070

echo "=== Running matmul_naive with multiple sizes ==="
scripts/run_trials.sh matmul_naive "--matN 256 --warmup 10 --reps 50" 40 0 10 4070
scripts/run_trials.sh matmul_naive "--matN 512 --warmup 10 --reps 50" 40 0 10 4070
scripts/run_trials.sh matmul_naive "--matN 1024 --warmup 10 --reps 50" 40 0 10 4070
scripts/run_trials.sh matmul_naive "--matN 2048 --warmup 10 --reps 50" 40 0 10 4070

echo "=== Running matmul_tiled with multiple sizes ==="
scripts/run_trials.sh matmul_tiled "--matN 256 --warmup 10 --reps 50" 37 8192 10 4070
scripts/run_trials.sh matmul_tiled "--matN 512 --warmup 10 --reps 50" 37 8192 10 4070
scripts/run_trials.sh matmul_tiled "--matN 1024 --warmup 10 --reps 50" 37 8192 10 4070
scripts/run_trials.sh matmul_tiled "--matN 2048 --warmup 10 --reps 50" 37 8192 10 4070

echo "=== Running reduce_sum with multiple sizes ==="
scripts/run_trials.sh reduce_sum "--N 262144 --block 256 --warmup 20 --reps 100" 10 1024 10 4070
scripts/run_trials.sh reduce_sum "--N 1048576 --block 256 --warmup 20 --reps 100" 10 1024 10 4070
scripts/run_trials.sh reduce_sum "--N 4194304 --block 256 --warmup 20 --reps 100" 10 1024 10 4070
scripts/run_trials.sh reduce_sum "--N 8388608 --block 256 --warmup 20 --reps 100" 10 1024 10 4070

echo "=== Running dot_product with multiple sizes ==="
scripts/run_trials.sh dot_product "--N 262144 --block 256 --warmup 20 --reps 100" 15 1024 10 4070
scripts/run_trials.sh dot_product "--N 1048576 --block 256 --warmup 20 --reps 100" 15 1024 10 4070
scripts/run_trials.sh dot_product "--N 4194304 --block 256 --warmup 20 --reps 100" 15 1024 10 4070
scripts/run_trials.sh dot_product "--N 8388608 --block 256 --warmup 20 --reps 100" 15 1024 10 4070

echo "=== Running histogram with multiple sizes ==="
scripts/run_trials.sh histogram "--N 262144 --block 256 --warmup 10 --reps 50" 10 1024 10 4070
scripts/run_trials.sh histogram "--N 1048576 --block 256 --warmup 10 --reps 50" 10 1024 10 4070
scripts/run_trials.sh histogram "--N 4194304 --block 256 --warmup 10 --reps 50" 10 1024 10 4070
scripts/run_trials.sh histogram "--N 8388608 --block 256 --warmup 10 --reps 50" 10 1024 10 4070

echo "=== Running conv2d_3x3 with multiple sizes ==="
scripts/run_trials.sh conv2d_3x3 "--H 512 --W 512 --warmup 10 --reps 50" 30 0 10 4070
scripts/run_trials.sh conv2d_3x3 "--H 1024 --W 1024 --warmup 10 --reps 50" 30 0 10 4070
scripts/run_trials.sh conv2d_3x3 "--H 2048 --W 2048 --warmup 10 --reps 50" 30 0 10 4070
scripts/run_trials.sh conv2d_3x3 "--H 3072 --W 3072 --warmup 10 --reps 50" 30 0 10 4070

echo "=== Running conv2d_7x7 with multiple sizes ==="
scripts/run_trials.sh conv2d_7x7 "--H 512 --W 512 --warmup 10 --reps 50" 40 0 10 4070
scripts/run_trials.sh conv2d_7x7 "--H 1024 --W 1024 --warmup 10 --reps 50" 40 0 10 4070
scripts/run_trials.sh conv2d_7x7 "--H 2048 --W 2048 --warmup 10 --reps 50" 40 0 10 4070
scripts/run_trials.sh conv2d_7x7 "--H 3072 --W 3072 --warmup 10 --reps 50" 40 0 10 4070

echo "=== Running random_access with multiple sizes ==="
scripts/run_trials.sh random_access "--N 262144 --block 256 --warmup 20 --reps 100" 10 0 10 4070
scripts/run_trials.sh random_access "--N 1048576 --block 256 --warmup 20 --reps 100" 10 0 10 4070
scripts/run_trials.sh random_access "--N 4194304 --block 256 --warmup 20 --reps 100" 10 0 10 4070
scripts/run_trials.sh random_access "--N 8388608 --block 256 --warmup 20 --reps 100" 10 0 10 4070

echo "=== Running vector_add_divergent with multiple sizes ==="
scripts/run_trials.sh vector_add_divergent "--N 262144 --block 256 --warmup 20 --reps 100" 15 0 10 4070
scripts/run_trials.sh vector_add_divergent "--N 1048576 --block 256 --warmup 20 --reps 100" 15 0 10 4070
scripts/run_trials.sh vector_add_divergent "--N 4194304 --block 256 --warmup 20 --reps 100" 15 0 10 4070
scripts/run_trials.sh vector_add_divergent "--N 8388608 --block 256 --warmup 20 --reps 100" 15 0 10 4070

echo "=== Running shared_bank_conflict (fixed size) ==="
scripts/run_trials.sh shared_bank_conflict "--warmup 20 --reps 100" 206 4096 10 4070

echo "=== Running atomic_hotspot with multiple sizes and iterations ==="
scripts/run_trials.sh atomic_hotspot "--N 262144 --block 256 --iters 50 --warmup 10 --reps 50" 7 0 10 4070
scripts/run_trials.sh atomic_hotspot "--N 262144 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 4070
scripts/run_trials.sh atomic_hotspot "--N 1048576 --block 256 --iters 50 --warmup 10 --reps 50" 7 0 10 4070
scripts/run_trials.sh atomic_hotspot "--N 1048576 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 4070
scripts/run_trials.sh atomic_hotspot "--N 4194304 --block 256 --iters 50 --warmup 10 --reps 50" 7 0 10 4070
scripts/run_trials.sh atomic_hotspot "--N 4194304 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 4070

echo "=== done generating trials ==="
echo "Total trial files generated:"
ls -1 data/trials_*__4070.csv | wc -l
ls -1 data/trials_*__4070.csv

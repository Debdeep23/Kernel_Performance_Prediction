#!/usr/bin/env bash
set -euo pipefail

echo "=== wipe old titanx trials ==="
rm -f data/trials_*__titanx.csv

echo "=== build runner for sm_52 (Maxwell / Titan X) ==="
mkdir -p bin data
nvcc -std=c++14 -O3 --ptxas-options=-v -lineinfo -arch=sm_52 -DTILE=32 \
  -o bin/runner runner/main.cu 2> data/ptxas_titanx.log
test -x bin/runner

chmod +x scripts/run_trials.sh

scripts/run_trials.sh vector_add            "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10  titanx
scripts/run_trials.sh saxpy                 "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10  titanx
scripts/run_trials.sh strided_copy_8        "--N 1048576 --block 256 --warmup 20 --reps 100"  8 0 10  titanx
scripts/run_trials.sh naive_transpose       "--rows 2048 --cols 2048 --warmup 20 --reps 100"  8 0 10  titanx
scripts/run_trials.sh shared_transpose      "--rows 2048 --cols 2048 --warmup 20 --reps 100" 10 4224 10 titanx
scripts/run_trials.sh matmul_tiled          "--matN 512 --warmup 10 --reps 50"               37 8192 10 titanx
scripts/run_trials.sh matmul_naive          "--matN 512 --warmup 10 --reps 50"               40 0    10 titanx
scripts/run_trials.sh reduce_sum            "--N 1048576 --block 256 --warmup 20 --reps 100" 10 1024 10 titanx
scripts/run_trials.sh dot_product           "--N 1048576 --block 256 --warmup 20 --reps 100" 15 1024 10 titanx
scripts/run_trials.sh histogram             "--N 1048576 --block 256 --warmup 10  --reps 50" 10 1024 10 titanx
scripts/run_trials.sh conv2d_3x3            "--H 1024 --W 1024 --warmup 10 --reps 50"        30 0    10 titanx
scripts/run_trials.sh conv2d_7x7            "--H 1024 --W 1024 --warmup 10 --reps 50"        40 0    10 titanx
scripts/run_trials.sh random_access         "--N 1048576 --block 256 --warmup 20 --reps 100" 10 0    10 titanx
scripts/run_trials.sh vector_add_divergent  "--N 1048576 --block 256 --warmup 20 --reps 100" 15 0    10 titanx
scripts/run_trials.sh shared_bank_conflict  "--warmup 20 --reps 100"                        206 4096 10 titanx
scripts/run_trials.sh atomic_hotspot        "--N 1048576 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 titanx

echo "=== done generating trials ==="
ls -1 data/trials_*__titanx.csv


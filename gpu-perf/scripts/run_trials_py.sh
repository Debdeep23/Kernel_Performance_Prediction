#!/usr/bin/env bash
# scripts/run_trials_py.sh
set -euo pipefail
TAG="2080ti"

python3 scripts/call_runner_and_log.py vector_add           "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10 $TAG
python3 scripts/call_runner_and_log.py saxpy                "--N 1048576 --block 256 --warmup 20 --reps 100" 12 0 10 $TAG
python3 scripts/call_runner_and_log.py strided_copy_8       "--N 1048576 --block 256 --warmup 20 --reps 100"  8 0 10 $TAG
python3 scripts/call_runner_and_log.py naive_transpose      "--rows 2048 --cols 2048 --warmup 20 --reps 100"  8 0 10 $TAG
python3 scripts/call_runner_and_log.py shared_transpose     "--rows 2048 --cols 2048 --warmup 20 --reps 100" 10 4224 10 $TAG
python3 scripts/call_runner_and_log.py matmul_tiled         "--matN 512 --warmup 10 --reps 50"               37 8192 10 $TAG
python3 scripts/call_runner_and_log.py matmul_naive         "--matN 512 --warmup 10 --reps 50"               40 0    10 $TAG
python3 scripts/call_runner_and_log.py reduce_sum           "--N 1048576 --block 256 --warmup 20 --reps 100" 10 1024 10 $TAG
python3 scripts/call_runner_and_log.py dot_product          "--N 1048576 --block 256 --warmup 20 --reps 100" 15 1024 10 $TAG
python3 scripts/call_runner_and_log.py histogram            "--N 1048576 --block 256 --warmup 10  --reps 50" 10 1024 10 $TAG
python3 scripts/call_runner_and_log.py conv2d_3x3           "--H 1024 --W 1024 --warmup 10 --reps 50"        30 0    10 $TAG
python3 scripts/call_runner_and_log.py conv2d_7x7           "--H 1024 --W 1024 --warmup 10 --reps 50"        40 0    10 $TAG
python3 scripts/call_runner_and_log.py random_access        "--N 1048576 --block 256 --warmup 20 --reps 100" 10 0    10 $TAG
python3 scripts/call_runner_and_log.py vector_add_divergent "--N 1048576 --block 256 --warmup 20 --reps 100" 15 0    10 $TAG
python3 scripts/call_runner_and_log.py shared_bank_conflict "--warmup 20 --reps 100"                        206 4096 10 $TAG
python3 scripts/call_runner_and_log.py atomic_hotspot       "--N 1048576 --block 256 --iters 100 --warmup 10 --reps 50" 7 0 10 $TAG


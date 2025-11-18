#!/usr/bin/env python3
"""Add iters column to final datasets and fix BYTES for atomic_hotspot"""
import csv
import re
import sys

def extract_iters_from_args(args):
    """Extract --iters N from args string"""
    m = re.search(r'--iters\s+(\d+)', args)
    return int(m.group(1)) if m else 0

def I(x):
    try:
        return int(x) if x and x != '' else 0
    except:
        return 0

def fix_4070_dataset():
    """Add iters column to 4070 dataset and fix BYTES"""

    # Step 1: Read raw data to get iters mapping
    # Use a more flexible key: kernel + problem_size (accounting for normalization)
    iters_map = {}  # (kernel, problem_size) -> iters
    with open('/home/user/test1/gpu-perf/data/runs_4070.csv') as f:
        reader = csv.DictReader(f)
        for r in reader:
            kernel = r['kernel']
            # Compute problem size from raw data (which uses rows/cols even for 1D)
            N_raw = I(r.get('N', ''))
            rows_raw = I(r.get('rows', ''))
            cols_raw = I(r.get('cols', ''))

            # Calculate effective problem size
            if N_raw > 0:
                problem_size = N_raw
            elif rows_raw > 0 and cols_raw > 1:
                problem_size = rows_raw * cols_raw  # 2D
            elif rows_raw > 0:
                problem_size = rows_raw  # 1D stored as rows
            else:
                problem_size = 0

            key = (kernel, problem_size)
            iters = extract_iters_from_args(r.get('args', ''))
            iters_map[key] = iters

    # Step 2: Read final dataset and add iters
    with open('/home/user/test1/gpu-perf/data/runs_4070_final.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Update each row
        for r in rows:
            kernel = r['kernel']

            # Calculate problem size from final dataset
            N_final = I(r.get('N', ''))
            rows_final = I(r.get('rows', ''))
            cols_final = I(r.get('cols', ''))

            if N_final > 0:
                problem_size = N_final
            elif rows_final > 0 and cols_final > 1:
                problem_size = rows_final * cols_final
            elif rows_final > 0:
                problem_size = rows_final
            else:
                problem_size = 0

            # Look up iters
            key = (kernel, problem_size)
            iters = iters_map.get(key, 0)
            r['iters'] = str(iters)

            # Fix BYTES for atomic_hotspot
            if kernel == 'atomic_hotspot' and iters > 0:
                if problem_size > 0:
                    # BYTES = N * iters * 8 (8 bytes per atomic RMW)
                    correct_bytes = problem_size * iters * 8
                    old_bytes = r['BYTES']
                    r['BYTES'] = str(correct_bytes)
                    print(f"Fixed atomic_hotspot N={problem_size} iters={iters}: BYTES {old_bytes} -> {correct_bytes}")

    # Step 3: Write updated dataset with iters column in correct position
    # Insert iters after cols, before size_kind (to match 2080ti schema)
    old_fields = ['kernel', 'regs', 'shmem', 'mean_ms', 'std_ms', 'N', 'rows', 'cols',
                  'block', 'grid_blocks', 'size_kind', 'FLOPs', 'BYTES', 'arithmetic_intensity',
                  'working_set_bytes', 'shared_bytes', 'mem_pattern', 'gpu_device_name',
                  'gpu_cc_major', 'gpu_sms', 'gpu_max_threads_per_sm', 'gpu_max_blocks_per_sm',
                  'gpu_regs_per_sm', 'gpu_shared_mem_per_sm', 'gpu_l2_bytes', 'gpu_warp_size',
                  'calibrated_mem_bandwidth_gbps', 'calibrated_compute_gflops',
                  'achieved_bandwidth_gbps', 'achieved_compute_gflops', 'T1_model_ms', 'speedup_model']

    new_fields = ['kernel', 'regs', 'shmem', 'mean_ms', 'std_ms', 'N', 'rows', 'cols', 'iters',
                  'block', 'grid_blocks', 'size_kind', 'FLOPs', 'BYTES', 'arithmetic_intensity',
                  'working_set_bytes', 'shared_bytes', 'mem_pattern', 'gpu_device_name',
                  'gpu_cc_major', 'gpu_sms', 'gpu_max_threads_per_sm', 'gpu_max_blocks_per_sm',
                  'gpu_regs_per_sm', 'gpu_shared_mem_per_sm', 'gpu_l2_bytes', 'gpu_warp_size',
                  'calibrated_mem_bandwidth_gbps', 'calibrated_compute_gflops',
                  'achieved_bandwidth_gbps', 'achieved_compute_gflops', 'T1_model_ms', 'speedup_model']

    with open('/home/user/test1/gpu-perf/data/runs_4070_final.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in new_fields})

    print(f"[OK] Updated runs_4070_final.csv with iters column")

if __name__ == '__main__':
    fix_4070_dataset()

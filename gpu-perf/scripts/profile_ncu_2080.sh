#!/usr/bin/env bash
# scripts/profile_ncu_2080.sh
# Profile kernels on 2080 using NVIDIA Nsight Compute (ncu)
# Usage: scripts/profile_ncu_2080.sh <kernel> "<argstr>" [output_dir]
set -euo pipefail

KERNEL="${1:-vector_add}"
ARGSTR="${2:---rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10}"
OUTPUT_DIR="${3:-data/profiling}"

mkdir -p "$OUTPUT_DIR"

echo "=== Profiling $KERNEL on 2080 with ncu (Nsight Compute) ==="

# Output files
NCU_REPORT="${OUTPUT_DIR}/ncu_${KERNEL}_report.ncu-rep"
NCU_TEXT="${OUTPUT_DIR}/ncu_${KERNEL}_details.txt"
NCU_CSV="${OUTPUT_DIR}/ncu_${KERNEL}_metrics.csv"

# Reduce warmup/reps for profiling (ncu runs kernels multiple times internally)
PROFILE_ARGS=$(echo "$ARGSTR" | sed 's/--warmup [0-9]*/--warmup 1/; s/--reps [0-9]*/--reps 1/')

# Full detailed profiling with ncu-rep file for GUI analysis
echo "Running comprehensive profiling (this may take a while)..."
ncu --set full \
    --target-processes all \
    --launch-count 1 \
    --export "$NCU_REPORT" \
    bin/runner --kernel "$KERNEL" $PROFILE_ARGS

echo "Full report saved to: $NCU_REPORT"
echo "(Open with 'ncu-ui $NCU_REPORT' for GUI analysis)"

# Generate detailed text output
echo "Generating detailed text report..."
ncu --set detailed \
    --target-processes all \
    --launch-count 1 \
    --page details \
    bin/runner --kernel "$KERNEL" $PROFILE_ARGS \
    > "$NCU_TEXT" 2>&1

echo "Detailed text saved to: $NCU_TEXT"

# Extract specific metrics to CSV for easy analysis
echo "Collecting specific metrics to CSV..."
ncu --csv \
    --target-processes all \
    --launch-count 1 \
    --metrics \
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.pct,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct,\
smsp__average_warps_issue_stalled_drain_per_issue_active.pct,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,\
smsp__inst_executed.avg.per_cycle_active \
    bin/runner --kernel "$KERNEL" $PROFILE_ARGS \
    > "$NCU_CSV" 2>&1

echo "CSV metrics saved to: $NCU_CSV"

echo ""
echo "=== Profiling Complete ==="
echo "Files generated:"
echo "  - $NCU_REPORT (open with ncu-ui for GUI)"
echo "  - $NCU_TEXT"
echo "  - $NCU_CSV"
echo ""
echo "Quick summary:"
ncu --print-summary base \
    bin/runner --kernel "$KERNEL" $ARGSTR 2>&1 | tail -n 20

#!/usr/bin/env bash
# scripts/profile_kernel_2080.sh
# Smart profiler that auto-detects available tools (nvprof/ncu/nsys)
# Usage: scripts/profile_kernel_2080.sh <kernel> "<argstr>" [output_dir]
set -euo pipefail

KERNEL="${1:-vector_add}"
ARGSTR="${2:---rows 1048576 --cols 1 --block 256 --warmup 5 --reps 10}"
OUTPUT_DIR="${3:-data/profiling}"

mkdir -p "$OUTPUT_DIR"

# Detect available profiling tools
HAS_NVPROF=false
HAS_NCU=false
HAS_NSYS=false

if command -v nvprof &> /dev/null; then
    HAS_NVPROF=true
fi

if command -v ncu &> /dev/null; then
    HAS_NCU=true
fi

if command -v nsys &> /dev/null; then
    HAS_NSYS=true
fi

echo "=== Profiling Tool Detection ==="
echo "nvprof: $HAS_NVPROF"
echo "ncu:    $HAS_NCU"
echo "nsys:   $HAS_NSYS"
echo ""

# Check if we have any profiler
if [[ "$HAS_NVPROF" == "false" && "$HAS_NCU" == "false" && "$HAS_NSYS" == "false" ]]; then
    echo "ERROR: No NVIDIA profiling tools found!"
    echo ""
    echo "Please install one of the following:"
    echo ""
    echo "1. NVIDIA Nsight Compute (ncu) - RECOMMENDED for modern GPUs"
    echo "   Download from: https://developer.nvidia.com/nsight-compute"
    echo "   Or install via: sudo apt install nvidia-nsight-compute (on Ubuntu)"
    echo ""
    echo "2. nvprof (deprecated but still works on older CUDA)"
    echo "   Install CUDA Toolkit 10.x or 11.x"
    echo "   Available at: https://developer.nvidia.com/cuda-toolkit-archive"
    echo ""
    echo "3. NVIDIA Nsight Systems (nsys) - For timeline profiling"
    echo "   Download from: https://developer.nvidia.com/nsight-systems"
    echo ""
    echo "After installation, ensure the tools are in your PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo ""
    echo "Alternatively, you can run basic timing without profiling:"
    echo "  bin/runner --kernel $KERNEL $ARGSTR"
    exit 1
fi

# Prefer ncu > nvprof > nsys for detailed metrics
TOOL=""
if [[ "$HAS_NCU" == "true" ]]; then
    TOOL="ncu"
    echo "Using: NVIDIA Nsight Compute (ncu)"
elif [[ "$HAS_NVPROF" == "true" ]]; then
    TOOL="nvprof"
    echo "Using: nvprof (legacy)"
elif [[ "$HAS_NSYS" == "true" ]]; then
    TOOL="nsys"
    echo "Using: NVIDIA Nsight Systems (nsys)"
fi

echo "Profiling: $KERNEL"
echo "Arguments: $ARGSTR"
echo "Output: $OUTPUT_DIR"
echo ""

# Run the appropriate profiler
if [[ "$TOOL" == "ncu" ]]; then
    # Use ncu profiling
    echo "Running comprehensive profiling with ncu..."

    NCU_REPORT="${OUTPUT_DIR}/ncu_${KERNEL}_report.ncu-rep"
    NCU_TEXT="${OUTPUT_DIR}/ncu_${KERNEL}_details.txt"
    NCU_CSV="${OUTPUT_DIR}/ncu_${KERNEL}_metrics.csv"

    # Modify args to reduce profiling overhead (ncu is slow)
    # Extract current args and reduce warmup/reps for profiling
    PROFILE_ARGS=$(echo "$ARGSTR" | sed 's/--warmup [0-9]*/--warmup 1/; s/--reps [0-9]*/--reps 1/')

    # Full detailed profiling
    # Use --target-processes all to ensure we capture the kernel
    # Use --launch-count 1 to profile just one kernel launch (faster)
    ncu --set full \
        --target-processes all \
        --launch-count 1 \
        --export "$NCU_REPORT" \
        bin/runner --kernel "$KERNEL" $PROFILE_ARGS

    echo "Full report: $NCU_REPORT"
    echo "(Open with 'ncu-ui $NCU_REPORT' for GUI)"

    # Generate text output with key metrics
    ncu --set detailed \
        --target-processes all \
        --launch-count 1 \
        --page details \
        bin/runner --kernel "$KERNEL" $PROFILE_ARGS \
        > "$NCU_TEXT" 2>&1 || true

    echo "Details: $NCU_TEXT"

    # Extract CSV metrics
    ncu --csv \
        --target-processes all \
        --launch-count 1 \
        --metrics \
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
        bin/runner --kernel "$KERNEL" $PROFILE_ARGS \
        > "$NCU_CSV" 2>&1 || true

    echo "Metrics CSV: $NCU_CSV"

elif [[ "$TOOL" == "nvprof" ]]; then
    # Use nvprof profiling
    echo "Running profiling with nvprof..."

    SUMMARY_FILE="${OUTPUT_DIR}/nvprof_${KERNEL}_summary.txt"
    METRICS_FILE="${OUTPUT_DIR}/nvprof_${KERNEL}_metrics.csv"

    # Basic profiling with summary
    nvprof --print-gpu-trace \
           --print-summary \
           bin/runner --kernel "$KERNEL" $ARGSTR \
           > "$SUMMARY_FILE" 2>&1

    echo "Summary: $SUMMARY_FILE"

    # Comprehensive metrics
    nvprof --csv \
           --metrics \
flop_count_sp,\
flop_sp_efficiency,\
dram_read_throughput,\
dram_write_throughput,\
dram_utilization,\
gld_throughput,\
gst_throughput,\
gld_efficiency,\
gst_efficiency,\
achieved_occupancy,\
sm_efficiency,\
warp_execution_efficiency,\
branch_efficiency \
           bin/runner --kernel "$KERNEL" $ARGSTR \
           > "$METRICS_FILE" 2>&1

    echo "Metrics CSV: $METRICS_FILE"

elif [[ "$TOOL" == "nsys" ]]; then
    # Use nsys profiling
    echo "Running profiling with nsys..."

    NSYS_REPORT="${OUTPUT_DIR}/nsys_${KERNEL}_report.nsys-rep"
    NSYS_STATS="${OUTPUT_DIR}/nsys_${KERNEL}_stats.txt"

    nsys profile \
         --output="$NSYS_REPORT" \
         --stats=true \
         bin/runner --kernel "$KERNEL" $ARGSTR \
         > "$NSYS_STATS" 2>&1

    echo "Report: $NSYS_REPORT"
    echo "(Open with 'nsys-ui $NSYS_REPORT' for GUI)"
    echo "Stats: $NSYS_STATS"
fi

echo ""
echo "=== Profiling Complete ==="
echo ""

# Show quick summary if possible
if [[ "$TOOL" == "nvprof" ]]; then
    echo "Quick summary:"
    tail -n 20 "${OUTPUT_DIR}/nvprof_${KERNEL}_summary.txt" || true
elif [[ "$TOOL" == "ncu" ]]; then
    echo "For detailed analysis, run:"
    echo "  ncu-ui ${OUTPUT_DIR}/ncu_${KERNEL}_report.ncu-rep"
fi

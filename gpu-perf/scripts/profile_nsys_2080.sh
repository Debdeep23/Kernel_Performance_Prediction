#!/bin/bash
# Profile a single kernel using nsys (NVIDIA Nsight Systems)
# This is an alternative to ncu when ncu has compatibility issues
# Usage: ./profile_nsys_2080.sh <kernel_name> [args...]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <kernel_name> [runner_args...]"
    echo "Example: $0 vector_add"
    echo "Example: $0 matmul_tiled --N 1024 --warmup 5 --reps 10"
    exit 1
fi

KERNEL=$1
shift  # Remove kernel name from args

# Default args if none provided
if [ $# -eq 0 ]; then
    ARGSTR="--warmup 5 --reps 10"
else
    ARGSTR="$@"
fi

# Output directory
PROFILE_DIR="data/profiling_2080"
mkdir -p "$PROFILE_DIR"

# Check if runner exists
if [ ! -f bin/runner ]; then
    echo "Error: bin/runner not found. Run 'make' first."
    exit 1
fi

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "Error: nsys not found. Please install NVIDIA Nsight Systems."
    echo "Download from: https://developer.nvidia.com/nsight-systems"
    exit 1
fi

echo "============================================"
echo "Profiling kernel: $KERNEL"
echo "Using: nsys (NVIDIA Nsight Systems)"
echo "Arguments: $ARGSTR"
echo "============================================"

# Output files
NSYS_REPORT="$PROFILE_DIR/nsys_${KERNEL}_report"
STATS_FILE="$PROFILE_DIR/nsys_${KERNEL}_stats.txt"
CSV_PREFIX="$PROFILE_DIR/nsys_${KERNEL}"

# Profile with nsys
echo "Running nsys profiling..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --output="$NSYS_REPORT" \
    --force-overwrite=true \
    --stats=true \
    bin/runner --kernel "$KERNEL" $ARGSTR

# Generate stats report
echo ""
echo "Generating statistics..."
nsys stats "$NSYS_REPORT.nsys-rep" > "$STATS_FILE" 2>&1 || true

# Export kernel statistics to CSV
echo "Exporting kernel stats to CSV..."
nsys stats \
    --report cuda_gpu_kern_sum \
    "$NSYS_REPORT.nsys-rep" \
    --format csv \
    --output "$PROFILE_DIR" \
    --force-export true \
    2>/dev/null || echo "Warning: Could not export cuda_gpu_kern_sum"

nsys stats \
    --report cuda_gpu_mem_size_sum \
    "$NSYS_REPORT.nsys-rep" \
    --format csv \
    --output "$PROFILE_DIR" \
    --force-export true \
    2>/dev/null || echo "Warning: Could not export cuda_gpu_mem_size_sum"

nsys stats \
    --report cuda_gpu_mem_time_sum \
    "$NSYS_REPORT.nsys-rep" \
    --format csv \
    --output "$PROFILE_DIR" \
    --force-export true \
    2>/dev/null || echo "Warning: Could not export cuda_gpu_mem_time_sum"

echo ""
echo "============================================"
echo "Profiling complete!"
echo "============================================"
echo "Reports generated:"
echo "  Binary report:  $NSYS_REPORT.nsys-rep"
echo "  Statistics:     $STATS_FILE"
echo "  CSV exports:    $PROFILE_DIR/*.csv"
echo ""
echo "View in GUI:"
echo "  nsys-ui $NSYS_REPORT.nsys-rep"
echo ""
echo "View stats:"
echo "  cat $STATS_FILE"
echo ""
echo "Parse results:"
echo "  python3 scripts/parse_nsys_results.py $PROFILE_DIR"
echo "============================================"

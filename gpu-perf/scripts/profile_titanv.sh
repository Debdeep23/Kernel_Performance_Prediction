#!/bin/bash
# Profile a single kernel on TITAN V using ncu
# Usage: ./profile_titanv.sh <kernel_name> [args...]
# Example: ./profile_titanv.sh vector_add --N 1048576

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <kernel_name> [runner_args...]"
    echo "Example: $0 vector_add --N 1048576"
    echo "Example: $0 matmul_tiled --rows 1024 --cols 1024"
    exit 1
fi

KERNEL=$1
shift

# Default args if none provided
if [ $# -eq 0 ]; then
    ARGSTR="--warmup 1 --reps 1"
else
    ARGSTR="$@"
fi

# Output directory
PROFILE_DIR="data/profiling_titanv"
mkdir -p "$PROFILE_DIR"

# Check if runner exists
if [ ! -f bin/runner ]; then
    echo "Error: bin/runner not found. Run 'make' first."
    exit 1
fi

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "Error: ncu not found. Make sure CUDA toolkit is loaded."
    echo "Try: module load cuda"
    exit 1
fi

echo "============================================"
echo "Profiling: $KERNEL on TITAN V"
echo "Arguments: $ARGSTR"
echo "============================================"

# Modify args for profiling (reduce warmup/reps to minimize profiling time)
PROFILE_ARGS=$(echo "$ARGSTR" | sed 's/--warmup [0-9]*/--warmup 1/; s/--reps [0-9]*/--reps 1/')

# Output files
NCU_REPORT="$PROFILE_DIR/ncu_${KERNEL}"
NCU_DETAILS="$PROFILE_DIR/ncu_${KERNEL}_details.txt"

# Profile with ncu
echo "Running ncu profiling..."
ncu --set full \
    --target-processes all \
    --launch-count 1 \
    --export "$NCU_REPORT" \
    --log-file "$NCU_DETAILS" \
    bin/runner --kernel "$KERNEL" $PROFILE_ARGS

echo ""
echo "============================================"
echo "Profiling complete!"
echo "============================================"
echo "Reports:"
echo "  NCU binary: $NCU_REPORT.ncu-rep"
echo "  Details:    $NCU_DETAILS"
echo ""
echo "View in ncu-ui:"
echo "  ncu-ui $NCU_REPORT.ncu-rep"
echo "============================================"

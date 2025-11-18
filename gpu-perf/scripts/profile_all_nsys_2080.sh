#!/bin/bash
# Profile all kernels using nsys
# Alternative to ncu when ncu has compatibility issues

set -e

SCRIPT_DIR=$(dirname "$0")

# List of all kernels
KERNELS=(
    "vector_add"
    "saxpy"
    "strided_copy_8"
    "reduce_sum"
    "dot_product"
    "histogram"
    "random_access"
    "naive_transpose"
    "shared_transpose"
    "matmul_naive"
    "matmul_tiled"
    "matmul_tiled_coarse"
    "shared_bank_conflict"
    "atomic_hotspot"
    "vector_add_divergent"
    "conv2d_3x3"
    "conv2d_7x7"
)

echo "============================================"
echo "Profiling all kernels with nsys"
echo "Total kernels: ${#KERNELS[@]}"
echo "============================================"
echo ""

# Profile each kernel
for kernel in "${KERNELS[@]}"; do
    echo ""
    echo ">>> Profiling: $kernel"
    echo ""

    # Use appropriate args for each kernel type
    case $kernel in
        atomic_hotspot)
            # atomic_hotspot needs --iters parameter
            "$SCRIPT_DIR/profile_nsys_2080.sh" "$kernel" --N 1048576 --iters 100 --warmup 5 --reps 10
            ;;
        shared_bank_conflict)
            # shared_bank_conflict has fixed size
            "$SCRIPT_DIR/profile_nsys_2080.sh" "$kernel" --warmup 5 --reps 10
            ;;
        matmul_* | *transpose | conv2d_*)
            # Matrix/image kernels use rows/cols
            "$SCRIPT_DIR/profile_nsys_2080.sh" "$kernel" --rows 1024 --cols 1024 --warmup 5 --reps 10
            ;;
        *)
            # Vector kernels use N
            "$SCRIPT_DIR/profile_nsys_2080.sh" "$kernel" --N 1048576 --warmup 5 --reps 10
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo ">>> SUCCESS: $kernel"
    else
        echo ">>> FAILED: $kernel" >&2
    fi

    echo ""
    echo "---"
done

echo ""
echo "============================================"
echo "All kernels profiled!"
echo "============================================"
echo ""
echo "Parse all results:"
echo "  python3 scripts/parse_nsys_results.py data/profiling_2080"
echo ""
echo "View individual reports in GUI:"
echo "  nsys-ui data/profiling_2080/nsys_<kernel>_report.nsys-rep"
echo "============================================"

#!/bin/bash
# Extract metrics from ncu binary reports to CSV format
# Usage: ./export_ncu_metrics.sh

# Save the repo root directory
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

PROFILE_DIR="data/profiling_4070"
OUTPUT_DIR="$PROFILE_DIR/csv_exports"

mkdir -p "$OUTPUT_DIR"

echo "Exporting ncu metrics to CSV..."
echo ""

cd "$PROFILE_DIR"

for ncu_file in ncu_*.ncu-rep; do
    if [ ! -f "$ncu_file" ]; then
        continue
    fi

    kernel=$(basename "$ncu_file" .ncu-rep | sed 's/ncu_//')

    echo "Exporting $kernel..."

    # Export basic metrics
    ncu --import "$ncu_file" --csv > "csv_exports/${kernel}_summary.csv" 2>/dev/null

    # Export detailed metrics
    ncu --import "$ncu_file" --page details --csv > "csv_exports/${kernel}_details.csv" 2>/dev/null

    echo "  -> csv_exports/${kernel}_summary.csv"
    echo "  -> csv_exports/${kernel}_details.csv"
done

echo ""
echo "============================================"
echo "Export complete!"
echo "============================================"
echo "CSV files saved to: $OUTPUT_DIR"
echo ""
echo "Combining all CSVs into one file..."
cd "$REPO_ROOT"
python3 scripts/combine_ncu_results.py

echo ""
echo "============================================"
echo "All done!"
echo "============================================"
echo "Individual CSVs: $OUTPUT_DIR/"
echo "Combined summary: data/profiling_4070/all_kernels_summary.csv"
echo "Combined details: data/profiling_4070/all_kernels_details.csv"
echo "============================================"

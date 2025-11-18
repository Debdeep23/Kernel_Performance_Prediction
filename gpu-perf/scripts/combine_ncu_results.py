#!/usr/bin/env python3
"""
Combine all ncu profiling CSV exports into one consolidated summary CSV.
Usage: python3 scripts/combine_ncu_results.py [input_dir] [output_file]
"""

import sys
import os
import csv
import glob
from collections import defaultdict

def parse_ncu_csv(csv_file):
    """Parse an ncu CSV export and extract key metrics."""
    metrics = {}

    if not os.path.exists(csv_file):
        return metrics

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Each row in ncu CSV has metric name and value
                for key, value in row.items():
                    if key and value:
                        clean_key = key.strip()
                        clean_value = value.strip()

                        # Skip empty values
                        if not clean_value or clean_value == '':
                            continue

                        # Try to convert to float
                        try:
                            metrics[clean_key] = float(clean_value.replace(',', ''))
                        except (ValueError, AttributeError):
                            metrics[clean_key] = clean_value

    except Exception as e:
        print(f"Warning: Could not parse {csv_file}: {e}", file=sys.stderr)

    return metrics

def extract_kernel_name(filename):
    """Extract kernel name from filename like 'vector_add_summary.csv'."""
    basename = os.path.basename(filename)
    # Remove _summary.csv or _details.csv
    kernel = basename.replace('_summary.csv', '').replace('_details.csv', '')
    return kernel

def combine_all_csvs(input_dir, output_file, csv_type='summary'):
    """Combine all ncu CSV files into one summary."""

    # Find all CSV files
    pattern = os.path.join(input_dir, f'*_{csv_type}.csv')
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No {csv_type} CSV files found in {input_dir}")
        return False

    print(f"Found {len(csv_files)} {csv_type} CSV files")

    # Parse all CSV files
    all_data = {}
    all_metric_names = set()

    for csv_file in sorted(csv_files):
        kernel = extract_kernel_name(csv_file)
        print(f"  Parsing {kernel}...")

        metrics = parse_ncu_csv(csv_file)
        if metrics:
            all_data[kernel] = metrics
            all_metric_names.update(metrics.keys())
        else:
            print(f"    Warning: No metrics found for {kernel}")

    if not all_data:
        print("No metrics found in any CSV files")
        return False

    # Sort metric names for consistent column order
    metric_names = sorted(all_metric_names)

    # Write combined CSV
    print(f"\nWriting combined CSV to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['kernel'] + metric_names)

        # Data rows
        for kernel in sorted(all_data.keys()):
            row = [kernel]
            for metric in metric_names:
                value = all_data[kernel].get(metric, '')
                row.append(value)
            writer.writerow(row)

    print(f"Success! Combined data for {len(all_data)} kernels")
    print(f"Total metrics: {len(metric_names)}")

    return True

def main():
    # Default paths
    input_dir = 'data/profiling_4070/csv_exports'
    output_summary = 'data/profiling_4070/all_kernels_summary.csv'
    output_details = 'data/profiling_4070/all_kernels_details.csv'

    # Override with command line args if provided
    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_summary = sys.argv[2]

    print("=" * 60)
    print("Combining NCU Profiling Results")
    print("=" * 60)
    print()

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print("Run scripts/export_ncu_metrics.sh first to generate CSV files.")
        sys.exit(1)

    # Combine summary CSVs
    print("Processing summary files...")
    print("-" * 60)
    success_summary = combine_all_csvs(input_dir, output_summary, 'summary')

    print()
    print("-" * 60)
    print()

    # Combine details CSVs
    print("Processing details files...")
    print("-" * 60)
    success_details = combine_all_csvs(input_dir, output_details, 'details')

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    if success_summary:
        print(f"\nSummary CSV: {output_summary}")
        print(f"  View: cat {output_summary}")

    if success_details:
        print(f"\nDetails CSV: {output_details}")
        print(f"  View: cat {output_details}")

    print()
    print("You can now analyze these files with:")
    print("  - pandas: pd.read_csv('all_kernels_summary.csv')")
    print("  - Excel: Open the CSV file")
    print("  - Command line: column -t -s, < all_kernels_summary.csv | less -S")
    print()

if __name__ == '__main__':
    main()

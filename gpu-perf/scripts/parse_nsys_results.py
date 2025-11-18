#!/usr/bin/env python3
"""
Parse nsys (NVIDIA Nsight Systems) profiling results and generate summary reports.
Usage: python3 scripts/parse_nsys_results.py <profiling_dir> [output.csv]
"""

import sys
import os
import csv
import glob
import re
from pathlib import Path

def parse_nsys_stats(stats_file):
    """Parse nsys statistics text file."""
    metrics = {}

    if not os.path.exists(stats_file):
        return metrics

    try:
        with open(stats_file, 'r') as f:
            content = f.read()

            # Extract CUDA Kernel Statistics
            kernel_section = re.search(r'CUDA Kernel Statistics.*?Time \(%\)\s+Total Time \(ns\)\s+Instances\s+Avg \(ns\)\s+Med \(ns\)\s+Min \(ns\)\s+Max \(ns\)\s+StdDev \(ns\)\s+Name\s+-+\s+(.*?)(?:\n\n|\Z)', content, re.DOTALL)
            if kernel_section:
                kernel_lines = kernel_section.group(1).strip().split('\n')
                for line in kernel_lines:
                    if line.strip() and not line.startswith('-'):
                        parts = line.split()
                        if len(parts) >= 9:
                            try:
                                metrics['time_pct'] = float(parts[0])
                                metrics['total_time_ns'] = float(parts[1].replace(',', ''))
                                metrics['instances'] = int(parts[2].replace(',', ''))
                                metrics['avg_time_ns'] = float(parts[3].replace(',', ''))
                                metrics['med_time_ns'] = float(parts[4].replace(',', ''))
                                metrics['min_time_ns'] = float(parts[5].replace(',', ''))
                                metrics['max_time_ns'] = float(parts[6].replace(',', ''))
                                metrics['stddev_ns'] = float(parts[7].replace(',', ''))
                                metrics['kernel_name'] = ' '.join(parts[8:])
                                break  # Take first kernel (main one)
                            except (ValueError, IndexError):
                                continue

            # Extract CUDA Memory Operations
            mem_section = re.search(r'CUDA Memory Operation Statistics \(by time\).*?Time \(%\)\s+Total Time \(ns\)\s+Count\s+Avg \(ns\)\s+Med \(ns\)\s+Min \(ns\)\s+Max \(ns\)\s+StdDev \(ns\)\s+Operation\s+-+\s+(.*?)(?:\n\n|\Z)', content, re.DOTALL)
            if mem_section:
                mem_lines = mem_section.group(1).strip().split('\n')
                htod_time = 0
                dtoh_time = 0
                dtod_time = 0

                for line in mem_lines:
                    if 'HtoD' in line or '[CUDA memcpy HtoD]' in line:
                        parts = line.split()
                        try:
                            htod_time += float(parts[1].replace(',', ''))
                        except (ValueError, IndexError):
                            pass
                    elif 'DtoH' in line or '[CUDA memcpy DtoH]' in line:
                        parts = line.split()
                        try:
                            dtoh_time += float(parts[1].replace(',', ''))
                        except (ValueError, IndexError):
                            pass
                    elif 'DtoD' in line or '[CUDA memcpy DtoD]' in line:
                        parts = line.split()
                        try:
                            dtod_time += float(parts[1].replace(',', ''))
                        except (ValueError, IndexError):
                            pass

                if htod_time > 0:
                    metrics['htod_time_ns'] = htod_time
                if dtoh_time > 0:
                    metrics['dtoh_time_ns'] = dtoh_time
                if dtod_time > 0:
                    metrics['dtod_time_ns'] = dtod_time

            # Extract CUDA API Statistics
            api_section = re.search(r'CUDA API Statistics.*?Time \(%\)\s+Total Time \(ns\)\s+Num Calls\s+Avg \(ns\)\s+Med \(ns\)\s+Min \(ns\)\s+Max \(ns\)\s+StdDev \(ns\)\s+Name\s+-+\s+(.*?)(?:\n\n|\Z)', content, re.DOTALL)
            if api_section:
                api_lines = api_section.group(1).strip().split('\n')
                for line in api_lines:
                    if 'cudaLaunch' in line or 'cudaMemcpy' in line:
                        parts = line.split()
                        if len(parts) >= 9:
                            try:
                                api_name = parts[-1]
                                if 'cudaLaunch' in api_name:
                                    metrics['launch_overhead_ns'] = float(parts[3].replace(',', ''))
                            except (ValueError, IndexError):
                                pass

    except Exception as e:
        print(f"Warning: Could not parse {stats_file}: {e}", file=sys.stderr)

    return metrics

def parse_nsys_csv(csv_file, report_type='cuda_gpu_kern_sum'):
    """Parse nsys CSV export files."""
    metrics = {}

    if not os.path.exists(csv_file):
        return metrics

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                if report_type == 'cuda_gpu_kern_sum':
                    # Kernel summary metrics
                    for key, value in row.items():
                        if key and value:
                            clean_key = key.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
                            try:
                                metrics[clean_key] = float(value.replace(',', ''))
                            except (ValueError, TypeError):
                                metrics[clean_key] = value.strip()

                elif report_type == 'cuda_gpu_mem_size_sum':
                    # Memory operation size metrics
                    for key, value in row.items():
                        if key and value and 'size' in key.lower():
                            clean_key = f"mem_{key.strip().lower().replace(' ', '_')}"
                            try:
                                metrics[clean_key] = float(value.replace(',', ''))
                            except (ValueError, TypeError):
                                metrics[clean_key] = value.strip()

                # Only take first row for summary
                break

    except Exception as e:
        print(f"Warning: Could not parse {csv_file}: {e}", file=sys.stderr)

    return metrics

def extract_key_metrics(stats_metrics, csv_metrics):
    """Extract the most important metrics from nsys results."""

    key_metrics = {}

    # From stats file
    important_stats = {
        'avg_time_ns': 'kernel_avg_ns',
        'total_time_ns': 'kernel_total_ns',
        'instances': 'kernel_instances',
        'stddev_ns': 'kernel_stddev_ns',
        'htod_time_ns': 'htod_time_ns',
        'dtoh_time_ns': 'dtoh_time_ns',
        'dtod_time_ns': 'dtod_time_ns',
        'launch_overhead_ns': 'launch_overhead_ns',
    }

    for metric_key, output_name in important_stats.items():
        if metric_key in stats_metrics:
            key_metrics[output_name] = stats_metrics[metric_key]

    # From CSV exports
    # These vary by nsys version, so we'll take what we can get
    for key, value in csv_metrics.items():
        if isinstance(value, (int, float)):
            key_metrics[key] = value

    return key_metrics

def analyze_profiling_dir(profiling_dir):
    """Analyze all nsys profiling results in a directory."""

    results = {}

    # Pattern for nsys stats files
    stats_pattern = os.path.join(profiling_dir, 'nsys_*_stats.txt')
    stats_files = glob.glob(stats_pattern)

    print(f"Found {len(stats_files)} nsys profiling result files")

    for stats_file in sorted(stats_files):
        # Extract kernel name from filename
        basename = os.path.basename(stats_file)
        kernel = basename.replace('nsys_', '').replace('_stats.txt', '')

        print(f"  Parsing {kernel}...")

        try:
            # Parse stats file
            stats_metrics = parse_nsys_stats(stats_file)

            # Try to find corresponding CSV files
            csv_kern_file = os.path.join(profiling_dir, f'nsys_{kernel}_report_cuda_gpu_kern_sum.csv')
            csv_mem_file = os.path.join(profiling_dir, f'nsys_{kernel}_report_cuda_gpu_mem_size_sum.csv')

            csv_metrics = {}
            if os.path.exists(csv_kern_file):
                csv_metrics.update(parse_nsys_csv(csv_kern_file, 'cuda_gpu_kern_sum'))
            if os.path.exists(csv_mem_file):
                csv_metrics.update(parse_nsys_csv(csv_mem_file, 'cuda_gpu_mem_size_sum'))

            # Extract key metrics
            key_metrics = extract_key_metrics(stats_metrics, csv_metrics)

            results[kernel] = {
                'all_metrics': {**stats_metrics, **csv_metrics},
                'key_metrics': key_metrics
            }

            # Debug: print how many metrics were found
            total_metrics = len(stats_metrics) + len(csv_metrics)
            if total_metrics > 0:
                print(f"    Found {total_metrics} total metrics, {len(key_metrics)} key metrics")
            else:
                print(f"    Warning: No metrics found in files")

        except Exception as e:
            print(f"    ERROR: Failed to parse {kernel}: {e}", file=sys.stderr)
            continue

    return results

def write_summary_csv(results, output_file):
    """Write summary of key metrics to CSV."""

    if not results:
        print("No results to write")
        return

    # Collect all unique metric names
    all_metric_names = set()
    for kernel_data in results.values():
        all_metric_names.update(kernel_data['key_metrics'].keys())

    metric_names = sorted(all_metric_names)

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['kernel'] + metric_names)

        # Data rows
        for kernel in sorted(results.keys()):
            row = [kernel]
            for metric in metric_names:
                value = results[kernel]['key_metrics'].get(metric, '')
                row.append(value)
            writer.writerow(row)

    print(f"\nSummary written to: {output_file}")

def print_summary(results):
    """Print a human-readable summary."""

    print("\n" + "="*80)
    print("NSYS PROFILING SUMMARY")
    print("="*80)

    for kernel in sorted(results.keys()):
        print(f"\n{kernel}:")
        print("-" * 40)

        key_metrics = results[kernel]['key_metrics']

        if not key_metrics:
            print("  No key metrics found")
            continue

        for metric, value in sorted(key_metrics.items()):
            if isinstance(value, float):
                print(f"  {metric:30s}: {value:>15.4f}")
            elif isinstance(value, int):
                print(f"  {metric:30s}: {value:>15d}")
            else:
                print(f"  {metric:30s}: {value:>15}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profiling_dir> [output.csv]")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} data/profiling_2080")
        print(f"  {sys.argv[0]} data/profiling_2080 nsys_summary.csv")
        sys.exit(1)

    profiling_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Analyzing nsys profiling results in: {profiling_dir}")
    print()

    # Analyze
    results = analyze_profiling_dir(profiling_dir)

    if not results:
        print("No nsys profiling results found!")
        print("Make sure you have nsys_*_stats.txt files in the directory.")
        sys.exit(1)

    # Print summary
    print_summary(results)

    # Write CSV if requested
    if output_file:
        write_summary_csv(results, output_file)
    else:
        # Default output filename
        default_output = os.path.join(profiling_dir, 'profiling_summary_nsys.csv')
        write_summary_csv(results, default_output)

if __name__ == '__main__':
    main()

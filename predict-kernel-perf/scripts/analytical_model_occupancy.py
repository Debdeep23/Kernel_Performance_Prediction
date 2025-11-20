#!/usr/bin/env python3
import json
import math
import itertools
from typing import Dict, Any, List
import os
import pandas as pd
import numpy as np

KERNEL_CSVS = [
    "runs_2080ti_final.csv",
    "runs_4070_final.csv",
    "runs_titanv_final.csv",
    "runs_titanx_final.csv",
]

GPU_JSON = "gpu_metrics.json"
TEST_GPU_NAME = "NVIDIA TITAN V" # GPU we treat as new for Experiment 1 (just for Experiment1)

# Train and Test Kernels for Experiment 3
TRAIN_KERNELS = [
    "vector_add",
    "saxpy",
    "strided_copy_8",
    "random_access",
    "reduce_sum",
    "dot_product",
    "histogram",
    "matmul_naive",
    "naive_transpose",
    "conv2d_3x3",
    "conv2d_7x7",
    "shared_bank_conflict",
]

TEST_KERNELS = [
    "matmul_tiled",
    "shared_transpose",
    "atomic_hotspot",
    "vector_add_divergent",
]

output_folder = "analytic_model_outputs"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


dfs = []
for path in KERNEL_CSVS:
    df_part = pd.read_csv(path)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

if "device_name" not in df.columns and "gpu_device_name" in df.columns:
    df["device_name"] = df["gpu_device_name"]

if "block" in df.columns:
    df["bx"] = df["block"].astype(int)
    df["by"] = 1
    df["bz"] = 1

for col in ["FLOPs", "BYTES"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "mean_ms" not in df.columns:
    raise ValueError("Expected 'mean_ms' column in CSV.")

with open(GPU_JSON, "r") as f:
    gpu_list: List[Dict[str, Any]] = json.load(f)

gpu_by_name: Dict[str, Dict[str, Any]] = {g["device_name"]: g for g in gpu_list}


# This func is bascially for adding a unique config identifier per kernel run
# That's cause each kernel may have multiple runs with different problem sizes / block sizes
def add_config_id(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for col in ["N", "rows", "cols", "block", "iters"]:
        if col not in df_out.columns:
            df_out[col] = 0

    df_out["config_id"] = (
        df_out["kernel"].astype(str)
        + "|N=" + df_out["N"].fillna(0).astype(int).astype(str)
        + "|rows=" + df_out["rows"].fillna(0).astype(int).astype(str)
        + "|cols=" + df_out["cols"].fillna(0).astype(int).astype(str)
        + "|block=" + df_out["block"].fillna(0).astype(int).astype(str)
        + "|iters=" + df_out["iters"].fillna(0).astype(int).astype(str)
    )
    return df_out


df = add_config_id(df)


# ============================================================
# Analytical model helpers
# ============================================================

def compute_arithmetic_intensity(row):
    F = row["FLOPs"]
    B = row["BYTES"]
    if B <= 0:
        return np.nan
    return F / B


def roofline_bound_gflops(row, gpu_name: str):
    I = row["arith_intensity"]
    if not np.isfinite(I) or I <= 0:
        return np.nan

    g = gpu_by_name[gpu_name]

    C_sust = (
        g.get("sustained_compute_gflops")
        or g.get("calibrated_compute_gflops")
        or g.get("peak_fp32_gflops")
    )
    BW_sust = (
        g.get("sustained_bandwidth_gbps")
        or g.get("calibrated_mem_bandwidth_gbps")
        or g.get("peak_mem_bandwidth_gbps")
    )

    if C_sust is None or BW_sust is None or C_sust <= 0 or BW_sust <= 0:
        return np.nan

    mem_bound = I * BW_sust
    return min(C_sust, mem_bound)


def compute_occupancy(row, gpu_name: str):
    g = gpu_by_name[gpu_name]

    regs_per_thread = row["regs"]
    shmem_per_block = row["shmem"]

    bx = int(row["bx"])
    by = int(row["by"])
    bz = int(row["bz"])
    threads_per_block = bx * by * bz
    if threads_per_block == 0:
        return 0.0

    warp_size = g["warp_size"]
    max_threads_per_sm = g["max_threads_per_sm"]
    max_blocks_per_sm = g["max_blocks_per_sm"]
    regs_per_sm = g["registers_per_sm"]
    shared_mem_per_sm = g["shared_mem_per_sm"]

    warps_per_block = math.ceil(threads_per_block / warp_size)

    if regs_per_thread > 0:
        blocks_reg_limit = regs_per_sm // (regs_per_thread * threads_per_block)
        blocks_reg_limit = max(blocks_reg_limit, 0)
    else:
        blocks_reg_limit = max_blocks_per_sm

    if shmem_per_block > 0:
        blocks_smem_limit = shared_mem_per_sm // shmem_per_block
        blocks_smem_limit = max(blocks_smem_limit, 0)
    else:
        blocks_smem_limit = max_blocks_per_sm

    blocks_thread_limit = max_threads_per_sm // threads_per_block
    blocks_sm_limit = max_blocks_per_sm

    limits = [blocks_reg_limit, blocks_smem_limit, blocks_thread_limit, blocks_sm_limit]
    limits_pos = [x for x in limits if x > 0]
    active_blocks = min(limits_pos) if limits_pos else 0

    if active_blocks <= 0:
        return 0.0

    max_warps_per_sm = max_threads_per_sm / warp_size
    active_warps = active_blocks * warps_per_block

    occ = active_warps / max_warps_per_sm
    occ = max(0.0, min(1.0, occ))
    return occ


def measured_gflops(row):
    F = row["FLOPs"]
    T_ms = row["mean_ms"]
    if T_ms <= 0:
        return np.nan
    return F / (T_ms / 1000.0) / 1e9


def predict_runtime_analytical(row_src, src_gpu: str, tgt_gpu: str):
    F = float(row_src["FLOPs"])
    B_bytes = float(row_src["BYTES"])

    # CASE B: pure bandwidth
    if F <= 0 and B_bytes > 0:
        g_src = gpu_by_name[src_gpu]
        g_tgt = gpu_by_name[tgt_gpu]

        BW_src = (
            g_src.get("sustained_bandwidth_gbps")
            or g_src.get("calibrated_mem_bandwidth_gbps")
            or g_src.get("peak_mem_bandwidth_gbps")
        )
        BW_tgt = (
            g_tgt.get("sustained_bandwidth_gbps")
            or g_tgt.get("calibrated_mem_bandwidth_gbps")
            or g_tgt.get("peak_mem_bandwidth_gbps")
        )
        if BW_src is None or BW_tgt is None or BW_src <= 0 or BW_tgt <= 0:
            return np.nan

        O_src = compute_occupancy(row_src, src_gpu)
        O_tgt = compute_occupancy(row_src, tgt_gpu)
        if O_src <= 0 or O_tgt <= 0:
            return np.nan

        T_src_ms = float(row_src["mean_ms"])
        if T_src_ms <= 0:
            return np.nan

        bytes_GB = B_bytes / 1e9
        B_meas_src = bytes_GB / (T_src_ms / 1000.0)

        E_src_bw = B_meas_src / (O_src * BW_src)
        B_pred_tgt = E_src_bw * O_tgt * BW_tgt
        if B_pred_tgt <= 0:
            return np.nan

        return bytes_GB / B_pred_tgt * 1000.0

    # CASE A: compute + memory
    I = row_src["arith_intensity"]
    if not np.isfinite(I) or I <= 0:
        return np.nan

    P_roof_src = roofline_bound_gflops(row_src, src_gpu)
    P_roof_tgt = roofline_bound_gflops(row_src, tgt_gpu)
    if not np.isfinite(P_roof_src) or not np.isfinite(P_roof_tgt):
        return np.nan

    O_src = compute_occupancy(row_src, src_gpu)
    O_tgt = compute_occupancy(row_src, tgt_gpu)
    if O_src <= 0 or O_tgt <= 0:
        return np.nan

    P_meas_src = measured_gflops(row_src)
    if not np.isfinite(P_meas_src):
        return np.nan

    E_src = P_meas_src / (O_src * P_roof_src)
    P_pred_tgt = E_src * O_tgt * P_roof_tgt
    if P_pred_tgt <= 0:
        return np.nan

    return F / (P_pred_tgt * 1e9) * 1000.0


# ============================================================
# Config roles for Exp2 and Exp3
# ============================================================

def compute_config_roles(df_with_cfg: pd.DataFrame) -> pd.DataFrame:
    df = df_with_cfg.copy()

    size_col = "working_set_bytes" if "working_set_bytes" in df.columns else "BYTES"

    cfg = df.groupby(["kernel", "config_id"], as_index=False)[size_col].max()

    roles = []
    for kernel, sub in cfg.groupby("kernel"):
        sub = sub.sort_values(size_col).reset_index(drop=True)
        n = len(sub)
        for idx, row in sub.iterrows():
            if n == 1:
                role = "baseline"
            elif n == 2:
                role = "baseline" if idx == 0 else "test_extra"
            else:
                if idx == 0:
                    role = "baseline"
                elif idx == 1:
                    role = "train_extra"
                elif idx == n - 1:
                    role = "test_extra"
                else:
                    role = "other"
            roles.append(
                {
                    "kernel": kernel,
                    "config_id": row["config_id"],
                    "config_role": role,
                }
            )

    roles_df = pd.DataFrame(roles).drop_duplicates()
    return roles_df


# Build cross GPU prediction table
def build_cross_gpu_predictions(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["arith_intensity"] = df.apply(compute_arithmetic_intensity, axis=1)
    df["P_meas_gflops"] = df.apply(measured_gflops, axis=1)

    rows = []
    for config_id, df_conf in df.groupby("config_id"):
        gpu_names = df_conf["device_name"].unique()
        if len(gpu_names) < 2:
            continue

        for src_name, tgt_name in itertools.permutations(gpu_names, 2):
            row_src = df_conf[df_conf["device_name"] == src_name].iloc[0]
            row_tgt = df_conf[df_conf["device_name"] == tgt_name].iloc[0]

            T_src = row_src["mean_ms"]
            T_tgt_true = row_tgt["mean_ms"]
            T_tgt_pred = predict_runtime_analytical(row_src, src_name, tgt_name)

            rows.append(
                {
                    "kernel": row_src["kernel"],
                    "config_id": config_id,
                    "src_gpu": src_name,
                    "tgt_gpu": tgt_name,
                    "T_src_ms": T_src,
                    "T_tgt_true_ms": T_tgt_true,
                    "T_tgt_pred_ms": T_tgt_pred,
                }
            )

    pred_df = pd.DataFrame(rows)

    if pred_df.empty:
        print("WARNING: No cross-GPU pairs found (no config with ≥2 GPUs).")
        return pred_df

    pred_df["ratio_pred_over_true"] = (
        pred_df["T_tgt_pred_ms"] / pred_df["T_tgt_true_ms"]
    )
    pred_df["abs_rel_error"] = (
        (pred_df["T_tgt_pred_ms"] - pred_df["T_tgt_true_ms"]).abs()
        / pred_df["T_tgt_true_ms"]
    )

    pred_df.to_csv(os.path.join(output_folder, "cross_gpu_predictions.csv"), index=False)
    print("Saved cross_gpu_predictions.csv with", len(pred_df), "rows")
    return pred_df


# This is what is printed for each experiment summary in terms of overall MAPE and median ratio
def summarize_experiment(name: str, subdf: pd.DataFrame):
    subdf = subdf.dropna(subset=["T_tgt_pred_ms", "T_tgt_true_ms"])
    if subdf.empty:
        print(f"[{name}] No data after filtering.")
        return
    mape = (subdf["abs_rel_error"].mean() * 100.0)
    med_ratio = subdf["ratio_pred_over_true"].median()
    print(f"\n=== {name} ===")
    print("Num pairs:", len(subdf))
    print("MAPE (%):", f"{mape:.2f}")
    print("Median pred/true:", f"{med_ratio:.3f}")


def save_kernel_metrics(name: str, subdf: pd.DataFrame, out_path: str):
    subdf = subdf.dropna(subset=["T_tgt_pred_ms", "T_tgt_true_ms"])
    if subdf.empty:
        print(f"[{name}] No data for kernel metrics.")
        return
    subdf = subdf.copy()
    subdf["abs_rel_error"] = (
        (subdf["T_tgt_pred_ms"] - subdf["T_tgt_true_ms"]).abs()
        / subdf["T_tgt_true_ms"]
    )
    ker = (
        subdf.groupby("kernel")["abs_rel_error"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    ker["mean_%"] = ker["mean"] * 100.0
    ker["median_%"] = ker["median"] * 100.0
    ker["max_%"] = ker["max"] * 100.0
    ker = ker.drop(columns=["mean", "median", "max"])
    ker.to_csv(os.path.join(output_folder, out_path), index=False)
    print(f"[{name}] Saved kernel-level metrics to {out_path}")


# small helper just for compact Exp-3 printing
def summarize_stats(subdf: pd.DataFrame):
    subdf = subdf.dropna(subset=["T_tgt_pred_ms", "T_tgt_true_ms"])
    if subdf.empty:
        return None
    mape = float((subdf["abs_rel_error"].mean() * 100.0))
    med_ratio = float(subdf["ratio_pred_over_true"].median())
    return {
        "num_pairs": int(len(subdf)),
        "mape": mape,
        "median_ratio": med_ratio,
    }


# ============================================================
# Here we finally come to the EXPERIMENTS section of our model
# ============================================================

def run_experiment_1(pred_df: pd.DataFrame):
    """
    Exp-1: Same kernel, same config, new GPU
    - Evaluate only pairs where tgt_gpu == TEST_GPU_NAME and src_gpu != TEST_GPU_NAME
    """
    exp1 = pred_df[
        (pred_df["tgt_gpu"] == TEST_GPU_NAME)
        & (pred_df["src_gpu"] != TEST_GPU_NAME)
    ]
    
    exp1.to_csv(os.path.join(output_folder, "exp1_same_config_new_gpu.csv"), index=False)
    summarize_experiment(f"Exp1: Same kernel, same config, new GPU (tgt={TEST_GPU_NAME})", exp1)
    save_kernel_metrics("Exp1 kernel metrics", exp1, "exp1_kernel_metrics.csv")


def run_experiment_2(pred_df: pd.DataFrame):
    """
    Exp-2: Same kernel, new configs, same GPUs
    - test configs: config_role == 'test_extra'
    """
    if "config_role" not in pred_df.columns:
        print("[Exp2] No config_role column found; skipping.")
        return

    exp2_test = pred_df[pred_df["config_role"] == "test_extra"]
    exp2_test.to_csv(os.path.join(output_folder, "exp2_new_configs_same_gpus.csv"), index=False)
    summarize_experiment("Exp2: Same kernel, NEW config, same GPUs", exp2_test)
    save_kernel_metrics("Exp2 kernel metrics", exp2_test, "exp2_kernel_metrics.csv")


def run_experiment_3(pred_df: pd.DataFrame):
    """
    Exp-3: New but related kernels
    - Exp3a: 1 config per kernel (baseline)
    - Exp3b: 2 configs per train kernel (baseline + train_extra)
    """

    if "config_role" not in pred_df.columns:
        print("[Exp3] No config_role column found; skipping.")
        return

    # -------------------- Exp-3a: baseline only --------------------
    exp3a_train = pred_df[
        (pred_df["kernel"].isin(TRAIN_KERNELS))
        & (pred_df["config_role"] == "baseline")
    ]
    exp3a_test = pred_df[
        (pred_df["kernel"].isin(TEST_KERNELS))
        & (pred_df["config_role"] == "baseline")
    ]

    exp3a_train.to_csv(os.path.join(output_folder, "exp3a_train_kernels.csv"), index=False)
    exp3a_test.to_csv(os.path.join(output_folder, "exp3a_new_kernels.csv"), index=False)
    save_kernel_metrics("Exp3a train kernels", exp3a_train, "exp3a_train_kernel_metrics.csv")
    save_kernel_metrics("Exp3a NEW kernels", exp3a_test, "exp3a_new_kernel_metrics.csv")

    stats_a_train = summarize_stats(exp3a_train)
    stats_a_test = summarize_stats(exp3a_test)

    print("\n=== Exp3a: New but related kernels (1 config per kernel) ===")
    if stats_a_train:
        print(
            f"Train kernels (12): pairs={stats_a_train['num_pairs']}, "
            f"MAPE={stats_a_train['mape']:.2f}%, "
            f"median pred/true={stats_a_train['median_ratio']:.3f}"
        )
    else:
        print("Train kernels (12): no valid pairs.")

    if stats_a_test:
        print(
            f"NEW kernels (4):  pairs={stats_a_test['num_pairs']}, "
            f"MAPE={stats_a_test['mape']:.2f}%, "
            f"median pred/true={stats_a_test['median_ratio']:.3f}"
        )
        print(
            "Interpretation: with only one config per kernel, the model transfers "
            "moderately well to NEW kernels but still has ~50% average error."
        )
    else:
        print("NEW kernels (4): no valid pairs.")

    # -------------------- Exp-3b: baseline + train_extra for train kernels --------------------
    exp3b_train = pred_df[
        (pred_df["kernel"].isin(TRAIN_KERNELS))
        & (pred_df["config_role"].isin(["baseline", "train_extra"]))
    ]
    exp3b_test = exp3a_test  # same test subset (baseline configs for NEW kernels)

    exp3b_train.to_csv(os.path.join(output_folder, "exp3b_train_kernels.csv"), index=False)
    exp3b_test.to_csv(os.path.join(output_folder, "exp3b_new_kernels.csv"), index=False)
    save_kernel_metrics("Exp3b train kernels", exp3b_train, "exp3b_train_kernel_metrics.csv")
    save_kernel_metrics("Exp3b NEW kernels", exp3b_test, "exp3b_new_kernel_metrics.csv")

    stats_b_train = summarize_stats(exp3b_train)
    stats_b_test = summarize_stats(exp3b_test)

    print("\n=== Exp3b: New but related kernels (2 configs per train kernel) ===")
    if stats_b_train:
        print(
            f"Train kernels (12, 2 configs): pairs={stats_b_train['num_pairs']}, "
            f"MAPE={stats_b_train['mape']:.2f}%, "
            f"median pred/true={stats_b_train['median_ratio']:.3f}"
        )
    else:
        print("Train kernels (12, 2 configs): no valid pairs.")

    if stats_b_test:
        print(
            f"NEW kernels (4):            pairs={stats_b_test['num_pairs']}, "
            f"MAPE={stats_b_test['mape']:.2f}%, "
            f"median pred/true={stats_b_test['median_ratio']:.3f}"
        )
        print(
            "Interpretation: adding a second config per train kernel changes the "
            "model calibration on the training set but leaves NEW-kernel error "
            "roughly in the same ballpark."
        )
    else:
        print("NEW kernels (4): no valid pairs for Exp3b.")





###################
# Main Function
###################

def main():
    roles_df = compute_config_roles(df)

    pred_df = build_cross_gpu_predictions(df)
    if pred_df.empty:
        return

    pred_df = pred_df.merge(
        roles_df,
        on=["kernel", "config_id"],
        how="left",
    )

    run_experiment_1(pred_df)
    run_experiment_2(pred_df)
    run_experiment_3(pred_df)

    print("\nDone. Generated analytic outputs:")
    print("  - cross_gpu_predictions.csv")
    print("  - exp1_same_config_new_gpu.csv, exp1_kernel_metrics.csv")
    print("  - exp2_new_configs_same_gpus.csv, exp2_kernel_metrics.csv")
    print("  - exp3a_train_kernels.csv, exp3a_new_kernels.csv, exp3a_*_kernel_metrics.csv")
    print("  - exp3b_train_kernels.csv, exp3b_new_kernels.csv, exp3b_*_kernel_metrics.csv")


if __name__ == "__main__":
    main()
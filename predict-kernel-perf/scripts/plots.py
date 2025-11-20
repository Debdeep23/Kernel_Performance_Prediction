import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

analytic_dir = "analytic_model_results/"
ml_dir = "ml_outputs/"

ANALYTIC_FILES = {
    "cross": f"{os.path.join(analytic_dir, 'cross_gpu_predictions.csv')}",
    "exp1_pairs": f"{os.path.join(analytic_dir, 'exp1_same_config_new_gpu.csv')}",
    "exp1_kernels": f"{os.path.join(analytic_dir, 'exp1_kernel_metrics.csv')}",
    "exp2_pairs": f"{os.path.join(analytic_dir, 'exp2_new_configs_same_gpus.csv')}",
    "exp2_kernels": f"{os.path.join(analytic_dir, 'exp2_kernel_metrics.csv')}",
    "exp3a_train_pairs": f"{os.path.join(analytic_dir, 'exp3a_train_kernels.csv')}",
    "exp3a_train_kernels": f"{os.path.join(analytic_dir, 'exp3a_train_kernel_metrics.csv')}",
    "exp3a_new_pairs": f"{os.path.join(analytic_dir, 'exp3a_new_kernels.csv')}",
    "exp3a_new_kernels": f"{os.path.join(analytic_dir, 'exp3a_new_kernel_metrics.csv')}",
    "exp3b_train_pairs": f"{os.path.join(analytic_dir, 'exp3b_train_kernels.csv')}",
    "exp3b_train_kernels": f"{os.path.join(analytic_dir, 'exp3b_train_kernel_metrics.csv')}",
    "exp3b_new_pairs": f"{os.path.join(analytic_dir, 'exp3b_new_kernels.csv')}",
    "exp3b_new_kernels": f"{os.path.join(analytic_dir, 'exp3b_new_kernel_metrics.csv')}",
}


ML_FILES = {
    "exp1_pairs": f"{os.path.join(ml_dir, 'exp1_same_config_new_gpu_ml_predictions.csv')}",
    "exp1_kernels": f"{os.path.join(ml_dir, 'exp1_same_config_new_gpu_kernel_metrics_ml.csv')}",
    "exp2_pairs": f"{os.path.join(ml_dir, 'exp2_new_configs_same_gpus_ml_predictions.csv')}",
    "exp2_kernels": f"{os.path.join(ml_dir, 'exp2_new_configs_same_gpus_kernel_metrics_ml.csv')}",
    "exp3_pairs": f"{os.path.join(ml_dir, 'exp3_new_kernels_ml_predictions.csv')}",
    "exp3_kernels": f"{os.path.join(ml_dir, 'exp3_new_kernels_kernel_metrics_ml.csv')}",
}


GPU_2080 = "NVIDIA GeForce RTX 2080 Ti"
GPU_4070 = "NVIDIA GeForce RTX 4070"
GPU_TITANV = "NVIDIA TITAN V"


PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


analytic = {k: load_csv_safe(v) for k, v in ANALYTIC_FILES.items()}
ml = {k: load_csv_safe(v) for k, v in ML_FILES.items()}

cross = analytic["cross"]

if "abs_rel_error" not in cross.columns:
    cross["abs_rel_error"] = (
        (cross["T_tgt_pred_ms"] - cross["T_tgt_true_ms"]).abs()
        / cross["T_tgt_true_ms"]
    )

# ============================================================
# 1) GPU-wise performance (analytic + ML)
# ============================================================

def gpu_pair_mape(df: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    """
    Compute mean MAPE per (src_gpu, tgt_gpu) pair.
    """
    df = df.dropna(subset=[pred_col, "T_tgt_true_ms"]).copy()
    if df.empty:
        print(f"[{label}] No data.")
        return pd.DataFrame()

    err = (df[pred_col] - df["T_tgt_true_ms"]).abs() / df["T_tgt_true_ms"]
    df["_rel_error_"] = err

    agg = (
        df.groupby(["src_gpu", "tgt_gpu"])["_rel_error_"]
        .mean()
        .reset_index()
        .rename(columns={"_rel_error_": "MAPE"})
    )
    agg["MAPE_%"] = agg["MAPE"] * 100.0
    print(f"\n[{label}] GPU-pair MAPE (sorted):")
    print(agg.sort_values("MAPE_%"))
    return agg


analytic_gpu_pairs = gpu_pair_mape(
    cross, pred_col="T_tgt_pred_ms", label="Analytic model"
)

# ML pair data: only in exp1/exp2/exp3 ml prediction files
def concat_ml_pairs(ml_dict: dict) -> pd.DataFrame:
    dfs = []
    for key in ["exp1_pairs", "exp2_pairs", "exp3_pairs"]:
        if key in ml_dict and not ml_dict[key].empty:
            dfs.append(ml_dict[key])
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

ml_pairs_all = concat_ml_pairs(ml)
if not ml_pairs_all.empty and "T_tgt_pred_ms_ml" in ml_pairs_all.columns:
    ml_gpu_pairs = gpu_pair_mape(
        ml_pairs_all, pred_col="T_tgt_pred_ms_ml", label="ML model"
    )
else:
    ml_gpu_pairs = pd.DataFrame()
    print("[ML] No combined pair data with T_tgt_pred_ms_ml found.")


# ---- Simple bar plot of GPU pair MAPE (analytic) ----
def plot_gpu_pair_mape_bar(agg: pd.DataFrame, title: str, filename: str):
    if agg.empty:
        return
    
    # Keep only rows with MAPE < 100%
    agg = agg[agg["MAPE_%"] < 100]

    if agg.empty:
        return
    
    # Sort by MAPE
    agg = agg.sort_values("MAPE_%")

    labels = [f"{s} → {t}" for s, t in zip(agg["src_gpu"], agg["tgt_gpu"])]
    values = agg["MAPE_%"].values

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values)
    plt.xlabel("MAPE (%)")
    plt.title(title)

    # Add MAPE text labels next to bars
    for bar, value in zip(bars, values):
        plt.text(
            value + 1,                         # position x
            bar.get_y() + bar.get_height()/2,  # position y
            f"{value:.1f}%",                   # label text
            va="center"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

plot_gpu_pair_mape_bar(
    analytic_gpu_pairs,
    title="Analytic model: MAPE per GPU pair",
    filename="analytic_gpu_pair_mape_bar.png",
)

if not ml_gpu_pairs.empty:
    plot_gpu_pair_mape_bar(
        ml_gpu_pairs,
        title="ML model: MAPE per GPU pair",
        filename="ml_gpu_pair_mape_bar.png",
    )


def rank_kernels_from_pairs(df: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    """
    Compute per-kernel MAPE from a pair-level dataframe.
    """
    df = df.dropna(subset=[pred_col, "T_tgt_true_ms"])
    if df.empty:
        print(f"[{label}] No data for kernel ranking.")
        return pd.DataFrame()

    rel_error = (df[pred_col] - df["T_tgt_true_ms"]).abs() / df["T_tgt_true_ms"]
    df = df.copy()
    df["_rel_error_"] = rel_error

    ker = (
        df.groupby("kernel")["_rel_error_"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    ker["MAPE_%"] = ker["mean"] * 100.0
    ker["MedAPE_%"] = ker["median"] * 100.0
    ker["MaxAPE_%"] = ker["max"] * 100.0
    ker = ker.drop(columns=["mean", "median", "max"])

    print(f"\n[{label}] Kernels ranked easiest → hardest (by MAPE):")
    print(ker.sort_values("MAPE_%"))

    return ker.sort_values("MAPE_%")


analytic_kernel_rank = rank_kernels_from_pairs(
    cross, pred_col="T_tgt_pred_ms", label="Analytic model (all pairs)"
)

if not ml_pairs_all.empty and "T_tgt_pred_ms_ml" in ml_pairs_all.columns:
    ml_kernel_rank = rank_kernels_from_pairs(
        ml_pairs_all, pred_col="T_tgt_pred_ms_ml", label="ML model (all pairs)"
    )
else:
    ml_kernel_rank = pd.DataFrame()

#  Bar plot for easiest kernels
def plot_kernel_rank_bar(ker_df: pd.DataFrame, title: str, filename: str, top_n: int = 3):
    if ker_df.empty:
        return
    top = ker_df.sort_values("MAPE_%").head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top["kernel"], top["MAPE_%"])
    plt.xlabel("MAPE (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

plot_kernel_rank_bar(
    analytic_kernel_rank,
    title="Analytic model: Avg. Lowest MAPE kernels for all GPUs",
    filename="analytic_easiest_kernels_bar.png",
    top_n=4
)

if not ml_kernel_rank.empty:
    plot_kernel_rank_bar(
        ml_kernel_rank,
        title="ML model: Avg. Lowest MAPE kernels for all GPUs",
        filename="ml_easiest_kernels_bar.png",
        top_n=3,
    )

# plot_kernel_rank_bar(
#     analytic_kernel_rank.sort_values("MAPE_%", ascending=False),
#     title="Analytic model: Hardest kernels (highest MAPE)",
#     filename="analytic_hardest_kernels_bar.png",
# )


# ============================================================
# 3) Compare 4070→TitanV vs 2080Ti→TitanV (analytic + ML)
# ============================================================

def compare_specific_pairs(df: pd.DataFrame, pred_col: str, label: str):
    df = df.dropna(subset=[pred_col, "T_tgt_true_ms"]).copy()

    mask_4070 = (df["src_gpu"] == GPU_4070) & (df["tgt_gpu"] == GPU_TITANV)
    mask_2080 = (df["src_gpu"] == GPU_2080) & (df["tgt_gpu"] == GPU_TITANV)

    df_4070 = df[mask_4070]
    df_2080 = df[mask_2080]

    def summarize(sub, pair_label):
        if sub.empty:
            print(f"[{label}] {pair_label}: no data")
            return
        rel = (sub[pred_col] - sub["T_tgt_true_ms"]).abs() / sub["T_tgt_true_ms"]
        print(f"\n[{label}] {pair_label} → Titan V")
        print(f"Num pairs: {len(sub)}")
        print(f"MAPE: {rel.mean() * 100:.2f}%")
        print(f"Median error: {np.median(rel) * 100:.2f}%")

    summarize(df_4070, "4070")
    summarize(df_2080, "2080Ti")

    # Simple scatter per pair (pred vs true)
    for name, sub in [("4070", df_4070), ("2080Ti", df_2080)]:
        if sub.empty:
            continue
        true = sub["T_tgt_true_ms"].values
        pred = sub[pred_col].values
        plt.figure(figsize=(5, 5))
        plt.scatter(true, pred)
        plt.plot([true.min(), true.max()], [true.min(), true.max()])
        plt.xlabel("True runtime (ms)")
        plt.ylabel("Predicted runtime (ms)")
        plt.title(f"{label}: {name} → TITAN V")
        plt.tight_layout()
        fname = f"{label.replace(' ', '_').lower()}_{name.lower()}_to_titanv_scatter.png"
        plt.savefig(os.path.join(PLOT_DIR, fname))
        plt.close()


compare_specific_pairs(
    cross, pred_col="T_tgt_pred_ms", label="Analytic model"
)

if not ml_pairs_all.empty and "T_tgt_pred_ms_ml" in ml_pairs_all.columns:
    compare_specific_pairs(
        ml_pairs_all, pred_col="T_tgt_pred_ms_ml", label="ML model"
    )


print(f"\nAll plots saved to: {os.path.abspath(PLOT_DIR)}")

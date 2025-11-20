import pandas as pd
import numpy as np
import math
import os


ANALYTIC_DIR = "analytic_model_outputs/"
ML_DIR = "ml_outputs/"

ANALYTIC_FILES = {
    "exp1_pairs": os.path.join(ANALYTIC_DIR, "exp1_same_config_new_gpu.csv"),
    "exp2_pairs": os.path.join(ANALYTIC_DIR, "exp2_new_configs_same_gpus.csv"),
    "exp3a_train_pairs": os.path.join(ANALYTIC_DIR, "exp3a_train_kernels.csv"),
    "exp3a_new_pairs": os.path.join(ANALYTIC_DIR, "exp3a_new_kernels.csv"),
    "exp3b_train_pairs": os.path.join(ANALYTIC_DIR, "exp3b_train_kernels.csv"),
    "exp3b_new_pairs": os.path.join(ANALYTIC_DIR, "exp3b_new_kernels.csv"),
}

ML_FILES = {
    "exp1_pairs": os.path.join(ML_DIR, "exp1_same_config_new_gpu_ml_predictions.csv"),
    "exp2_pairs": os.path.join(ML_DIR, "exp2_new_configs_same_gpus_ml_predictions.csv"),
    "exp3_pairs": os.path.join(ML_DIR, "exp3_new_kernels_ml_predictions.csv"),
}

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_df(df: pd.DataFrame, pred_col: str) -> dict:
    """
    Compute intuitive metrics for one experiment:
      - num pairs
      - MAPE %
      - median pred/true
      - MAE (ms)
      - RMSE (ms)
      - % within 10 / 25 / 50 % error
    """
    df = df.dropna(subset=[pred_col, "T_tgt_true_ms"]).copy()
    if df.empty:
        return {
            "NumPairs": 0,
            "MAPE_%": np.nan,
            "Med_pred/true": np.nan,
            "MAE_ms": np.nan,
            "RMSE_ms": np.nan,
            "Within10_%": np.nan,
            "Within25_%": np.nan,
            "Within50_%": np.nan,
        }

    true = df["T_tgt_true_ms"].values
    pred = df[pred_col].values

    rel_err = np.abs(pred - true) / true
    ratios = pred / true

    mape = rel_err.mean() * 100.0
    med_ratio = np.median(ratios)
    mae = np.mean(np.abs(pred - true))
    rmse = math.sqrt(np.mean((pred - true) ** 2))

    within_10 = (rel_err < 0.10).mean() * 100.0
    within_25 = (rel_err < 0.25).mean() * 100.0
    within_50 = (rel_err < 0.50).mean() * 100.0

    return {
        "NumPairs": len(df),
        "MAPE_%": mape,
        "Med_pred/true": med_ratio,
        "MAE_ms": mae,
        "RMSE_ms": rmse,
        "Within10_%": within_10,
        "Within25_%": within_25,
        "Within50_%": within_50,
    }


def build_analytic_experiment_table() -> pd.DataFrame:
    rows = []

    # --- Exp1: new GPU ---
    exp1 = load_csv(ANALYTIC_FILES["exp1_pairs"])
    if not exp1.empty:
        metrics = summarize_df(exp1, pred_col="T_tgt_pred_ms")
        rows.append({
            "Experiment": "Exp1",
            "Setting": "New GPU (same kernel+config)",
            "Split": "All pairs in Exp1",
            "Model": "Analytic",
            **metrics,
        })

    # --- Exp2: new configs ---
    exp2 = load_csv(ANALYTIC_FILES["exp2_pairs"])
    if not exp2.empty:
        metrics = summarize_df(exp2, pred_col="T_tgt_pred_ms")
        rows.append({
            "Experiment": "Exp2",
            "Setting": "New configs (same kernels & GPUs)",
            "Split": "test_extra configs",
            "Model": "Analytic",
            **metrics,
        })

    # --- Exp3a: new kernels, 1 config per kernel ---
    exp3a_train = load_csv(ANALYTIC_FILES["exp3a_train_pairs"])
    exp3a_new = load_csv(ANALYTIC_FILES["exp3a_new_pairs"])

    if not exp3a_train.empty:
        m_train = summarize_df(exp3a_train, pred_col="T_tgt_pred_ms")
        rows.append({
            "Experiment": "Exp3a",
            "Setting": "Train kernels (1 config/kernel)",
            "Split": "Train kernels",
            "Model": "Analytic",
            **m_train,
        })

    if not exp3a_new.empty:
        m_new = summarize_df(exp3a_new, pred_col="T_tgt_pred_ms")
        rows.append({
            "Experiment": "Exp3a",
            "Setting": "New kernels (1 config/kernel)",
            "Split": "Test kernels",
            "Model": "Analytic",
            **m_new,
        })

    # --- Exp3b: new kernels, 2 configs for train kernels ---
    exp3b_train = load_csv(ANALYTIC_FILES["exp3b_train_pairs"])
    exp3b_new = load_csv(ANALYTIC_FILES["exp3b_new_pairs"])

    if not exp3b_train.empty:
        m_train = summarize_df(exp3b_train, pred_col="T_tgt_pred_ms")
        rows.append({
            "Experiment": "Exp3b",
            "Setting": "Train kernels (2 configs/kernel)",
            "Split": "Train kernels",
            "Model": "Analytic",
            **m_train,
        })

    if not exp3b_new.empty:
        m_new = summarize_df(exp3b_new, pred_col="T_tgt_pred_ms")
        rows.append({
            "Experiment": "Exp3b",
            "Setting": "New kernels (baseline config)",
            "Split": "Test kernels",
            "Model": "Analytic",
            **m_new,
        })

    if not rows:
        print("[Analytic] No experiment data found.")
        return pd.DataFrame()

    df_table = pd.DataFrame(rows)

    for col in ["MAPE_%", "Med_pred/true", "MAE_ms", "RMSE_ms", "Within10_%", "Within25_%", "Within50_%"]:
        df_table[col] = df_table[col].astype(float).round(2)

    return df_table


def build_ml_experiment_table() -> pd.DataFrame:
    rows = []

    # --- Exp1 ML ---
    exp1_ml = load_csv(ML_FILES["exp1_pairs"])
    if not exp1_ml.empty:
        m1 = summarize_df(exp1_ml, pred_col="T_tgt_pred_ms_ml")
        rows.append({
            "Experiment": "Exp1",
            "Setting": "New GPU (same kernel+config)",
            "Split": "tgt = held-out GPU",
            "Model": "ML (RF)",
            **m1,
        })

    # --- Exp2 ML ---
    exp2_ml = load_csv(ML_FILES["exp2_pairs"])
    if not exp2_ml.empty:
        m2 = summarize_df(exp2_ml, pred_col="T_tgt_pred_ms_ml")
        rows.append({
            "Experiment": "Exp2",
            "Setting": "New configs (same kernels & GPUs)",
            "Split": "test_extra configs",
            "Model": "ML (RF)",
            **m2,
        })

    # --- Exp3 ML ---
    exp3_ml = load_csv(ML_FILES["exp3_pairs"])
    if not exp3_ml.empty:
        m3 = summarize_df(exp3_ml, pred_col="T_tgt_pred_ms_ml")
        rows.append({
            "Experiment": "Exp3",
            "Setting": "New kernels",
            "Split": "Test kernels only",
            "Model": "ML (RF)",
            **m3,
        })

    if not rows:
        print("[ML] No experiment data found.")
        return pd.DataFrame()

    df_table = pd.DataFrame(rows)

    for col in ["MAPE_%", "Med_pred/true", "MAE_ms", "RMSE_ms", "Within10_%", "Within25_%", "Within50_%"]:
        df_table[col] = df_table[col].astype(float).round(2)

    return df_table


if __name__ == "__main__":
    analytic_table = build_analytic_experiment_table()
    ml_table = build_ml_experiment_table()

    print("\n=== Analytic model experiment summary ===")
    if not analytic_table.empty:
        print(analytic_table.to_markdown(index=False))
    else:
        print("No data.")

    print("\n=== ML model experiment summary ===")
    if not ml_table.empty:
        print(ml_table.to_markdown(index=False))
    else:
        print("No data.")

    if not analytic_table.empty:
        analytic_table.to_csv("analytic_experiment_summary.csv", index=False)
    if not ml_table.empty:
        ml_table.to_csv("ml_experiment_summary.csv", index=False)
